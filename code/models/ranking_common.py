"""树模型排序实验公共工具；负责样本分组、指标选择和组合输出。"""

import json
import os
import random
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..config import config
from ..features.baseline import date_group_sizes, preprocess_stock_data_samples, preprocess_stock_history_samples
from ..features.windows import feature_num_for_window
from ..portfolio.postprocess import PortfolioConfig, select_portfolio
from ..utils.runtime_split import load_market_data
from ..utils.submission import validate_result_file
from ..utils.validation import build_validation_plan
from .xgboost.loss import average_ranks, rank_ic_score, topk_return_metrics
from .spine import RankDataset, metric_schema


RankData = RankDataset


def set_seed(seed: int) -> None:
    """固定随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def group_ptr_from_sizes(groups: list[int]) -> np.ndarray:
    """由 group size 构造左闭右开边界。"""
    return np.concatenate([[0], np.cumsum(np.asarray(groups, dtype=np.int64))])


def group_ids_from_sizes(groups: list[int]) -> np.ndarray:
    """生成 CatBoost query group_id，要求样本已按日期连续排列。"""
    return np.repeat(np.arange(len(groups), dtype=np.int32), np.asarray(groups, dtype=np.int32))


def relevance_from_returns(labels: np.ndarray, groups: list[int], max_relevance: int) -> np.ndarray:
    """把每个日期截面内的未来收益转成 LambdaRank 可用的非负相关性等级。"""
    if max_relevance < 1:
        raise ValueError('max_relevance must be positive')

    labels = np.asarray(labels, dtype=np.float64)
    relevance = np.zeros(len(labels), dtype=np.int32)
    ptr = group_ptr_from_sizes(groups)
    for begin, end in zip(ptr[:-1], ptr[1:]):
        n = int(end - begin)
        if n <= 1:
            continue
        ranks = average_ranks(labels[begin:end]) - 1.0
        relevance[begin:end] = np.rint(ranks / (n - 1) * max_relevance).astype(np.int32)
    return relevance


def make_rank_data(df: pd.DataFrame, features: list[str], max_relevance: int) -> RankData:
    """按日期 group 构造通用排序矩阵。"""
    grouped_df, groups = date_group_sizes(df, int(config['min_group_size']))
    x = grouped_df[features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    raw_label = grouped_df['label'].to_numpy(dtype=np.float32)
    relevance = relevance_from_returns(raw_label, groups, max_relevance)
    return RankData(grouped_df, x, raw_label, relevance, groups)


def evaluate_predictions(pred: np.ndarray, raw_label: np.ndarray, groups: list[int]) -> tuple[float, dict, dict]:
    """用原始未来收益评估 RankIC 和 TopK 组合收益。"""
    ptr = group_ptr_from_sizes(groups)
    rank_ic = rank_ic_score(pred, raw_label, ptr)
    top5 = topk_return_metrics(pred, raw_label, groups, top_k=5)
    top10 = topk_return_metrics(pred, raw_label, groups, top_k=10)
    return rank_ic, top5, top10


def choose_best_iteration(validation_metric: list[float], metric_name: str) -> dict:
    """选择验证指标最高的迭代；完全相同则取更早轮次。"""
    if not validation_metric:
        raise ValueError(f'empty {metric_name} metric history')
    values = np.asarray(validation_metric, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f'{metric_name} metric history contains non-finite values')
    best_iteration = int(np.argmax(values))
    return {
        'best_iteration': best_iteration,
        'score': float(values[best_iteration]),
        'metric': metric_name,
        'selection_rule': f'max_{metric_name}_earliest_tie',
        'num_iterations': int(len(values)),
    }


def select_best_iteration(
    predict_iteration: Callable[[int], np.ndarray],
    raw_label: np.ndarray,
    groups: list[int],
    num_boost_round: int,
    desc: str,
) -> tuple[dict, list[float]]:
    """逐轮预测验证集，并按比赛绝对 Top5 收益选最佳迭代。"""
    values = []
    for iteration in tqdm(range(num_boost_round), desc=desc, unit='round', dynamic_ncols=True):
        pred = predict_iteration(iteration)
        top5 = topk_return_metrics(pred, raw_label, groups, top_k=5)
        values.append(float(top5['pred_top5_return_avg']))
    selection = choose_best_iteration(values, 'validation_top5_return')
    return selection, values


def target_range(df: pd.DataFrame) -> tuple[str, str]:
    """返回样本目标周日期范围，用于训练日志审计。"""
    dates = pd.to_datetime(df['日期']).dt.date
    return str(dates.min()), str(dates.max())


def model_bindings(family_config: dict) -> list[tuple[str, int, str, str]]:
    """把模型族配置解析成模型名、输入窗口、特征类型、特征编号。"""
    bindings = []
    for name in family_config['model_names']:
        model_config = family_config['model_params'][name]
        input_window = int(model_config['input_window'])
        feature_type = model_config['feature_type']
        feature_num = feature_num_for_window(input_window, feature_type)
        bindings.append((name, input_window, feature_type, feature_num))
    return bindings


def build_train_datasets(validation_plan, holdout_fold, bindings, stockid2idx):
    """按唯一特征配置构造 train/validation/holdout 数据。"""
    datasets = {}
    for input_window, feature_num in sorted({(item[1], item[3]) for item in bindings}):
        train_samples = validation_plan.get_train_samples(holdout_fold, input_window)
        validation_samples = validation_plan.get_validation_samples(holdout_fold, input_window)
        holdout_samples = validation_plan.get_test_samples(input_window)
        train_df, features = preprocess_stock_data_samples(train_samples, feature_num, stockid2idx)
        val_df, _ = preprocess_stock_data_samples(validation_samples, feature_num, stockid2idx)
        holdout_df, _ = preprocess_stock_data_samples(holdout_samples, feature_num, stockid2idx)
        train_df = train_df.dropna(subset=['label']).sort_values(['日期', '股票代码']).reset_index(drop=True)
        val_df = val_df.dropna(subset=['label']).sort_values(['日期', '股票代码']).reset_index(drop=True)
        holdout_df = holdout_df.dropna(subset=['label']).sort_values(['日期', '股票代码']).reset_index(drop=True)
        datasets[(input_window, feature_num)] = (train_df, val_df, holdout_df, features)
    return datasets


def print_data_ranges(raw_df, validation_plan, first_input_window, train_df, val_df, holdout_df) -> None:
    """打印训练、验证、测试和预测窗口。"""
    prediction_samples = validation_plan.get_prediction_samples(first_input_window)
    print(f"Data: {raw_df['日期'].min().date()} to {raw_df['日期'].max().date()} | rows={len(raw_df)} | stocks={raw_df['股票代码'].nunique()}")
    print(f'Train: {target_range(train_df)[0]} to {target_range(train_df)[1]}')
    print(f'Validation: {target_range(val_df)[0]} to {target_range(val_df)[1]}')
    print(f'Holdout Test: {target_range(holdout_df)[0]} to {target_range(holdout_df)[1]}')
    print(
        f"Prediction: target={validation_plan.test_date.date()} | "
        f"input={validation_plan.input_start(validation_plan.test_date, first_input_window).date()} to "
        f"{(validation_plan.test_date - pd.Timedelta(days=3)).date()} | stocks={len(prediction_samples)}"
    )


def write_train_outputs(output_dir: Path, model_type: str, family_config: dict, models, stock_ids, validation_plan, holdout_fold, bindings) -> dict:
    """保存训练元数据和分数文本。"""
    best_model = max(models, key=lambda item: item['validation_score'])
    metadata = {
        'model_type': model_type,
        'model_names': list(family_config['model_names']),
        'models': models,
        'stock_ids': stock_ids,
        'config': config,
        'validation_mode': config['validation']['mode'],
        'num_validation_weeks': validation_plan.num_validation_weeks,
        'num_test_weeks': validation_plan.num_test_weeks,
        'validation_target_start': str(validation_plan.validation_target_start.date()),
        'holdout_test_target_start': str(validation_plan.holdout_test_target_start.date()),
        'test_date': str(validation_plan.test_date.date()),
        'model_windows': [
            {
                'name': name,
                'input_window': input_window,
                'feature_type': feature_type,
                'feature_num': feature_num,
                'validation_input_start': str(validation_plan.input_start(holdout_fold.validation_start, input_window).date()),
                'holdout_test_input_start': str(validation_plan.input_start(validation_plan.holdout_test_target_start, input_window).date()),
                'test_input_start': str(validation_plan.input_start(validation_plan.test_date, input_window).date()),
            }
            for name, input_window, feature_type, feature_num in bindings
        ],
        'best_model': best_model['name'],
        'validation_score_metric': best_model['best_selection_metric'],
        'validation_rank_ic': best_model['validation_rank_ic'],
        'validation_score': best_model['validation_score'],
        'holdout_test_rank_ic': best_model['holdout_test_rank_ic'],
        'holdout_test_top5': best_model['holdout_test_top5'],
        'metric_schema': metric_schema(),
    }
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with open(output_dir / 'final_score.txt', 'w', encoding='utf-8') as f:
        for model in models:
            f.write(f"{model['name']} artifact: {model['artifact_name']}{model['artifact_ext']}\n")
            f.write(f"{model['name']} input_window: {model['input_window']}\n")
            f.write(f"{model['name']} feature_type: {model['feature_type']}\n")
            f.write(f"{model['name']} feature_num: {model['feature_num']}\n")
            f.write(f"{model['name']} best iteration: {model['best_iteration']}\n")
            f.write(f"{model['name']} best selection metric: {model['best_selection_metric']}\n")
            f.write(f"{model['name']} best selection score: {model['best_selection_score']:.8f}\n")
            f.write(f"{model['name']} best selection details: {model['best_selection']}\n")
            f.write(f"{model['name']} validation RankIC: {model['validation_rank_ic']:.8f}\n")
            f.write(f"{model['name']} validation pred top10 return avg: {model['validation_top10']['pred_top10_return_avg']:.8f}\n")
            f.write(f"{model['name']} validation pred top10 excess return avg: {model['validation_top10']['pred_top10_excess_return_avg']:.8f}\n")
            f.write(f"{model['name']} validation pred top10 precision avg: {model['validation_top10']['pred_top10_precision_avg']:.8f}\n")
            f.write(f"{model['name']} validation pred top5 return avg: {model['validation_top5']['pred_top5_return_avg']:.8f}\n")
            f.write(f"{model['name']} validation universe return avg: {model['validation_top5']['benchmark_top5_return_avg']:.8f}\n")
            f.write(f"{model['name']} validation pred top5 excess return avg: {model['validation_top5']['pred_top5_excess_return_avg']:.8f}\n")
            f.write(f"{model['name']} validation pred top5 excess return std: {model['validation_top5']['pred_top5_excess_return_std']:.8f}\n")
            f.write(f"{model['name']} validation pred top5 excess return min: {model['validation_top5']['pred_top5_excess_return_min']:.8f}\n")
            f.write(f"{model['name']} validation pred top5 excess positive rate: {model['validation_top5']['pred_top5_excess_positive_rate']:.8f}\n")
            f.write(f"{model['name']} validation pred top5 precision avg: {model['validation_top5']['pred_top5_precision_avg']:.8f}\n")
            f.write(f"{model['name']} holdout test RankIC: {model['holdout_test_rank_ic']:.8f}\n")
            f.write(f"{model['name']} holdout test pred top10 return avg: {model['holdout_test_top10']['pred_top10_return_avg']:.8f}\n")
            f.write(f"{model['name']} holdout test pred top10 excess return avg: {model['holdout_test_top10']['pred_top10_excess_return_avg']:.8f}\n")
            f.write(f"{model['name']} holdout test pred top10 precision avg: {model['holdout_test_top10']['pred_top10_precision_avg']:.8f}\n")
            f.write(f"{model['name']} holdout test pred top5 return avg: {model['holdout_test_top5']['pred_top5_return_avg']:.8f}\n")
            f.write(f"{model['name']} holdout test universe return avg: {model['holdout_test_top5']['benchmark_top5_return_avg']:.8f}\n")
            f.write(f"{model['name']} holdout test pred top5 excess return avg: {model['holdout_test_top5']['pred_top5_excess_return_avg']:.8f}\n")
            f.write(f"{model['name']} holdout test pred top5 excess return std: {model['holdout_test_top5']['pred_top5_excess_return_std']:.8f}\n")
            f.write(f"{model['name']} holdout test pred top5 excess return min: {model['holdout_test_top5']['pred_top5_excess_return_min']:.8f}\n")
            f.write(f"{model['name']} holdout test pred top5 excess positive rate: {model['holdout_test_top5']['pred_top5_excess_positive_rate']:.8f}\n")
            f.write(f"{model['name']} holdout test pred top5 precision avg: {model['holdout_test_top5']['pred_top5_precision_avg']:.8f}\n\n")
        f.write(f"Best validation top5 return model: {best_model['name']}\n")
    return best_model


def configured_models(metadata: dict, family_config: dict) -> list[dict]:
    """从 metadata 中筛出当前模型族配置对应的 artifact。"""
    models = []
    for name in family_config['model_names']:
        model_config = family_config['model_params'][name]
        input_window = int(model_config['input_window'])
        feature_type = model_config['feature_type']
        feature_num = feature_num_for_window(input_window, feature_type)
        matches = [
            model
            for model in metadata['models']
            if model['name'] == name
            and model['input_window'] == input_window
            and model['feature_type'] == feature_type
            and model['feature_num'] == feature_num
        ]
        if not matches:
            raise ValueError(f'no model artifact for current config: {name} {input_window} {feature_type}')
        models.append(matches[-1])
    return models


def score_prediction_samples(models, validation_plan, stockid2idx, score_model: Callable[[dict, np.ndarray], np.ndarray]) -> pd.DataFrame:
    """按模型所需窗口生成预测特征并合并分数。"""
    latest = None
    latest_date = validation_plan.test_date
    for input_window, feature_num in sorted({(model['input_window'], model['feature_num']) for model in models}):
        prediction_samples = validation_plan.get_prediction_samples(input_window)
        feature_df, features = preprocess_stock_history_samples(prediction_samples, feature_num, stockid2idx)
        feature_df = feature_df[feature_df['股票代码'].isin(stockid2idx)].sort_values('股票代码').reset_index(drop=True)
        if latest is None:
            if len(feature_df) < 5:
                raise ValueError(f'not enough stocks on latest date {latest_date.date()}: {len(feature_df)}')
            latest = feature_df[['股票代码', '日期']].copy()
        else:
            latest = latest.merge(feature_df[['股票代码', '日期']], on=['股票代码', '日期'], how='inner')

        x = feature_df[features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
        for model in [item for item in models if item['input_window'] == input_window and item['feature_num'] == feature_num]:
            scored = feature_df[['股票代码', '日期']].copy()
            scored[f"{model['name']}_score"] = score_model(model, x)
            latest = latest.merge(scored, on=['股票代码', '日期'], how='inner')
    if latest is None:
        raise ValueError('no configured models found in metadata')
    if len(latest) < 5:
        raise ValueError(f'not enough common stocks on latest date {latest_date.date()}: {len(latest)}')
    return latest


def portfolio_config_for_models(model_names: list[str]) -> PortfolioConfig:
    """预测时使用当前模型族自己的 Top10 候选池。"""
    return PortfolioConfig(union_top_k_by_model=tuple((name, 10) for name in model_names))


def write_prediction_outputs(latest, models, family_config, validation_plan, raw_df, holdout_fold) -> None:
    """生成当前模型族的 result.csv 和诊断文件。"""
    output_path = Path('./output/result.csv')
    top10_path = Path('./output/top10_by_model.csv')
    candidates_path = Path('./output/candidates.csv')
    portfolio_path = Path('./output/portfolio_selection.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_names = list(family_config['model_names'])
    portfolio_config = portfolio_config_for_models(model_names)
    top10, candidates, selected = select_portfolio(
        latest,
        raw_df,
        validation_plan.test_date,
        validation_plan.get_train_returns(holdout_fold, max(model['input_window'] for model in models)),
        model_names,
        portfolio_config,
    )
    top10.to_csv(top10_path, index=False)
    candidates.to_csv(candidates_path, index=False)
    selected.to_csv(portfolio_path, index=False)
    pd.DataFrame({
        'stock_id': selected['stock_id'].tolist(),
        'weight': selected['final_weight'].round(10).tolist(),
    }).to_csv(output_path, index=False)
    validate_result_file(output_path)

    topk_description = ','.join(f'{name}:{top_k}' for name, top_k in portfolio_config.union_top_k_by_model)
    print(f"Prediction: target={validation_plan.test_date.date()} | universe={len(latest)}")
    print(f'Candidate pool: model_topk_rows={len(top10)} | unique={len(candidates)} | {topk_description}')
    print('Model TopK:')
    print(top10[['model', 'model_rank', 'stock_id', 'score', 'model_rank_norm']].to_string(index=False))
    print('Final Portfolio:')
    print(selected[['final_rank', 'stock_id', 'source_models', 'final_weight', 'consensus_score', 'mu']].to_string(index=False))
    print(f'result written to: {output_path}')
    print(f'top10 written to: {top10_path}')
    print(f'candidates written to: {candidates_path}')
    print(f'portfolio selection written to: {portfolio_path}')


def train_family(model_type: str, family_config: dict, train_one: Callable, output_dir: Path) -> float:
    """通用 holdout 训练入口。"""
    set_seed(int(config['seed']))
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_df = load_market_data(config)
    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
    validation_plan = build_validation_plan(raw_df, config)
    holdout_fold = validation_plan.holdout_split()
    bindings = model_bindings(family_config)
    datasets = build_train_datasets(validation_plan, holdout_fold, bindings, stockid2idx)

    _, first_input_window, _, first_feature_num = bindings[0]
    train_df, val_df, holdout_df, _ = datasets[(first_input_window, first_feature_num)]
    print_data_ranges(raw_df, validation_plan, first_input_window, train_df, val_df, holdout_df)

    models = []
    for name, input_window, feature_type, feature_num in bindings:
        train_df, val_df, holdout_df, features = datasets[(input_window, feature_num)]
        print(f'{name} input_window={input_window} feature_type={feature_type} feature_num={feature_num} | features={len(features)}')
        models.append(train_one(name, input_window, feature_type, feature_num, train_df, val_df, holdout_df, features))

    best_model = write_train_outputs(output_dir, model_type, family_config, models, stock_ids, validation_plan, holdout_fold, bindings)
    print(f"best validation top5 return model: {best_model['name']} ({best_model['validation_score']:.6f})")
    return best_model['validation_score']


def predict_family(family_config: dict, output_dir: Path, score_model: Callable[[dict, np.ndarray], np.ndarray]) -> None:
    """通用预测入口。"""
    metadata_path = output_dir / 'metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f'metadata file not found: {metadata_path}')

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    stockid2idx = {sid: idx for idx, sid in enumerate(metadata['stock_ids'])}
    models = configured_models(metadata, family_config)
    raw_df = load_market_data(config)
    validation_plan = build_validation_plan(raw_df, config)
    holdout_fold = validation_plan.holdout_split()
    latest = score_prediction_samples(models, validation_plan, stockid2idx, score_model)
    write_prediction_outputs(latest, models, family_config, validation_plan, raw_df, holdout_fold)
