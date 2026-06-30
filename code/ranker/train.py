import json
import multiprocessing as mp
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import config
from ..features.baseline import date_group_sizes, preprocess_stock_data_samples
from ..utils.runtime_split import load_market_data, split_runtime_data
from .lambdarankic import lambdarankic_objective
from .rankic import (
    topk_return_metrics,
    xgb_rank_ic_metric,
    xgb_rank_return_metrics,
)


def set_seed(seed: int) -> None:
    """固定随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def make_dmatrix(df: pd.DataFrame, features: list[str]) -> tuple[xgb.DMatrix, list[int]]:
    """按日期 group 构造 XGBoost ranking 数据。"""
    grouped_df, groups = date_group_sizes(df, int(config['min_group_size']))
    x = grouped_df[features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    y = grouped_df['label'].to_numpy(dtype=np.float32)
    dmatrix = xgb.DMatrix(x, label=y, missing=np.nan, group=np.asarray(groups, dtype=np.uint32))
    return dmatrix, groups


def choose_top5_iteration(validation_top5: list[float]) -> dict:
    """选择 validation top5 return 最高的迭代；完全相同则取更早轮次。"""
    if not validation_top5:
        raise ValueError('empty validation top5 metric history')

    top5 = np.asarray(validation_top5, dtype=np.float64)
    if not np.isfinite(top5).all():
        raise ValueError('validation top5 metric history contains non-finite values')

    best_iteration = int(np.argmax(top5))
    return {
        'best_iteration': best_iteration,
        'top5_return': float(top5[best_iteration]),
        'selection_rule': 'max_validation_top5_return_earliest_tie',
        'num_iterations': int(len(top5)),
    }


def target_range(df: pd.DataFrame) -> tuple[str, str]:
    """返回样本目标周日期范围，用于训练日志审计。"""
    dates = pd.to_datetime(df['日期']).dt.date
    return str(dates.min()), str(dates.max())


def train_one_model(
    name: str,
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    val_groups: list[int],
    dholdout: xgb.DMatrix,
    holdout_groups: list[int],
    output_dir: Path,
) -> dict:
    """训练一个排序模型并保存。"""
    params = dict(config['xgb_params'][name])
    params['disable_default_eval_metric'] = 1
    if name == 'lambdarankic':
        obj = lambdarankic_objective
    else:
        obj = None

    model_dir = output_dir / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    evals_result = {}
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(config['num_boost_round']),
        evals=[(dtrain, 'train'), (dval, 'validation')],
        obj=obj,
        custom_metric=xgb_rank_return_metrics,
        maximize=True,
        evals_result=evals_result,
        verbose_eval=1,
    )

    validation_metrics = evals_result.get('validation', {})
    validation_top5_values = [float(value) for value in validation_metrics.get('top5_return', [])]
    selection = choose_top5_iteration(validation_top5_values)
    best_iteration = selection['best_iteration']
    iteration_range = (0, best_iteration + 1)
    val_pred = booster.predict(dval, iteration_range=iteration_range)
    val_rank_ic = xgb_rank_ic_metric(val_pred, dval)[1]
    val_top5 = topk_return_metrics(val_pred, dval.get_label(), val_groups, top_k=5)
    val_top10 = topk_return_metrics(val_pred, dval.get_label(), val_groups, top_k=10)
    holdout_pred = booster.predict(dholdout, iteration_range=iteration_range)
    holdout_rank_ic = xgb_rank_ic_metric(holdout_pred, dholdout)[1]
    holdout_top5 = topk_return_metrics(holdout_pred, dholdout.get_label(), holdout_groups, top_k=5)
    holdout_top10 = topk_return_metrics(holdout_pred, dholdout.get_label(), holdout_groups, top_k=10)

    model_path = model_dir / f'{name}.json'
    booster.save_model(model_path)
    with open(model_dir / f'{name}_evals.json', 'w', encoding='utf-8') as f:
        json.dump(evals_result, f, ensure_ascii=False, indent=2)

    print(f"saved {name}: {model_path}")
    print(f"{name} best iteration by validation top5: {best_iteration}")
    print(f"{name} best validation top5 return: {selection['top5_return']:.6f}")
    print(f"{name} validation RankIC: {val_rank_ic:.6f}")
    print(f"{name} validation pred top5 return avg: {val_top5['pred_top5_return_avg']:.6f}")
    print(f"{name} validation pred top10 return avg: {val_top10['pred_top10_return_avg']:.6f}")
    print(f"{name} holdout test RankIC: {holdout_rank_ic:.6f}")
    print(f"{name} holdout test pred top5 return avg: {holdout_top5['pred_top5_return_avg']:.6f}")
    print(f"{name} holdout test pred top10 return avg: {holdout_top10['pred_top10_return_avg']:.6f}")

    return {
        'name': name,
        'path': str(model_path.relative_to(output_dir)),
        'xgb_params': params,
        'best_iteration': best_iteration,
        'best_selection_metric': 'validation_top5_return',
        'best_selection_score': selection['top5_return'],
        'best_selection': selection,
        'validation_rank_ic': val_rank_ic,
        'validation_score': selection['top5_return'],
        'validation_top10': val_top10,
        'validation_top5': val_top5,
        'holdout_test_rank_ic': holdout_rank_ic,
        'holdout_test_top10': holdout_top10,
        'holdout_test_top5': holdout_top5,
    }


def main() -> float:
    """按配置训练多个排序模型。"""
    set_seed(int(config['seed']))
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_market_data(config)
    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

    runtime = split_runtime_data(raw_df, config)
    train_samples = runtime.get_train_samples()
    validation_samples = runtime.get_validation_samples()
    holdout_samples = runtime.get_test_samples()
    prediction_samples = runtime.get_prediction_samples()
    train_df, features = preprocess_stock_data_samples(
        train_samples,
        config['feature_num'],
        stockid2idx,
    )
    val_df, _ = preprocess_stock_data_samples(
        validation_samples,
        config['feature_num'],
        stockid2idx,
    )
    holdout_df, _ = preprocess_stock_data_samples(
        holdout_samples,
        config['feature_num'],
        stockid2idx,
    )
    train_df = train_df.dropna(subset=['label']).sort_values(['日期', '股票代码']).reset_index(drop=True)
    val_df = val_df.dropna(subset=['label']).sort_values(['日期', '股票代码']).reset_index(drop=True)
    holdout_df = holdout_df.dropna(subset=['label']).sort_values(['日期', '股票代码']).reset_index(drop=True)
    train_start, train_end = target_range(train_df)
    validation_start, validation_end = target_range(val_df)
    holdout_start, holdout_end = target_range(holdout_df)

    print(f"Data: {raw_df['日期'].min().date()} to {raw_df['日期'].max().date()} | rows={len(raw_df)} | stocks={raw_df['股票代码'].nunique()}")
    print(f"Train: {train_start} to {train_end} ")
    print(f"Validation: {validation_start} to {validation_end}")
    print(f"Holdout Test: {holdout_start} to {holdout_end}")
    print(f"Prediction: target={runtime.test_date.date()} | input={runtime.test_input_start.date()} to {(runtime.test_date - pd.Timedelta(days=3)).date()} | stocks={len(prediction_samples)}")

    dtrain, train_groups = make_dmatrix(train_df, features)
    dval, val_groups = make_dmatrix(val_df, features)
    dholdout, holdout_groups = make_dmatrix(holdout_df, features)
    print(f"Train Samples: {len(train_groups)} | rows={dtrain.num_row()}")
    print(f"Validation Samples: {len(val_groups)} | rows={dval.num_row()}")
    print(f"Holdout Test Samples: {len(holdout_groups)} | rows={dholdout.num_row()}")

    model_names = list(config['model_names'])
    missing_params = [name for name in model_names if name not in config['xgb_params']]
    if missing_params:
        raise ValueError(f'missing xgb params: {missing_params}')
    models = [
        train_one_model(name, dtrain, dval, val_groups, dholdout, holdout_groups, output_dir)
        for name in model_names
    ]
    best_model = max(models, key=lambda item: item['validation_score'])

    metadata = {
        'model_type': 'xgboost_rankers',
        'model_names': model_names,
        'models': models,
        'features': features,
        'stock_ids': stock_ids,
        'config': config,
        'input_window': runtime.input_window,
        'num_validation_weeks': runtime.num_validation_weeks,
        'num_test_weeks': runtime.num_test_weeks,
        'validation_input_start': str(runtime.validation_input_start.date()),
        'validation_target_start': str(runtime.validation_target_start.date()),
        'holdout_test_input_start': str(runtime.holdout_test_input_start.date()),
        'holdout_test_target_start': str(runtime.holdout_test_target_start.date()),
        'test_input_start': str(runtime.test_input_start.date()),
        'test_date': str(runtime.test_date.date()),
        'best_model': best_model['name'],
        'validation_rank_ic': best_model['validation_rank_ic'],
        'validation_score': best_model['validation_score'],
        'holdout_test_rank_ic': best_model['holdout_test_rank_ic'],
        'holdout_test_top5': best_model['holdout_test_top5'],
    }
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with open(output_dir / 'final_score.txt', 'w', encoding='utf-8') as f:
        for model in models:
            f.write(f"{model['name']} best iteration: {model['best_iteration']}\n")
            f.write(f"{model['name']} best selection metric: {model['best_selection_metric']}\n")
            f.write(f"{model['name']} best selection score: {model['best_selection_score']:.8f}\n")
            f.write(f"{model['name']} best selection details: {model['best_selection']}\n")
            f.write(f"{model['name']} validation RankIC: {model['validation_rank_ic']:.8f}\n")
            f.write(f"{model['name']} validation pred top10 return avg: {model['validation_top10']['pred_top10_return_avg']:.8f}\n")
            f.write(f"{model['name']} validation pred top10 group returns: {model['validation_top10']['pred_top10_group_returns']}\n")
            f.write(f"{model['name']} validation pred top5 return avg: {model['validation_top5']['pred_top5_return_avg']:.8f}\n")
            f.write(f"{model['name']} validation pred top5 group returns: {model['validation_top5']['pred_top5_group_returns']}\n")
            f.write(f"{model['name']} holdout test RankIC: {model['holdout_test_rank_ic']:.8f}\n")
            f.write(f"{model['name']} holdout test pred top10 return avg: {model['holdout_test_top10']['pred_top10_return_avg']:.8f}\n")
            f.write(f"{model['name']} holdout test pred top10 group returns: {model['holdout_test_top10']['pred_top10_group_returns']}\n")
            f.write(f"{model['name']} holdout test pred top5 return avg: {model['holdout_test_top5']['pred_top5_return_avg']:.8f}\n")
            f.write(f"{model['name']} holdout test pred top5 group returns: {model['holdout_test_top5']['pred_top5_group_returns']}\n")
            f.write('\n')
        f.write(f"Best validation top5 model: {best_model['name']}\n")

    print(f"best validation top5 model: {best_model['name']} ({best_model['validation_score']:.6f})")
    return best_model['validation_score']


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    score = main()
    print(f"\n########## rankers done, best validation top5: {score:.6f} ##########")
