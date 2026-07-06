"""XGBoost 排序模型接口；使用 `XGBoostRankModel().train()` / `.predict()` 完成训练和预测。"""

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm.auto import tqdm

from ...config import config
from ...features.baseline import date_group_sizes, preprocess_stock_data_samples, preprocess_stock_history_samples
from ...features.windows import feature_num_for_window
from ...portfolio.postprocess import DEFAULT_PORTFOLIO_CONFIG, select_portfolio
from ...utils.runtime_split import load_market_data
from ...utils.validation import build_validation_plan
from .loss import (
    lambdarankic_objective,
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


def choose_top5_iteration(validation_top5: list[float]) -> dict:
    """兼容旧实验：选择 validation top5 return 最高的迭代。"""
    selection = choose_best_iteration(validation_top5, 'validation_top5_return')
    selection['top5_return'] = selection['score']
    return selection


def target_range(df: pd.DataFrame) -> tuple[str, str]:
    """返回样本目标周日期范围，用于训练日志审计。"""
    dates = pd.to_datetime(df['日期']).dt.date
    return str(dates.min()), str(dates.max())


class TqdmTrainingCallback(xgb.callback.TrainingCallback):
    """XGBoost 训练进度条，逐轮展示核心排序指标。"""

    def __init__(self, total_rounds: int, desc: str) -> None:
        self.total_rounds = total_rounds
        self.desc = desc
        self.bar = None
        self.metrics = (
            ('validation', 'top5_excess_return', 'val_excess'),
            ('validation', 'top5_return', 'val_top5'),
            ('validation', 'top5_precision', 'p@5'),
            ('validation', 'rank_ic', 'val_ic'),
            ('train', 'rank_ic', 'train_ic'),
        )

    def before_training(self, model):
        self.bar = tqdm(total=self.total_rounds, desc=self.desc, unit='round', dynamic_ncols=True)
        return model

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        if self.bar is None:
            return False
        postfix = {}
        for dataset, metric, name in self.metrics:
            if dataset in evals_log and metric in evals_log[dataset]:
                postfix[name] = f"{float(evals_log[dataset][metric][-1]):.6f}"
        self.bar.update(1)
        if postfix:
            self.bar.set_postfix(postfix)
        return False

    def after_training(self, model):
        if self.bar is not None:
            self.bar.close()
        return model


class XGBoostRankModel:
    """XGBoost 排序模型，统一提供 train/predict 标准接口。"""

    def __init__(self) -> None:
        self.output_dir = Path(config['output_dir'])

    def model_bindings(self) -> list[tuple[str, int, str, str]]:
        """把配置解析成模型名、输入窗口、特征类型、特征编号。"""
        bindings = []
        for name in config['model_names']:
            model_config = config['model_params'][name]
            input_window = int(model_config['input_window'])
            feature_type = model_config['feature_type']
            feature_num = feature_num_for_window(input_window, feature_type)
            bindings.append((name, input_window, feature_type, feature_num))
        return bindings

    def train_one(self, name, input_window, feature_type, feature_num, train_df, val_df, holdout_df, features) -> dict:
        """训练单个排序模型并保存模型文件。"""
        model_config = dict(config['model_params'][name])
        params = {key: value for key, value in model_config.items() if key not in {'input_window', 'feature_type'}}
        params['disable_default_eval_metric'] = 1
        obj = lambdarankic_objective if name == 'lambdarankic' else None

        dtrain, train_groups = make_dmatrix(train_df, features)
        dval, val_groups = make_dmatrix(val_df, features)
        dholdout, holdout_groups = make_dmatrix(holdout_df, features)
        print(f"{name} Train Samples: {len(train_groups)} | rows={dtrain.num_row()}")
        print(f"{name} Validation Samples: {len(val_groups)} | rows={dval.num_row()}")
        print(f"{name} Holdout Test Samples: {len(holdout_groups)} | rows={dholdout.num_row()}")

        model_dir = self.output_dir / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        evals_result = {}
        num_boost_round = int(config['num_boost_round'])
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, 'train'), (dval, 'validation')],
            obj=obj,
            custom_metric=xgb_rank_return_metrics,
            maximize=True,
            evals_result=evals_result,
            verbose_eval=False,
            callbacks=[TqdmTrainingCallback(num_boost_round, f'train {name}')],
        )

        validation_top5_excess_values = [float(value) for value in evals_result['validation']['top5_excess_return']]
        selection = choose_best_iteration(validation_top5_excess_values, 'validation_top5_excess_return')
        best_iteration = selection['best_iteration']
        iteration_range = (0, best_iteration + 1)
        val_pred = booster.predict(dval, iteration_range=iteration_range)
        holdout_pred = booster.predict(dholdout, iteration_range=iteration_range)
        val_rank_ic = xgb_rank_ic_metric(val_pred, dval)[1]
        holdout_rank_ic = xgb_rank_ic_metric(holdout_pred, dholdout)[1]
        val_top5 = topk_return_metrics(val_pred, dval.get_label(), val_groups, top_k=5)
        val_top10 = topk_return_metrics(val_pred, dval.get_label(), val_groups, top_k=10)
        holdout_top5 = topk_return_metrics(holdout_pred, dholdout.get_label(), holdout_groups, top_k=5)
        holdout_top10 = topk_return_metrics(holdout_pred, dholdout.get_label(), holdout_groups, top_k=10)

        artifact = f'{name}_w{input_window}_{feature_num}_{best_iteration}'
        model_path = model_dir / f'{artifact}.json'
        booster.save_model(model_path)
        with open(model_dir / f'{artifact}_evals.json', 'w', encoding='utf-8') as f:
            json.dump(evals_result, f, ensure_ascii=False, indent=2)

        print(f'saved {name}: {model_path}')
        print(f'{name} best iteration by validation top5 excess: {best_iteration}')
        print(f"{name} best validation top5 excess return: {selection['score']:.6f}")
        print(f'{name} validation RankIC: {val_rank_ic:.6f}')
        print(f"{name} validation pred top5 return avg: {val_top5['pred_top5_return_avg']:.6f}")
        print(f"{name} validation pred top5 excess return avg: {val_top5['pred_top5_excess_return_avg']:.6f}")
        print(f"{name} validation pred top5 precision avg: {val_top5['pred_top5_precision_avg']:.6f}")
        print(f"{name} validation pred top10 return avg: {val_top10['pred_top10_return_avg']:.6f}")
        print(f"{name} validation pred top10 excess return avg: {val_top10['pred_top10_excess_return_avg']:.6f}")
        print(f"{name} validation pred top10 precision avg: {val_top10['pred_top10_precision_avg']:.6f}")
        print(f'{name} holdout test RankIC: {holdout_rank_ic:.6f}')
        print(f"{name} holdout test pred top5 return avg: {holdout_top5['pred_top5_return_avg']:.6f}")
        print(f"{name} holdout test pred top5 excess return avg: {holdout_top5['pred_top5_excess_return_avg']:.6f}")
        print(f"{name} holdout test pred top5 precision avg: {holdout_top5['pred_top5_precision_avg']:.6f}")
        print(f"{name} holdout test pred top10 return avg: {holdout_top10['pred_top10_return_avg']:.6f}")
        print(f"{name} holdout test pred top10 excess return avg: {holdout_top10['pred_top10_excess_return_avg']:.6f}")
        print(f"{name} holdout test pred top10 precision avg: {holdout_top10['pred_top10_precision_avg']:.6f}")

        return {
            'name': name,
            'artifact_name': artifact,
            'input_window': input_window,
            'feature_type': feature_type,
            'feature_num': feature_num,
            'features': features,
            'path': str(model_path.relative_to(self.output_dir)),
            'model_params': model_config,
            'best_iteration': best_iteration,
            'best_selection_metric': selection['metric'],
            'best_selection_score': selection['score'],
            'best_selection': selection,
            'validation_rank_ic': val_rank_ic,
            'validation_score': val_top5['pred_top5_excess_return_avg'],
            'validation_top10': val_top10,
            'validation_top5': val_top5,
            'holdout_test_rank_ic': holdout_rank_ic,
            'holdout_test_top10': holdout_top10,
            'holdout_test_top5': holdout_top5,
        }

    def build_train_datasets(self, validation_plan, holdout_fold, bindings, stockid2idx):
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

    def write_train_outputs(self, models, stock_ids, validation_plan, holdout_fold, bindings) -> dict:
        """保存训练元数据和分数文本。"""
        best_model = max(models, key=lambda item: item['validation_score'])
        metadata = {
            'model_type': 'xgboost_models',
            'model_names': list(config['model_names']),
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
        }
        with open(self.output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        with open(self.output_dir / 'final_score.txt', 'w', encoding='utf-8') as f:
            for model in models:
                f.write(f"{model['name']} artifact: {model['artifact_name']}.json\n")
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
                f.write(f"{model['name']} validation pred top10 excess return std: {model['validation_top10']['pred_top10_excess_return_std']:.8f}\n")
                f.write(f"{model['name']} validation pred top10 excess return min: {model['validation_top10']['pred_top10_excess_return_min']:.8f}\n")
                f.write(f"{model['name']} validation pred top10 excess positive rate: {model['validation_top10']['pred_top10_excess_positive_rate']:.8f}\n")
                f.write(f"{model['name']} validation pred top10 precision avg: {model['validation_top10']['pred_top10_precision_avg']:.8f}\n")
                f.write(f"{model['name']} validation pred top10 group returns: {model['validation_top10']['pred_top10_group_returns']}\n")
                f.write(f"{model['name']} validation pred top5 return avg: {model['validation_top5']['pred_top5_return_avg']:.8f}\n")
                f.write(f"{model['name']} validation universe return avg: {model['validation_top5']['benchmark_top5_return_avg']:.8f}\n")
                f.write(f"{model['name']} validation pred top5 excess return avg: {model['validation_top5']['pred_top5_excess_return_avg']:.8f}\n")
                f.write(f"{model['name']} validation pred top5 excess return std: {model['validation_top5']['pred_top5_excess_return_std']:.8f}\n")
                f.write(f"{model['name']} validation pred top5 excess return min: {model['validation_top5']['pred_top5_excess_return_min']:.8f}\n")
                f.write(f"{model['name']} validation pred top5 excess positive rate: {model['validation_top5']['pred_top5_excess_positive_rate']:.8f}\n")
                f.write(f"{model['name']} validation pred top5 precision avg: {model['validation_top5']['pred_top5_precision_avg']:.8f}\n")
                f.write(f"{model['name']} validation pred top5 group returns: {model['validation_top5']['pred_top5_group_returns']}\n")
                f.write(f"{model['name']} validation pred top5 excess group returns: {model['validation_top5']['pred_top5_excess_group_returns']}\n")
                f.write(f"{model['name']} holdout test RankIC: {model['holdout_test_rank_ic']:.8f}\n")
                f.write(f"{model['name']} holdout test pred top10 return avg: {model['holdout_test_top10']['pred_top10_return_avg']:.8f}\n")
                f.write(f"{model['name']} holdout test pred top10 excess return avg: {model['holdout_test_top10']['pred_top10_excess_return_avg']:.8f}\n")
                f.write(f"{model['name']} holdout test pred top10 excess return std: {model['holdout_test_top10']['pred_top10_excess_return_std']:.8f}\n")
                f.write(f"{model['name']} holdout test pred top10 excess return min: {model['holdout_test_top10']['pred_top10_excess_return_min']:.8f}\n")
                f.write(f"{model['name']} holdout test pred top10 excess positive rate: {model['holdout_test_top10']['pred_top10_excess_positive_rate']:.8f}\n")
                f.write(f"{model['name']} holdout test pred top10 precision avg: {model['holdout_test_top10']['pred_top10_precision_avg']:.8f}\n")
                f.write(f"{model['name']} holdout test pred top10 group returns: {model['holdout_test_top10']['pred_top10_group_returns']}\n")
                f.write(f"{model['name']} holdout test pred top5 return avg: {model['holdout_test_top5']['pred_top5_return_avg']:.8f}\n")
                f.write(f"{model['name']} holdout test universe return avg: {model['holdout_test_top5']['benchmark_top5_return_avg']:.8f}\n")
                f.write(f"{model['name']} holdout test pred top5 excess return avg: {model['holdout_test_top5']['pred_top5_excess_return_avg']:.8f}\n")
                f.write(f"{model['name']} holdout test pred top5 excess return std: {model['holdout_test_top5']['pred_top5_excess_return_std']:.8f}\n")
                f.write(f"{model['name']} holdout test pred top5 excess return min: {model['holdout_test_top5']['pred_top5_excess_return_min']:.8f}\n")
                f.write(f"{model['name']} holdout test pred top5 excess positive rate: {model['holdout_test_top5']['pred_top5_excess_positive_rate']:.8f}\n")
                f.write(f"{model['name']} holdout test pred top5 precision avg: {model['holdout_test_top5']['pred_top5_precision_avg']:.8f}\n")
                f.write(f"{model['name']} holdout test pred top5 group returns: {model['holdout_test_top5']['pred_top5_group_returns']}\n")
                f.write(f"{model['name']} holdout test pred top5 excess group returns: {model['holdout_test_top5']['pred_top5_excess_group_returns']}\n")
                f.write('\n')
            f.write(f"Best validation top5 excess model: {best_model['name']}\n")
        return best_model

    def train(self) -> float:
        """正式 holdout 训练入口。"""
        set_seed(int(config['seed']))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        raw_df = load_market_data(config)
        stock_ids = sorted(raw_df['股票代码'].unique())
        stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
        validation_plan = build_validation_plan(raw_df, config)
        holdout_fold = validation_plan.holdout_split()
        bindings = self.model_bindings()
        datasets = self.build_train_datasets(validation_plan, holdout_fold, bindings, stockid2idx)

        _, first_input_window, _, first_feature_num = bindings[0]
        train_df, val_df, holdout_df, _ = datasets[(first_input_window, first_feature_num)]
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

        models = []
        for name, input_window, feature_type, feature_num in bindings:
            train_df, val_df, holdout_df, features = datasets[(input_window, feature_num)]
            print(f'{name} input_window={input_window} feature_type={feature_type} feature_num={feature_num} | features={len(features)}')
            models.append(self.train_one(name, input_window, feature_type, feature_num, train_df, val_df, holdout_df, features))

        best_model = self.write_train_outputs(models, stock_ids, validation_plan, holdout_fold, bindings)
        print(f"best validation top5 excess model: {best_model['name']} ({best_model['validation_score']:.6f})")
        return best_model['validation_score']

    def configured_models(self, metadata: dict) -> list[dict]:
        """从 metadata 中筛出当前全局配置对应的模型 artifact。"""
        models = []
        for name in config['model_names']:
            model_config = config['model_params'][name]
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

    def score_prediction_samples(self, models, metadata, validation_plan, stockid2idx) -> pd.DataFrame:
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
            dtest = xgb.DMatrix(x, missing=np.nan)
            for model in [item for item in models if item['input_window'] == input_window and item['feature_num'] == feature_num]:
                name = model['name']
                model_path = self.output_dir / model['path']
                if not model_path.exists():
                    raise FileNotFoundError(f'model file not found: {model_path}')
                booster = xgb.Booster()
                booster.load_model(model_path)
                # 预测阶段按当前模型参数恢复设备。
                booster.set_param({'device': config['model_params'][name]['device']})
                iteration_range = (0, int(model['best_iteration']) + 1)
                scored = feature_df[['股票代码', '日期']].copy()
                scored[f'{name}_score'] = booster.predict(dtest, iteration_range=iteration_range)
                latest = latest.merge(scored, on=['股票代码', '日期'], how='inner')
        if latest is None:
            raise ValueError('no configured models found in metadata')
        if len(latest) < 5:
            raise ValueError(f'not enough common stocks on latest date {latest_date.date()}: {len(latest)}')
        return latest

    def predict(self) -> None:
        """加载训练 artifact，生成 output/result.csv。"""
        metadata_path = self.output_dir / 'metadata.json'
        output_path = Path('./output/result.csv')
        top10_path = Path('./output/top10_by_model.csv')
        candidates_path = Path('./output/candidates.csv')
        portfolio_path = Path('./output/portfolio_selection.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not metadata_path.exists():
            raise FileNotFoundError(f'metadata file not found: {metadata_path}')

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        stockid2idx = {sid: idx for idx, sid in enumerate(metadata['stock_ids'])}
        models = self.configured_models(metadata)
        raw_df = load_market_data(config)
        validation_plan = build_validation_plan(raw_df, config)
        holdout_fold = validation_plan.holdout_split()
        latest = self.score_prediction_samples(models, metadata, validation_plan, stockid2idx)
        model_names = list(config['model_names'])

        # 各模型取验证集选择的 topK，合并候选池后由组合优化器选出最终 top5。
        top10, candidates, selected = select_portfolio(
            latest,
            raw_df,
            validation_plan.test_date,
            validation_plan.get_train_returns(holdout_fold, max(model['input_window'] for model in models)),
            model_names,
            DEFAULT_PORTFOLIO_CONFIG,
        )
        top10.to_csv(top10_path, index=False)
        candidates.to_csv(candidates_path, index=False)
        selected.to_csv(portfolio_path, index=False)
        pd.DataFrame({
            'stock_id': selected['stock_id'].tolist(),
            'weight': selected['final_weight'].round(10).tolist(),
        }).to_csv(output_path, index=False)

        overlap_count = int(top10['stock_id'].duplicated().sum())
        if DEFAULT_PORTFOLIO_CONFIG.union_top_k_by_model:
            topk_description = ','.join(f'{name}:{top_k}' for name, top_k in DEFAULT_PORTFOLIO_CONFIG.union_top_k_by_model)
        else:
            topk_description = f'top_k_per_model={DEFAULT_PORTFOLIO_CONFIG.top_k_per_model}'
        print(f"Prediction: target={validation_plan.test_date.date()} | universe={len(latest)}")
        print(f'Candidate pool: model_topk_rows={len(top10)} | unique={len(candidates)} | overlap={overlap_count} | {topk_description}')
        print('Model TopK:')
        print(top10[['model', 'model_rank', 'stock_id', 'score', 'model_rank_norm']].to_string(index=False))
        print('Final Portfolio:')
        print(selected[['final_rank', 'stock_id', 'source_models', 'final_weight', 'consensus_score', 'mu']].to_string(index=False))
        print(f'result written to: {output_path}')
        print(f'top10 written to: {top10_path}')
        print(f'candidates written to: {candidates_path}')
        print(f'portfolio selection written to: {portfolio_path}')
