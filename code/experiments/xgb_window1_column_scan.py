"""逐列寻找 window_1 最有效因子。"""

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import pandas as pd
import xgboost as xgb

from code.config import config as base_config
from code.features.baseline import FEATURE_COLUMNS, preprocess_stock_data_samples
from code.features.window_1 import BASE_FEATURE_COLUMNS
from code.models.xgboost.loss import topk_return_metrics, xgb_rank_ic_metric
from code.models.xgboost.model import choose_best_iteration
from code.features.windows import feature_num_for_window
from code.utils.runtime_split import load_market_data
from code.utils.validation import build_validation_plan

from .xgb_window_factor_scan import make_dmatrix, write_outputs


def set_seed(seed: int) -> None:
    """固定随机种子。"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_existing(path: Path) -> dict:
    """读取已有结果，用于断点续跑。"""
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return {'rows': [], 'summary': []}


def candidate_columns(feature_num: str, include_base: bool, limit_to: set[str] | None) -> list[str]:
    """生成待测试因子列。"""
    columns = FEATURE_COLUMNS[feature_num]
    result = []
    for column in columns:
        if not include_base and column in BASE_FEATURE_COLUMNS:
            continue
        if limit_to is not None and column not in limit_to:
            continue
        result.append(column)
    return result


def summarize_columns(rows: list[dict]) -> list[dict]:
    """按单列因子聚合 fold 指标。"""
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    summary = frame.groupby(['feature_variant', 'factor_column'], sort=False).agg(
        fold_count=('fold', 'nunique'),
        validation_top5_excess_return_mean=('validation_top5_excess_return', 'mean'),
        validation_top5_excess_return_std=('validation_top5_excess_return', 'std'),
        worst_fold=('validation_top5_excess_return', 'min'),
        validation_top10_excess_return_mean=('validation_top10_excess_return', 'mean'),
        validation_rank_ic_mean=('validation_rank_ic', 'mean'),
        validation_rank_ic_std=('validation_rank_ic', 'std'),
        validation_top5_precision_mean=('validation_top5_precision', 'mean'),
        validation_top5_excess_positive_rate_mean=('validation_top5_excess_positive_rate', 'mean'),
        best_iteration_mean=('best_iteration', 'mean'),
    ).reset_index()
    summary = summary.sort_values(
        ['validation_top5_excess_return_mean', 'worst_fold', 'validation_top10_excess_return_mean'],
        ascending=[False, False, False],
    )
    return summary.to_dict('records')


def safe_rank_ic(pred, dmatrix) -> float:
    """单因子模型可能给出常数预测，此时 RankIC 记为 0。"""
    try:
        return float(xgb_rank_ic_metric(pred, dmatrix)[1])
    except ValueError:
        return 0.0


def train_eval_column(params: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, column: str, rounds: int, min_group_size: int) -> dict:
    """训练单列模型，训练后逐轮选择验证 Top5 excess。"""
    dtrain, train_groups = make_dmatrix(train_df, [column], min_group_size)
    dval, val_groups = make_dmatrix(val_df, [column], min_group_size)
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=rounds,
        evals=[(dval, 'validation')],
        verbose_eval=False,
    )

    validation_values = []
    for iteration in range(rounds):
        pred = booster.predict(dval, iteration_range=(0, iteration + 1))
        top5 = topk_return_metrics(pred, dval.get_label(), val_groups, top_k=5)
        validation_values.append(float(top5['pred_top5_excess_return_avg']))

    selection = choose_best_iteration(validation_values, 'validation_top5_excess_return')
    pred = booster.predict(dval, iteration_range=(0, selection['best_iteration'] + 1))
    top5 = topk_return_metrics(pred, dval.get_label(), val_groups, top_k=5)
    top10 = topk_return_metrics(pred, dval.get_label(), val_groups, top_k=10)
    return {
        'features': 1,
        'rounds': rounds,
        'best_iteration': selection['best_iteration'],
        'best_selection_score': selection['score'],
        'validation_rank_ic': safe_rank_ic(pred, dval),
        'validation_universe_return': top5['benchmark_top5_return_avg'],
        'validation_top5_return': top5['pred_top5_return_avg'],
        'validation_top5_excess_return': top5['pred_top5_excess_return_avg'],
        'validation_top5_excess_return_std': top5['pred_top5_excess_return_std'],
        'validation_top5_excess_return_min': top5['pred_top5_excess_return_min'],
        'validation_top5_excess_positive_rate': top5['pred_top5_excess_positive_rate'],
        'validation_top5_precision': top5['pred_top5_precision_avg'],
        'validation_top10_return': top10['pred_top10_return_avg'],
        'validation_top10_excess_return': top10['pred_top10_excess_return_avg'],
        'validation_top10_excess_return_min': top10['pred_top10_excess_return_min'],
        'validation_top10_excess_positive_rate': top10['pred_top10_excess_positive_rate'],
        'validation_top10_precision': top10['pred_top10_precision_avg'],
        'train_groups': len(train_groups),
        'validation_groups': len(val_groups),
        'train_rows': dtrain.num_row(),
        'validation_rows': dval.num_row(),
    }


def parse_columns(values: list[str] | None) -> set[str] | None:
    """解析命令行列名或列名文件。"""
    if not values:
        return None
    columns = set()
    for value in values:
        path = Path(value)
        if path.exists():
            columns.update(line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip())
        else:
            columns.add(value)
    return columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=120)
    parser.add_argument('--variants', nargs='+', default=['raw', 'xsec'])
    parser.add_argument('--columns', nargs='*')
    parser.add_argument('--include-base', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('output/xgb_pairwise_window1_column_scan.json'))
    args = parser.parse_args()

    config = copy.deepcopy(base_config)
    config['validation'] = {**config['validation'], 'mode': 'rolling_kfold'}
    params = {
        key: value
        for key, value in config['model_params']['xgb_rank_pairwise'].items()
        if key not in {'input_window', 'feature_type', 'num_boost_round'}
    }
    params['disable_default_eval_metric'] = 1
    set_seed(int(config['seed']))

    started = time.time()
    raw_df = load_market_data(config)
    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
    plan = build_validation_plan(raw_df, config)
    folds = plan.validation_splits()
    payload = load_existing(args.output)
    rows = list(payload.get('rows', []))
    completed = {
        (row['feature_variant'], row['factor_column'], row['fold'])
        for row in rows
    }
    limit_to = parse_columns(args.columns)
    dataset_cache = {}

    for variant in args.variants:
        feature_type = 'window_1+xsec' if variant == 'xsec' else 'window_1'
        feature_num = feature_num_for_window(1, feature_type)
        columns = candidate_columns(feature_num, args.include_base, limit_to)
        for fold in folds:
            pending_columns = [
                column
                for column in columns
                if (variant, column, fold.name) not in completed
            ]
            if not pending_columns:
                continue
            dataset_key = (variant, fold.name)
            if dataset_key not in dataset_cache:
                print(f'prepare variant={variant} fold={fold.name} feature_num={feature_num}', flush=True)
                train_samples = plan.get_train_samples(fold, 1)
                val_samples = plan.get_validation_samples(fold, 1)
                train_df, all_features = preprocess_stock_data_samples(train_samples, feature_num, stockid2idx)
                val_df, _ = preprocess_stock_data_samples(val_samples, feature_num, stockid2idx)
                train_df = train_df.dropna(subset=['label']).sort_values(['日期', '股票代码']).reset_index(drop=True)
                val_df = val_df.dropna(subset=['label']).sort_values(['日期', '股票代码']).reset_index(drop=True)
                dataset_cache[dataset_key] = (train_df, val_df, set(all_features))

            train_df, val_df, all_features = dataset_cache[dataset_key]
            for column in pending_columns:
                if column not in all_features:
                    continue
                row_key = (variant, column, fold.name)
                print(f'train variant={variant} column={column} fold={fold.name}', flush=True)
                tic = time.time()
                metrics = train_eval_column(params, train_df, val_df, column, args.rounds, int(config['min_group_size']))
                row = {
                    'fold': fold.name,
                    'train_start': str(fold.train_start.date()),
                    'train_end': str(fold.train_end.date()),
                    'validation_start': str(fold.validation_start.date()),
                    'validation_end': str(fold.validation_end.date()),
                    'model': 'xgb_rank_pairwise',
                    'input_window': 1,
                    'feature_variant': variant,
                    'feature_type': feature_type,
                    'feature_num': feature_num,
                    'factor_column': column,
                    'factor_group': column,
                    **metrics,
                    'seconds': time.time() - tic,
                }
                rows.append(row)
                completed.add(row_key)
                payload = {
                    'experiment': 'xgb_rank_pairwise_window1_column_scan',
                    'seed': config['seed'],
                    'rounds': args.rounds,
                    'variants': args.variants,
                    'include_base': args.include_base,
                    'rows': rows,
                    'summary': summarize_columns(rows),
                    'elapsed_seconds': time.time() - started,
                }
                write_outputs(args.output, payload)
                print(
                    f"done top5_excess={row['validation_top5_excess_return']:.6f} "
                    f"rank_ic={row['validation_rank_ic']:.6f} seconds={row['seconds']:.1f}",
                    flush=True,
                )

    payload = {
        'experiment': 'xgb_rank_pairwise_window1_column_scan',
        'seed': config['seed'],
        'rounds': args.rounds,
        'variants': args.variants,
        'include_base': args.include_base,
        'rows': rows,
        'summary': summarize_columns(rows),
        'elapsed_seconds': time.time() - started,
    }
    write_outputs(args.output, payload)
    print(pd.DataFrame(payload['summary']).head(50).to_string(index=False))


if __name__ == '__main__':
    main()
