"""XGBoost pairwise 短窗口因子族滚动验证实验。"""

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from code.config import config as base_config
from code.features.baseline import (
    CROSS_SECTIONAL_SPECS,
    MARKET_STATE_SPECS,
    FEATURE_COLUMNS,
    date_group_sizes,
    preprocess_stock_data_samples,
)
from code.features.windows import BASE_FEATURE_COLUMNS, DAILY_FEATURE_COLUMNS, WINDOW_SPECS, bound_window_feature, feature_num_for_window
from code.models.xgboost.loss import topk_return_metrics, xgb_rank_ic_metric, xgb_rank_return_metrics
from code.models.xgboost.model import choose_best_iteration
from code.utils.runtime_split import load_market_data
from code.utils.validation import build_validation_plan


FACTOR_GROUPS = {
    'base': (),
    'daily_state': tuple(DAILY_FEATURE_COLUMNS),
    'momentum': ('WF_RET_1', 'WF_RET_{w}', 'WF_RET_MEAN_{w}', 'WF_RET_IR_{w}', 'WF_UP_RATE_{w}'),
    'reversal': ('WF_RET_1', 'WF_REV_{w}'),
    'volatility': ('WF_RANGE', 'WF_VOL_{w}', 'WF_DOWNSIDE_VOL_{w}', 'WF_RANGE_MEAN_{w}'),
    'liquidity': ('WF_LOG_AMOUNT', 'WF_TURN_MEAN_{w}', 'WF_TURN_Z_{w}', 'WF_AMOUNT_MEAN_{w}', 'WF_VOLUME_RATIO_{w}'),
    'position': ('WF_CLV', 'WF_VWAP_DIST', 'WF_CLV_MEAN_{w}', 'WF_HIGH_DIST_{w}', 'WF_LOW_DIST_{w}', 'WF_RSV_{w}'),
    'gap_pressure': ('WF_GAP', 'WF_BODY', 'WF_GAP_MEAN_{w}', 'WF_CLV_MEAN_{w}'),
    'all_window': ('*',),
}


def set_seed(seed: int) -> None:
    """固定随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def expand_factor_columns(input_window: int, group_name: str) -> list[str]:
    """把因子族模板展开成当前输入窗口实际列名。"""
    feature_num = bound_window_feature(input_window)
    if group_name == 'all_window':
        return list(FEATURE_COLUMNS[feature_num])

    columns = list(BASE_FEATURE_COLUMNS)
    spec_windows = WINDOW_SPECS[feature_num]['windows']
    for template in FACTOR_GROUPS[group_name]:
        if '{w}' in template:
            columns.extend(template.format(w=w) for w in spec_windows)
        else:
            columns.append(template)
    return list(dict.fromkeys(columns))


def xsec_aliases_for_columns(columns: list[str]) -> set[str]:
    """根据原始因子列找出对应横截面派生列别名。"""
    source_to_aliases = {}
    for alias, source_col in (*CROSS_SECTIONAL_SPECS, *MARKET_STATE_SPECS):
        source_to_aliases.setdefault(source_col, set()).add(alias)
    aliases = set()
    for column in columns:
        aliases.update(source_to_aliases.get(column, set()))
    return aliases


def select_features(all_features: list[str], input_window: int, group_name: str, use_xsec: bool) -> list[str]:
    """选择某个因子族的原始列和对应横截面派生列。"""
    if group_name == 'all_window':
        return list(all_features)

    raw_columns = expand_factor_columns(input_window, group_name)
    selected = [column for column in raw_columns if column in all_features]
    if use_xsec:
        aliases = xsec_aliases_for_columns(selected)
        for column in all_features:
            if any(
                column.endswith(f'_{alias}')
                and (
                    column.startswith('XS_RANK_')
                    or column.startswith('XS_REL_')
                    or column.startswith('XS_Z_')
                    or column.startswith('MKT_MEAN_')
                    or column.startswith('MKT_STD_')
                    or column.startswith('MKT_POS_RATIO_')
                    or column.startswith('MKT_TOP20_')
                    or column.startswith('MKT_BOTTOM20_')
                )
                for alias in aliases
            ):
                selected.append(column)
    return list(dict.fromkeys(selected))


def make_dmatrix(df: pd.DataFrame, features: list[str], min_group_size: int) -> tuple[xgb.DMatrix, list[int]]:
    """按日期 group 构造 XGBoost ranking 数据。"""
    grouped_df, groups = date_group_sizes(df, min_group_size)
    x = grouped_df[features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    y = grouped_df['label'].to_numpy(dtype=np.float32)
    dmatrix = xgb.DMatrix(x, label=y, missing=np.nan, group=np.asarray(groups, dtype=np.uint32), feature_names=features)
    return dmatrix, groups


def train_eval_fold(params: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, features: list[str], rounds: int, min_group_size: int) -> dict:
    """训练单个 fold，并按验证 Top5 excess 选择最佳迭代。"""
    dtrain, train_groups = make_dmatrix(train_df, features, min_group_size)
    dval, val_groups = make_dmatrix(val_df, features, min_group_size)
    evals_result = {}
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=rounds,
        evals=[(dval, 'validation')],
        custom_metric=xgb_rank_return_metrics,
        maximize=True,
        evals_result=evals_result,
        verbose_eval=False,
    )
    selection = choose_best_iteration(
        [float(value) for value in evals_result['validation']['top5_excess_return']],
        'validation_top5_excess_return',
    )
    pred = booster.predict(dval, iteration_range=(0, selection['best_iteration'] + 1))
    rank_ic = xgb_rank_ic_metric(pred, dval)[1]
    top5 = topk_return_metrics(pred, dval.get_label(), val_groups, top_k=5)
    top10 = topk_return_metrics(pred, dval.get_label(), val_groups, top_k=10)
    return {
        'features': len(features),
        'rounds': rounds,
        'best_iteration': selection['best_iteration'],
        'best_selection_score': selection['score'],
        'validation_rank_ic': rank_ic,
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


def summarize(rows: list[dict]) -> list[dict]:
    """按输入窗口和因子族聚合 fold 指标。"""
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    group_cols = ['input_window', 'feature_variant', 'factor_group']
    metrics = [
        'validation_top5_excess_return',
        'validation_top5_excess_return_min',
        'validation_top5_excess_positive_rate',
        'validation_top10_excess_return',
        'validation_rank_ic',
        'validation_top5_precision',
        'features',
        'best_iteration',
    ]
    summary = frame.groupby(group_cols, sort=False).agg(
        fold_count=('fold', 'nunique'),
        **{f'{metric}_mean': (metric, 'mean') for metric in metrics},
        **{f'{metric}_std': (metric, 'std') for metric in metrics[:6]},
        worst_fold=('validation_top5_excess_return', 'min'),
    ).reset_index()
    summary = summary.sort_values(
        ['input_window', 'validation_top5_excess_return_mean', 'worst_fold', 'validation_top10_excess_return_mean'],
        ascending=[True, False, False, False],
    )
    return summary.to_dict('records')


def write_outputs(path: Path, payload: dict) -> None:
    """保存 JSON 和扁平 CSV，方便 notebook 继续分析。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    pd.DataFrame(payload['rows']).to_csv(path.with_suffix('.rows.csv'), index=False)
    pd.DataFrame(payload['summary']).to_csv(path.with_suffix('.summary.csv'), index=False)


def load_existing(path: Path) -> dict:
    """读取已有结果，用于断点续跑。"""
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return {'rows': [], 'summary': []}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--windows', nargs='+', type=int, default=[1, 2, 4])
    parser.add_argument('--rounds', type=int, default=120)
    parser.add_argument('--groups', nargs='+', default=list(FACTOR_GROUPS))
    parser.add_argument('--variants', nargs='+', default=['raw', 'xsec'])
    parser.add_argument('--output', type=Path, default=Path('output/xgb_pairwise_window_factor_family_scan.json'))
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
        (row['input_window'], row['feature_variant'], row['factor_group'], row['fold'])
        for row in rows
    }
    dataset_cache = {}

    for input_window in args.windows:
        for variant in args.variants:
            feature_type = 'window+xsec' if variant == 'xsec' else 'window'
            feature_num = feature_num_for_window(input_window, feature_type)
            for fold in folds:
                dataset_key = (input_window, feature_num, fold.name)
                if dataset_key not in dataset_cache:
                    print(f'prepare window={input_window} variant={variant} fold={fold.name} feature_num={feature_num}', flush=True)
                    train_samples = plan.get_train_samples(fold, input_window)
                    val_samples = plan.get_validation_samples(fold, input_window)
                    train_df, all_features = preprocess_stock_data_samples(train_samples, feature_num, stockid2idx)
                    val_df, _ = preprocess_stock_data_samples(val_samples, feature_num, stockid2idx)
                    train_df = train_df.dropna(subset=['label']).sort_values(['日期', '股票代码']).reset_index(drop=True)
                    val_df = val_df.dropna(subset=['label']).sort_values(['日期', '股票代码']).reset_index(drop=True)
                    dataset_cache[dataset_key] = (train_df, val_df, all_features)

                train_df, val_df, all_features = dataset_cache[dataset_key]
                for group_name in args.groups:
                    row_key = (input_window, variant, group_name, fold.name)
                    if row_key in completed:
                        continue
                    use_xsec = variant == 'xsec'
                    features = select_features(all_features, input_window, group_name, use_xsec)
                    if len(features) < 2:
                        raise ValueError(f'not enough features for {row_key}: {features}')
                    print(f'train window={input_window} variant={variant} factor={group_name} fold={fold.name} features={len(features)}', flush=True)
                    tic = time.time()
                    metrics = train_eval_fold(params, train_df, val_df, features, args.rounds, int(config['min_group_size']))
                    row = {
                        'fold': fold.name,
                        'train_start': str(fold.train_start.date()),
                        'train_end': str(fold.train_end.date()),
                        'validation_start': str(fold.validation_start.date()),
                        'validation_end': str(fold.validation_end.date()),
                        'model': 'xgb_rank_pairwise',
                        'input_window': input_window,
                        'feature_variant': variant,
                        'feature_type': feature_type,
                        'feature_num': feature_num,
                        'factor_group': group_name,
                        **metrics,
                        'seconds': time.time() - tic,
                    }
                    rows.append(row)
                    completed.add(row_key)
                    payload = {
                        'experiment': 'xgb_rank_pairwise_window_factor_family_scan',
                        'seed': config['seed'],
                        'rounds': args.rounds,
                        'windows': args.windows,
                        'groups': args.groups,
                        'variants': args.variants,
                        'validation_config': config['validation'],
                        'rows': rows,
                        'summary': summarize(rows),
                        'elapsed_seconds': time.time() - started,
                    }
                    write_outputs(args.output, payload)
                    print(
                        f"done top5_excess={row['validation_top5_excess_return']:.6f} "
                        f"rank_ic={row['validation_rank_ic']:.6f} seconds={row['seconds']:.1f}",
                        flush=True,
                    )

    payload = {
        'experiment': 'xgb_rank_pairwise_window_factor_family_scan',
        'seed': config['seed'],
        'rounds': args.rounds,
        'windows': args.windows,
        'groups': args.groups,
        'variants': args.variants,
        'validation_config': config['validation'],
        'rows': rows,
        'summary': summarize(rows),
        'elapsed_seconds': time.time() - started,
    }
    write_outputs(args.output, payload)
    print(pd.DataFrame(payload['summary']).to_string(index=False))


if __name__ == '__main__':
    main()
