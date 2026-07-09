"""XGBoost pairwise 验证 window_1 专用因子族。"""

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import pandas as pd

from code.config import config as base_config
from code.features.baseline import preprocess_stock_data_samples
from code.features.window_1 import WINDOW_1_FEATURE_GROUPS, window_1_group_columns
from code.features.windows import feature_num_for_window
from code.utils.runtime_split import load_market_data
from code.utils.validation import build_validation_plan

from .xgb_window_factor_scan import summarize, train_eval_fold, write_outputs, xsec_aliases_for_columns


def set_seed(seed: int) -> None:
    """固定随机种子。"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def select_window_1_features(all_features: list[str], group_name: str, use_xsec: bool) -> list[str]:
    """选择 window_1 因子族，并可追加对应横截面派生特征。"""
    if group_name == 'all_window_1':
        return list(all_features)

    selected = [column for column in window_1_group_columns(group_name) if column in all_features]
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


def load_existing(path: Path) -> dict:
    """读取已有结果，用于断点续跑。"""
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return {'rows': [], 'summary': []}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=200)
    parser.add_argument('--variants', nargs='+', default=['raw', 'xsec'])
    parser.add_argument('--groups', nargs='+', default=['all_window_1', *WINDOW_1_FEATURE_GROUPS.keys()])
    parser.add_argument('--output', type=Path, default=Path('output/xgb_pairwise_window1_deep_scan.json'))
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
        (row['feature_variant'], row['factor_group'], row['fold'])
        for row in rows
    }
    dataset_cache = {}

    for variant in args.variants:
        feature_type = 'window_1+xsec' if variant == 'xsec' else 'window_1'
        feature_num = feature_num_for_window(1, feature_type)
        for fold in folds:
            if fold.name not in dataset_cache:
                dataset_cache[fold.name] = {}
            if feature_num not in dataset_cache[fold.name]:
                print(f'prepare window=1 variant={variant} fold={fold.name} feature_num={feature_num}', flush=True)
                train_samples = plan.get_train_samples(fold, 1)
                val_samples = plan.get_validation_samples(fold, 1)
                train_df, all_features = preprocess_stock_data_samples(train_samples, feature_num, stockid2idx)
                val_df, _ = preprocess_stock_data_samples(val_samples, feature_num, stockid2idx)
                train_df = train_df.dropna(subset=['label']).sort_values(['日期', '股票代码']).reset_index(drop=True)
                val_df = val_df.dropna(subset=['label']).sort_values(['日期', '股票代码']).reset_index(drop=True)
                dataset_cache[fold.name][feature_num] = (train_df, val_df, all_features)

            train_df, val_df, all_features = dataset_cache[fold.name][feature_num]
            for group_name in args.groups:
                row_key = (variant, group_name, fold.name)
                if row_key in completed:
                    continue
                features = select_window_1_features(all_features, group_name, variant == 'xsec')
                print(f'train window=1 variant={variant} factor={group_name} fold={fold.name} features={len(features)}', flush=True)
                tic = time.time()
                metrics = train_eval_fold(params, train_df, val_df, features, args.rounds, int(config['min_group_size']))
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
                    'factor_group': group_name,
                    **metrics,
                    'seconds': time.time() - tic,
                }
                rows.append(row)
                completed.add(row_key)
                payload = {
                    'experiment': 'xgb_rank_pairwise_window1_deep_scan',
                    'seed': config['seed'],
                    'rounds': args.rounds,
                    'variants': args.variants,
                    'groups': args.groups,
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
        'experiment': 'xgb_rank_pairwise_window1_deep_scan',
        'seed': config['seed'],
        'rounds': args.rounds,
        'variants': args.variants,
        'groups': args.groups,
        'validation_config': config['validation'],
        'rows': rows,
        'summary': summarize(rows),
        'elapsed_seconds': time.time() - started,
    }
    write_outputs(args.output, payload)
    print(pd.DataFrame(payload['summary']).to_string(index=False))


if __name__ == '__main__':
    main()
