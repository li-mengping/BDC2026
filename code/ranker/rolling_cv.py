import json
import multiprocessing as mp
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import config
from .rankic import topk_return_metrics, xgb_rank_ic_metric, xgb_rank_return_metrics
from .train import choose_top5_iteration, make_dmatrix, target_range
from ..features.baseline import preprocess_stock_data_samples
from ..utils.runtime_split import load_market_data, split_runtime_data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def cv_fold_ranges(runtime) -> list[dict]:
    """生成 holdout test 之前的滚动验证折。"""
    folds = int(config.get('rolling_cv_folds', 4))
    validation_weeks = int(config.get('rolling_cv_validation_weeks', 4))
    gap_weeks = int(config.get('rolling_cv_gap_weeks', 1))
    min_train_weeks = int(config.get('rolling_cv_min_train_weeks', 120))
    start_pos = int(runtime.week_starts.searchsorted(runtime.start_date, side='left'))
    fold_ranges = []

    for fold_idx in range(folds):
        val_end_pos = runtime.holdout_test_start_pos - fold_idx * validation_weeks
        val_start_pos = val_end_pos - validation_weeks
        train_end_pos = val_start_pos - gap_weeks
        if train_end_pos <= start_pos:
            continue
        if train_end_pos - start_pos < min_train_weeks:
            continue

        fold_ranges.append({
            'fold': folds - fold_idx,
            'train_start': pd.Timestamp(runtime.week_starts[start_pos]).normalize(),
            'train_end': pd.Timestamp(runtime.week_starts[train_end_pos]).normalize(),
            'validation_start': pd.Timestamp(runtime.week_starts[val_start_pos]).normalize(),
            'validation_end': pd.Timestamp(runtime.week_starts[val_end_pos]).normalize(),
        })

    if not fold_ranges:
        raise ValueError('no rolling cv folds can be built with current config')
    return sorted(fold_ranges, key=lambda item: item['validation_start'])


def train_pairwise_fold(train_df: pd.DataFrame, val_df: pd.DataFrame, features: list[str]) -> tuple[dict, dict]:
    """训练一个 pairwise fold 并返回指标。"""
    dtrain, train_groups = make_dmatrix(train_df, features)
    dval, val_groups = make_dmatrix(val_df, features)
    params = dict(config['xgb_params']['xgb_rank_pairwise'])
    params['disable_default_eval_metric'] = 1

    evals_result = {}
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(config.get('rolling_cv_num_boost_round', config['num_boost_round'])),
        evals=[(dtrain, 'train'), (dval, 'validation')],
        custom_metric=xgb_rank_return_metrics,
        maximize=True,
        evals_result=evals_result,
        verbose_eval=False,
    )

    validation_top5 = [float(value) for value in evals_result.get('validation', {}).get('top5_return', [])]
    selection = choose_top5_iteration(validation_top5)
    iteration_range = (0, int(selection['best_iteration']) + 1)
    val_pred = booster.predict(dval, iteration_range=iteration_range)
    val_rank_ic = xgb_rank_ic_metric(val_pred, dval)[1]
    val_top5 = topk_return_metrics(val_pred, dval.get_label(), val_groups, top_k=5)
    val_top10 = topk_return_metrics(val_pred, dval.get_label(), val_groups, top_k=10)

    metrics = {
        'best_iteration': int(selection['best_iteration']),
        'best_selection_score': float(selection['top5_return']),
        'validation_rank_ic': float(val_rank_ic),
        'validation_top5_return': float(val_top5['pred_top5_return_avg']),
        'validation_top10_return': float(val_top10['pred_top10_return_avg']),
        'validation_top5_group_returns': val_top5['pred_top5_group_returns'],
        'validation_top10_group_returns': val_top10['pred_top10_group_returns'],
        'train_groups': len(train_groups),
        'validation_groups': len(val_groups),
        'train_rows': int(dtrain.num_row()),
        'validation_rows': int(dval.num_row()),
    }
    return metrics, evals_result


def summarize(fold_df: pd.DataFrame) -> dict:
    returns = fold_df['validation_top5_return'].to_numpy(dtype=np.float64)
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    return {
        'folds': int(len(fold_df)),
        'mean_top5_return': float(returns.mean()),
        'std_top5_return': std,
        'mean_over_std_top5': float(returns.mean() / std) if std > 0 else None,
        'hit_rate_top5': float((returns > 0.0).mean()),
        'min_top5_return': float(returns.min()),
        'max_top5_return': float(returns.max()),
        'mean_rank_ic': float(fold_df['validation_rank_ic'].mean()),
        'mean_top10_return': float(fold_df['validation_top10_return'].mean()),
    }


def main() -> dict:
    set_seed(int(config['seed']))
    output_dir = Path(config['output_dir']) / 'rolling_cv'
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_market_data(config)
    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
    runtime = split_runtime_data(raw_df, config)
    fold_ranges = cv_fold_ranges(runtime)

    samples = runtime.get_stock_data(runtime.start_date, runtime.holdout_test_target_start)
    all_df, features = preprocess_stock_data_samples(samples, config['feature_num'], stockid2idx)
    all_df = all_df.dropna(subset=['label']).sort_values(['日期', '股票代码']).reset_index(drop=True)

    print(f"Rolling CV feature_num={config['feature_num']} | features={len(features)}")
    print(f"Samples: {target_range(all_df)[0]} to {target_range(all_df)[1]} | rows={len(all_df)}")

    rows = []
    evals_payload = {}
    for idx, fold in enumerate(fold_ranges, start=1):
        train_df = all_df[(all_df['日期'] >= fold['train_start']) & (all_df['日期'] < fold['train_end'])].copy()
        val_df = all_df[
            (all_df['日期'] >= fold['validation_start'])
            & (all_df['日期'] < fold['validation_end'])
        ].copy()
        metrics, evals_result = train_pairwise_fold(train_df, val_df, features)
        row = {
            'fold': idx,
            'train_start': str(fold['train_start'].date()),
            'train_end': str(fold['train_end'].date()),
            'validation_start': str(fold['validation_start'].date()),
            'validation_end': str(fold['validation_end'].date()),
            **{key: value for key, value in metrics.items() if not key.endswith('_group_returns')},
        }
        rows.append(row)
        evals_payload[f'fold_{idx}'] = {
            'fold': row,
            'validation_top5_group_returns': metrics['validation_top5_group_returns'],
            'validation_top10_group_returns': metrics['validation_top10_group_returns'],
            'evals_result': evals_result,
        }
        print(
            f"fold={idx} val={row['validation_start']}..{row['validation_end']} "
            f"iter={row['best_iteration']} top5={row['validation_top5_return']:.6f} "
            f"top10={row['validation_top10_return']:.6f} rank_ic={row['validation_rank_ic']:.6f}"
        )

    fold_df = pd.DataFrame(rows)
    summary = summarize(fold_df)
    fold_df.to_csv(output_dir / 'folds.csv', index=False)
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'folds': rows}, f, ensure_ascii=False, indent=2)
    with open(output_dir / 'evals.json', 'w', encoding='utf-8') as f:
        json.dump(evals_payload, f, ensure_ascii=False, indent=2)

    print('Rolling CV summary:')
    for key, value in summary.items():
        print(f'{key}: {value}')
    print(f'fold metrics written to: {output_dir / "folds.csv"}')
    return summary


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
