"""用 leave-one-fold-out 规则选择 checkpoint，复核股票身份消融。"""

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from code.config import config as base_config
from code.experiments.xgb_stock_identity_ablation import pareto_decision
from code.experiments.xgb_window_factor_scan import make_dmatrix, set_seed
from code.features.baseline import preprocess_stock_data_samples
from code.features.windows import feature_num_for_window
from code.models.xgboost.loss import xgb_rank_return_metrics
from code.utils.runtime_split import load_market_data
from code.utils.validation import build_validation_plan


METRICS = (
    'top5_excess_return',
    'top5_excess_std',
    'top5_excess_min',
    'top5_excess_positive_rate',
    'rank_ic',
)


def select_round(curves: dict[str, dict[str, list[float]]], held_out: str) -> dict:
    """只用非 held-out folds 的 Top5 曲线选择 round。"""
    training_folds = [name for name in sorted(curves) if name != held_out]
    if len(training_folds) < 2:
        raise ValueError('cross-fitted checkpoint selection needs at least two training folds')
    round_count = len(curves[training_folds[0]]['top5_excess_return'])
    candidates = []
    for round_index in range(round_count):
        values = np.asarray(
            [curves[name]['top5_excess_return'][round_index] for name in training_folds],
            dtype=np.float64,
        )
        mean = float(values.mean())
        worst = float(values.min())
        std = float(values.std(ddof=0))
        candidates.append({
            'round': round_index + 1,
            'mean': mean,
            'worst': worst,
            'std': std,
            'score': mean + 0.5 * worst - 0.25 * std,
        })
    selected = max(candidates, key=lambda item: (item['score'], -item['round']))
    return {**selected, 'selected_from_folds': training_folds, 'held_out_fold': held_out}


def crossfit_summary(curves: dict[str, dict[str, list[float]]]) -> dict:
    """为每个 outer fold 用其余 folds 选 round，再聚合一次性评估值。"""
    rows = []
    for held_out in sorted(curves):
        selection = select_round(curves, held_out)
        index = selection['round'] - 1
        fold_curve = curves[held_out]
        rows.append({
            'fold': held_out,
            'selected_round': selection['round'],
            'selected_from_folds': selection['selected_from_folds'],
            'selection_score': selection['score'],
            **{metric: float(fold_curve[metric][index]) for metric in METRICS},
        })
    excess = np.asarray([row['top5_excess_return'] for row in rows], dtype=np.float64)
    return {
        'mean': float(excess.mean()),
        'worst': float(excess.min()),
        'std': float(excess.std(ddof=0)),
        'positive_rate': float(np.mean([row['top5_excess_positive_rate'] for row in rows])),
        'rank_ic': float(np.mean([row['rank_ic'] for row in rows])),
        'selected_rounds': [row['selected_round'] for row in rows],
        'rows': rows,
    }


def train_curve(params: dict, train_df, validation_df, features: list[str], rounds: int, min_group_size: int) -> dict:
    dtrain, _ = make_dmatrix(train_df, features, min_group_size)
    dvalidation, _ = make_dmatrix(validation_df, features, min_group_size)
    history = {}
    xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=rounds,
        evals=[(dvalidation, 'validation')],
        custom_metric=xgb_rank_return_metrics,
        maximize=True,
        evals_result=history,
        verbose_eval=False,
    )
    values = history['validation']
    return {metric: [float(value) for value in values[metric]] for metric in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=160)
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('.agents/loops/loop-002/evidence/crossfit_checkpoint.json'),
    )
    args = parser.parse_args()
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    config = copy.deepcopy(base_config)
    config['validation'] = {**config['validation'], 'mode': 'rolling_kfold'}
    model_config = config['model_params']['xgb_rank_pairwise']
    input_window = int(model_config['input_window'])
    feature_num = feature_num_for_window(input_window, model_config['feature_type'])
    params = {
        key: value for key, value in model_config.items()
        if key not in {'input_window', 'feature_type', 'num_boost_round'}
    }
    params['disable_default_eval_metric'] = 1
    set_seed(int(config['seed']))

    raw_df = load_market_data(config)
    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {stock_id: index for index, stock_id in enumerate(stock_ids)}
    plan = build_validation_plan(raw_df, config)
    fold_data = []
    for fold in plan.validation_splits():
        train_samples = plan.get_train_samples(fold, input_window)
        validation_samples = plan.get_validation_samples(fold, input_window)
        train_df, features = preprocess_stock_data_samples(train_samples, feature_num, stockid2idx)
        validation_df, _ = preprocess_stock_data_samples(validation_samples, feature_num, stockid2idx)
        fold_data.append((
            fold.name,
            train_df.dropna(subset=['label']).sort_values(['日期', '股票代码']),
            validation_df.dropna(subset=['label']).sort_values(['日期', '股票代码']),
            features,
        ))

    variants = {}
    for variant, excluded in {'with_instrument': set(), 'without_instrument': {'instrument'}}.items():
        curves = {}
        for fold_name, train_df, validation_df, features in fold_data:
            selected_features = [feature for feature in features if feature not in excluded]
            curves[fold_name] = train_curve(
                params, train_df, validation_df, selected_features, args.rounds, int(config['min_group_size']),
            )
        variants[variant] = {
            'feature_count': len(fold_data[0][3]) - len(excluded),
            'crossfit': crossfit_summary(curves),
            'production_round_selection': select_round(curves, '__all_folds__'),
            'curves': curves,
        }

    baseline = variants['with_instrument']['crossfit']
    candidate = variants['without_instrument']['crossfit']
    manifest = json.loads((root / 'data' / 'manifest.json').read_text(encoding='utf-8'))
    commit = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()
    payload = {
        'experiment': 'xgb_crossfit_checkpoint_identity_ablation',
        'parent_commit': commit,
        'data_sha256': manifest['data']['sha256'],
        'seed': config['seed'],
        'rounds': args.rounds,
        'selection_scope': 'leave_one_fold_out_checkpoint; holdout_not_loaded',
        'validation_config': config['validation'],
        'variants': variants,
        'decision': pareto_decision(baseline, candidate),
        'elapsed_seconds': time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({
        'baseline': baseline,
        'candidate': candidate,
        'decision': payload['decision'],
        'elapsed_seconds': payload['elapsed_seconds'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
