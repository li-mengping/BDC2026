"""XGBoost 39 特征滚动验证与有限 Optuna 搜索。"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import optuna
import xgboost as xgb

from code.config import config as base_config
from code.features.baseline import date_group_sizes, preprocess_stock_data_samples
from code.features.windows import feature_num_for_window
from code.models.xgboost.loss import topk_return_metrics, xgb_rank_return_metrics
from code.models.xgboost.model import choose_best_iteration
from code.utils.runtime_split import load_market_data
from code.utils.validation import build_validation_plan


def evaluate_fold(params: dict, train_df, val_df, features: list[str], rounds: int, min_group_size: int) -> dict:
    train_grouped, train_groups = date_group_sizes(train_df, min_group_size)
    val_grouped, val_groups = date_group_sizes(val_df, min_group_size)
    train_matrix = xgb.DMatrix(
        train_grouped[features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32),
        label=train_grouped['label'].to_numpy(dtype=np.float32),
        missing=np.nan,
        group=np.asarray(train_groups, dtype=np.uint32),
        feature_names=features,
    )
    val_matrix = xgb.DMatrix(
        val_grouped[features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32),
        label=val_grouped['label'].to_numpy(dtype=np.float32),
        missing=np.nan,
        group=np.asarray(val_groups, dtype=np.uint32),
        feature_names=features,
    )
    evals_result = {}
    booster = xgb.train(
        params=params,
        dtrain=train_matrix,
        num_boost_round=rounds,
        evals=[(val_matrix, 'validation')],
        custom_metric=xgb_rank_return_metrics,
        maximize=True,
        evals_result=evals_result,
        verbose_eval=False,
    )
    selection = choose_best_iteration(
        [float(value) for value in evals_result['validation']['top5_excess_return']],
        'validation_top5_excess_return',
    )
    pred = booster.predict(val_matrix, iteration_range=(0, selection['best_iteration'] + 1))
    top5 = topk_return_metrics(pred, val_matrix.get_label(), val_groups, top_k=5)
    return {
        'validation_top5_excess_return': top5['pred_top5_excess_return_avg'],
        'validation_top5_excess_return_std': top5['pred_top5_excess_return_std'],
        'validation_top5_excess_positive_rate': top5['pred_top5_excess_positive_rate'],
        'validation_rank_ic': float(evals_result['validation']['rank_ic'][selection['best_iteration']]),
        'best_iteration': selection['best_iteration'],
    }


def aggregate(rows: list[dict]) -> dict:
    values = np.asarray([row['validation_top5_excess_return'] for row in rows], dtype=np.float64)
    return {
        'mean': float(values.mean()),
        'worst': float(values.min()),
        'std': float(values.std(ddof=0)),
        'positive_rate': float(np.mean([row['validation_top5_excess_positive_rate'] for row in rows])),
        'rank_ic': float(np.mean([row['validation_rank_ic'] for row in rows])),
        'best_iterations': [int(row['best_iteration']) for row in rows],
    }


def score_candidate(item: dict) -> float:
    return item['mean'] + 0.5 * item['worst'] - 0.25 * item['std']


def select_candidate(baseline: dict, trials: list[dict]) -> dict:
    if not trials:
        return {'number': None, 'improves_baseline': False, 'reason': 'no completed trials'}
    pareto = []
    for candidate in trials:
        dominated = any(
            other['mean'] >= candidate['mean']
            and other['worst'] >= candidate['worst']
            and other['std'] <= candidate['std']
            and (
                other['mean'] > candidate['mean']
                or other['worst'] > candidate['worst']
                or other['std'] < candidate['std']
            )
            for other in trials
        )
        if not dominated:
            pareto.append(candidate)
    chosen = max(pareto, key=score_candidate)
    chosen = dict(chosen)
    chosen['pareto_size'] = len(pareto)
    chosen['selection_score'] = score_candidate(chosen)
    chosen['improves_baseline'] = bool(
        chosen['mean'] > baseline['mean'] and chosen['worst'] >= baseline['worst']
    )
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=8)
    parser.add_argument('--rounds', type=int, default=160)
    parser.add_argument('--output', type=Path, default=Path('.agents/loops/loop-001/evidence/xgb_optuna.json'))
    args = parser.parse_args()
    config = copy.deepcopy(base_config)
    config['validation'] = {**config['validation'], 'mode': 'rolling_kfold'}
    config['model_names'] = ['xgb_rank_pairwise']
    model_config = config['model_params']['xgb_rank_pairwise']
    input_window = int(model_config['input_window'])
    feature_num = feature_num_for_window(input_window, model_config['feature_type'])
    features_cache = []
    raw_df = load_market_data(config)
    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
    plan = build_validation_plan(raw_df, config)
    for fold in plan.validation_splits():
        train_samples = plan.get_train_samples(fold, input_window)
        val_samples = plan.get_validation_samples(fold, input_window)
        train_df, features = preprocess_stock_data_samples(train_samples, feature_num, stockid2idx)
        val_df, _ = preprocess_stock_data_samples(val_samples, feature_num, stockid2idx)
        features_cache.append((fold.name, train_df.dropna(subset=['label']).sort_values(['日期', '股票代码']), val_df.dropna(subset=['label']).sort_values(['日期', '股票代码']), features))

    base_params = {
        key: value for key, value in model_config.items()
        if key not in {'input_window', 'feature_type', 'num_boost_round'}
    }
    base_params['disable_default_eval_metric'] = 1
    baseline_rows = [evaluate_fold(base_params, train_df, val_df, features, args.rounds, int(config['min_group_size'])) for _, train_df, val_df, features in features_cache]
    baseline = aggregate(baseline_rows)

    def objective(trial: optuna.Trial) -> tuple[float, float, float, float]:
        params = dict(base_params)
        params.update({
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'eta': trial.suggest_float('eta', 0.01, 0.06, log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 10.0, 100.0, log=True),
            'lambda': trial.suggest_float('lambda', 5.0, 60.0, log=True),
            'alpha': trial.suggest_float('alpha', 0.0, 1.0),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'max_delta_step': trial.suggest_float('max_delta_step', 0.0, 2.0),
        })
        started = time.perf_counter()
        rows = [evaluate_fold(params, train_df, val_df, features, args.rounds, int(config['min_group_size'])) for _, train_df, val_df, features in features_cache]
        metrics = aggregate(rows)
        trial.set_user_attr('metrics', metrics)
        trial.set_user_attr('seconds', time.perf_counter() - started)
        return metrics['mean'], metrics['worst'], metrics['std'], float(time.perf_counter() - started)

    study = optuna.create_study(
        directions=['maximize', 'maximize', 'minimize', 'minimize'],
        sampler=optuna.samplers.TPESampler(seed=int(config['seed']), multivariate=True),
        study_name='bdc2026-xgb-rolling',
    )
    study.enqueue_trial({
        key: base_params[key] for key in ('max_depth', 'eta', 'min_child_weight', 'lambda', 'alpha', 'subsample', 'colsample_bytree', 'max_delta_step')
    })
    study.optimize(objective, n_trials=args.trials, timeout=None, show_progress_bar=False)
    completed = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        metrics = trial.user_attrs['metrics']
        completed.append({'number': trial.number, **metrics, 'seconds': trial.user_attrs['seconds'], 'params': trial.params})
    selected = select_candidate(baseline, completed)
    payload = {
        'experiment': 'xgb_rank_pairwise_39_optuna_rolling',
        'seed': config['seed'],
        'rounds': args.rounds,
        'trials_requested': args.trials,
        'validation_config': config['validation'],
        'baseline': baseline,
        'baseline_rows': baseline_rows,
        'trials': completed,
        'selected': selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'baseline': baseline, 'selected': selected}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
