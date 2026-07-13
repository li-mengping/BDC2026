"""比较正式 pairwise 与 Top5 对齐 NDCG 排序目标。"""

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from code.config import config as base_config
from code.experiments.xgb_crossfit_checkpoint import METRICS, crossfit_summary, select_round
from code.experiments.xgb_stock_identity_ablation import pareto_decision
from code.features.baseline import date_group_sizes, preprocess_stock_data_samples
from code.features.windows import feature_num_for_window
from code.models.ranking_common import relevance_from_returns, set_seed
from code.models.xgboost.loss import rank_ic_score, topk_return_metrics
from code.utils.runtime_split import load_market_data
from code.utils.validation import build_validation_plan


def make_matrix(frame, features: list[str], min_group_size: int, relevance: bool) -> tuple[xgb.DMatrix, np.ndarray, list[int]]:
    grouped, groups = date_group_sizes(frame, min_group_size)
    raw_label = grouped['label'].to_numpy(dtype=np.float32)
    label = relevance_from_returns(raw_label, groups, 30) if relevance else raw_label
    matrix = xgb.DMatrix(
        grouped[features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32),
        label=label,
        missing=np.nan,
        group=np.asarray(groups, dtype=np.uint32),
        feature_names=features,
    )
    return matrix, raw_label, groups


def train_curve(
    params: dict,
    train_df,
    validation_df,
    features: list[str],
    rounds: int,
    min_group_size: int,
    relevance: bool,
) -> dict[str, list[float]]:
    dtrain, _, _ = make_matrix(train_df, features, min_group_size, relevance)
    dvalidation, raw_label, validation_groups = make_matrix(
        validation_df, features, min_group_size, relevance,
    )
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=rounds,
        verbose_eval=False,
    )
    ptr = np.concatenate([[0], np.cumsum(np.asarray(validation_groups, dtype=np.int64))])
    curves = {metric: [] for metric in METRICS}
    for round_index in range(rounds):
        prediction = booster.predict(dvalidation, iteration_range=(0, round_index + 1))
        top5 = topk_return_metrics(prediction, raw_label, validation_groups, top_k=5)
        curves['top5_excess_return'].append(top5['pred_top5_excess_return_avg'])
        curves['top5_excess_std'].append(top5['pred_top5_excess_return_std'])
        curves['top5_excess_min'].append(top5['pred_top5_excess_return_min'])
        curves['top5_excess_positive_rate'].append(top5['pred_top5_excess_positive_rate'])
        curves['rank_ic'].append(rank_ic_score(prediction, raw_label, ptr))
    return curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=160)
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('.agents/loops/loop-004/evidence/top5_objective.json'),
    )
    args = parser.parse_args()
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    config = copy.deepcopy(base_config)
    config['validation'] = {**config['validation'], 'mode': 'rolling_kfold'}
    model_config = config['model_params']['xgb_rank_pairwise']
    input_window = int(model_config['input_window'])
    feature_num = feature_num_for_window(input_window, model_config['feature_type'])
    base_params = {
        key: value for key, value in model_config.items()
        if key not in {'input_window', 'feature_type', 'num_boost_round'}
    }
    base_params['disable_default_eval_metric'] = 1
    variants = {
        'rank_pairwise': {'params': dict(base_params), 'relevance': False},
        'rank_ndcg_top5': {
            'params': {
                **base_params,
                'objective': 'rank:ndcg',
                'lambdarank_pair_method': 'topk',
                'lambdarank_num_pair_per_sample': 5,
            },
            'relevance': True,
        },
    }
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

    results = {}
    for name, variant in variants.items():
        curves = {}
        for fold_name, train_df, validation_df, features in fold_data:
            curves[fold_name] = train_curve(
                variant['params'],
                train_df,
                validation_df,
                features,
                args.rounds,
                int(config['min_group_size']),
                variant['relevance'],
            )
        results[name] = {
            'params': variant['params'],
            'training_label': 'weekly_relevance_0_30' if variant['relevance'] else 'raw_weekly_return',
            'evaluation_label': 'raw_weekly_return',
            'crossfit': crossfit_summary(curves),
            'production_round_selection': select_round(curves, '__all_folds__'),
            'curves': curves,
        }

    baseline = results['rank_pairwise']['crossfit']
    candidate = results['rank_ndcg_top5']['crossfit']
    manifest = json.loads((root / 'data' / 'manifest.json').read_text(encoding='utf-8'))
    commit = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()
    payload = {
        'experiment': 'xgb_top5_objective_crossfit',
        'parent_commit': commit,
        'data_sha256': manifest['data']['sha256'],
        'seed': config['seed'],
        'rounds': args.rounds,
        'selection_scope': 'leave_one_fold_out_checkpoint; holdout_not_loaded',
        'results': results,
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
