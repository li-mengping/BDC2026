"""在统一滚动证据契约下比较 CatBoost YetiRankPairwise 与 XGBoost 基线。"""

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostRanker, Pool

from code.config import config as base_config
from code.experiments.xgb_crossfit_checkpoint import METRICS, crossfit_summary, select_round
from code.experiments.xgb_stock_identity_ablation import pareto_decision
from code.features.baseline import date_group_sizes, preprocess_stock_data_samples
from code.features.windows import feature_num_for_window
from code.models.ranking_common import group_ids_from_sizes, relevance_from_returns, set_seed
from code.models.xgboost.loss import rank_ic_score, topk_return_metrics
from code.utils.runtime_split import load_market_data
from code.utils.validation import build_validation_plan


def train_curve(
    params: dict,
    train_df,
    validation_df,
    features: list[str],
    rounds: int,
    min_group_size: int,
) -> dict[str, list[float]]:
    train_grouped, train_groups = date_group_sizes(train_df, min_group_size)
    validation_grouped, validation_groups = date_group_sizes(validation_df, min_group_size)
    train_label = train_grouped['label'].to_numpy(dtype=np.float32)
    validation_label = validation_grouped['label'].to_numpy(dtype=np.float32)
    train_pool = Pool(
        train_grouped[features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32),
        label=relevance_from_returns(train_label, train_groups, 30),
        group_id=group_ids_from_sizes(train_groups),
        feature_names=features,
    )
    validation_pool = Pool(
        validation_grouped[features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32),
        group_id=group_ids_from_sizes(validation_groups),
        feature_names=features,
    )
    model = CatBoostRanker(iterations=rounds, **params)
    model.fit(train_pool)
    ptr = np.concatenate([[0], np.cumsum(np.asarray(validation_groups, dtype=np.int64))])
    curves = {metric: [] for metric in METRICS}
    for round_index in range(rounds):
        prediction = model.predict(validation_pool, ntree_end=round_index + 1)
        top5 = topk_return_metrics(prediction, validation_label, validation_groups, top_k=5)
        curves['top5_excess_return'].append(top5['pred_top5_excess_return_avg'])
        curves['top5_excess_std'].append(top5['pred_top5_excess_return_std'])
        curves['top5_excess_min'].append(top5['pred_top5_excess_return_min'])
        curves['top5_excess_positive_rate'].append(top5['pred_top5_excess_positive_rate'])
        curves['rank_ic'].append(rank_ic_score(prediction, validation_label, ptr))
    return curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=160)
    parser.add_argument(
        '--output', type=Path,
        default=Path('.agents/loops/loop-006/evidence/catboost_family.json'),
    )
    args = parser.parse_args()
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    config = copy.deepcopy(base_config)
    config['validation'] = {**config['validation'], 'mode': 'rolling_kfold'}
    xgb_config = config['model_params']['xgb_rank_pairwise']
    cat_config = config['catboost']['model_params']['cat_yetirank_pairwise']
    input_window = int(xgb_config['input_window'])
    feature_num = feature_num_for_window(input_window, xgb_config['feature_type'])
    params = {
        key: value for key, value in cat_config.items()
        if key not in {'input_window', 'feature_type', 'num_boost_round', 'max_relevance'}
    }
    params['devices'] = '0'
    set_seed(int(config['seed']))

    baseline_path = root / '.agents' / 'loops' / 'loop-004' / 'evidence' / 'top5_objective.json'
    baseline_evidence = json.loads(baseline_path.read_text(encoding='utf-8'))
    baseline = crossfit_summary(baseline_evidence['results']['rank_pairwise']['curves'])

    raw_df = load_market_data(config)
    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {stock_id: index for index, stock_id in enumerate(stock_ids)}
    plan = build_validation_plan(raw_df, config)
    curves = {}
    for fold in plan.validation_splits():
        train_samples = plan.get_train_samples(fold, input_window)
        validation_samples = plan.get_validation_samples(fold, input_window)
        train_df, features = preprocess_stock_data_samples(train_samples, feature_num, stockid2idx)
        validation_df, _ = preprocess_stock_data_samples(validation_samples, feature_num, stockid2idx)
        curves[fold.name] = train_curve(
            params,
            train_df.dropna(subset=['label']).sort_values(['日期', '股票代码']),
            validation_df.dropna(subset=['label']).sort_values(['日期', '股票代码']),
            features,
            args.rounds,
            int(config['min_group_size']),
        )
    candidate = crossfit_summary(curves)
    manifest = json.loads((root / 'data' / 'manifest.json').read_text(encoding='utf-8'))
    commit = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()
    payload = {
        'experiment': 'catboost_yetirank_pairwise_family_crossfit',
        'parent_commit': commit,
        'data_sha256': manifest['data']['sha256'],
        'seed': config['seed'],
        'rounds': args.rounds,
        'selection_scope': 'leave_one_fold_out_checkpoint; holdout_not_loaded',
        'baseline_source': str(baseline_path.relative_to(root)),
        'baseline': baseline,
        'candidate': candidate,
        'candidate_params': params,
        'production_round_selection': select_round(curves, '__all_folds__'),
        'curves': curves,
        'decision': pareto_decision(baseline, candidate),
        'elapsed_seconds': time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({
        'baseline': baseline,
        'candidate': candidate,
        'decision': payload['decision'],
        'production_round_selection': payload['production_round_selection'],
        'elapsed_seconds': payload['elapsed_seconds'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
