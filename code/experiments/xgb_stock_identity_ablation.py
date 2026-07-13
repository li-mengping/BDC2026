"""比较正式 XGBoost 是否需要股票身份特征 instrument。"""

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

from code.config import config as base_config
from code.experiments.xgb_optuna import aggregate, evaluate_fold
from code.features.baseline import preprocess_stock_data_samples
from code.features.windows import feature_num_for_window
from code.utils.runtime_split import load_market_data
from code.utils.validation import build_validation_plan


def pareto_decision(baseline: dict, candidate: dict) -> dict:
    """要求均值、最差折和标准差均不退化，且至少一项严格改善。"""
    no_regression = (
        candidate['mean'] >= baseline['mean']
        and candidate['worst'] >= baseline['worst']
        and candidate['std'] <= baseline['std']
    )
    strict = (
        candidate['mean'] > baseline['mean']
        or candidate['worst'] > baseline['worst']
        or candidate['std'] < baseline['std']
    )
    accepted = bool(no_regression and strict)
    return {
        'accepted': accepted,
        'status': 'validated_candidate' if accepted else 'rejected_no_pareto_improvement',
        'rule': 'mean>=baseline and worst>=baseline and std<=baseline, with one strict improvement',
    }


def git_commit(root: Path) -> str:
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=160)
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('.agents/loops/loop-001/evidence/stock_identity_ablation.json'),
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
    for variant, exclude in {'with_instrument': set(), 'without_instrument': {'instrument'}}.items():
        rows = []
        for fold_name, train_df, validation_df, features in fold_data:
            selected_features = [feature for feature in features if feature not in exclude]
            metrics = evaluate_fold(
                params,
                train_df,
                validation_df,
                selected_features,
                args.rounds,
                int(config['min_group_size']),
            )
            rows.append({'fold': fold_name, 'feature_count': len(selected_features), **metrics})
        variants[variant] = {'aggregate': aggregate(rows), 'rows': rows}

    baseline = variants['with_instrument']['aggregate']
    candidate = variants['without_instrument']['aggregate']
    manifest = json.loads((root / 'data' / 'manifest.json').read_text(encoding='utf-8'))
    payload = {
        'experiment': 'xgb_stock_identity_ablation',
        'parent_commit': git_commit(root),
        'data_sha256': manifest['data']['sha256'],
        'seed': config['seed'],
        'rounds': args.rounds,
        'validation_config': config['validation'],
        'hypothesis': '删除 instrument 可减少股票身份记忆并改善跨折稳定性',
        'selection_scope': 'rolling_validation_only; holdout_not_loaded',
        'variants': variants,
        'decision': pareto_decision(baseline, candidate),
        'elapsed_seconds': time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
