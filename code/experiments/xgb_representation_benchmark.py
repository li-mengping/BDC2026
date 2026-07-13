"""E1：在冻结真实行情上比较 XGBoost 横截面表示与股票身份编码。

每个 outer rolling fold 只用于最终评估。checkpoint 在 outer-train 尾部的
inner validation 上选择，然后用完整 outer-train 重训；outer validation 从不进入训练日志。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

from code.config import config as base_config
from code.features.baseline import date_group_sizes, preprocess_stock_data_samples
from code.features.windows import feature_num_for_window
from code.models.spine import FoldSpec
from code.models.xgboost.loss import (
    group_ptr_from_sizes,
    rank_ic_score,
    topk_return_metrics,
    xgb_rank_return_metrics,
)
from code.models.xgboost.model import choose_best_iteration
from code.utils.runtime_split import build_stock_data_samples, load_market_data
from code.utils.validation import build_validation_plan


E1_VARIANTS = (
    'raw_ordered_identity',
    'identity_disabled',
    'cross_section_rank_identity_disabled',
    'cross_section_robust_zscore_identity_disabled',
    'categorical_identity',
)


@dataclass(frozen=True)
class RepresentationBenchmarkConfig:
    """E1 固定预算；四个变体必须共享同一实例。"""

    outer_folds: int = 4
    inner_validation_weeks: int = 4
    max_rounds: int = 200
    seed: int = 42
    top_k: int = 5
    nthread: int = 8
    device: str = 'cuda'

    def __post_init__(self) -> None:
        if self.outer_folds != 4:
            raise ValueError('E1 requires exactly four outer folds')
        if self.inner_validation_weeks < 1 or self.max_rounds < 1:
            raise ValueError('inner validation weeks and max rounds must be positive')
        if self.top_k < 1 or self.nthread < 1:
            raise ValueError('top_k and nthread must be positive')
        if self.device not in {'cpu', 'cuda'}:
            raise ValueError("device must be 'cpu' or 'cuda'")


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _frame_scope_hash(frame: pd.DataFrame) -> str:
    keys = frame.loc[:, ['日期', '股票代码', 'label']].copy()
    keys['日期'] = pd.to_datetime(keys['日期']).dt.strftime('%Y-%m-%d')
    payload = keys.to_csv(index=False, lineterminator='\n', float_format='%.12g')
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _git_state(root: Path) -> dict[str, object]:
    commit = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ['git', 'status', '--porcelain'], cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip())
    return {'commit': commit, 'working_tree_dirty': dirty}


def split_outer_inner(
    frame: pd.DataFrame,
    fold: FoldSpec,
    inner_validation_weeks: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """把 outer-train 的最后若干目标周切为 inner validation。"""
    if inner_validation_weeks < 1:
        raise ValueError('inner_validation_weeks must be positive')
    work = frame.copy()
    work['日期'] = pd.to_datetime(work['日期'], errors='coerce').dt.normalize()
    if work['日期'].isna().any():
        raise ValueError('invalid target dates in benchmark frame')
    work = work.sort_values(['日期', '股票代码']).reset_index(drop=True)

    outer_train = work[(work['日期'] >= fold.train_start) & (work['日期'] < fold.train_end)]
    outer_validation = work[
        (work['日期'] >= fold.validation_start) & (work['日期'] < fold.validation_end)
    ]
    train_weeks = pd.DatetimeIndex(outer_train['日期'].unique()).sort_values()
    if len(train_weeks) <= inner_validation_weeks:
        raise ValueError(f'{fold.name} has insufficient weeks for nested validation')
    inner_start = pd.Timestamp(train_weeks[-inner_validation_weeks]).normalize()
    inner_train = outer_train[outer_train['日期'] < inner_start]
    inner_validation = outer_train[outer_train['日期'] >= inner_start]
    if inner_train.empty or inner_validation.empty or outer_validation.empty:
        raise ValueError(f'{fold.name} contains an empty nested split')
    if not (
        inner_train['日期'].max()
        < inner_validation['日期'].min()
        < outer_validation['日期'].min()
    ):
        raise ValueError(f'{fold.name} nested split is not strictly ordered')
    return inner_train.copy(), inner_validation.copy(), outer_validation.copy()


def _cross_section_rank(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """按目标周把每个特征映射到 [-0.5, 0.5] 的稳定平均秩。"""
    result = {}
    grouped = frame.groupby('日期', sort=False)
    for feature in features:
        values = pd.to_numeric(frame[feature], errors='coerce')
        by_date = values.groupby(frame['日期'], sort=False)
        rank = by_date.rank(method='average', na_option='keep')
        count = by_date.transform('count').astype(float)
        scaled = (rank - 1.0) / (count - 1.0) - 0.5
        result[feature] = scaled.where(count > 1.0, 0.0).astype(np.float32)
    ranked = pd.DataFrame(result, index=frame.index)
    if len(ranked) != sum(len(group) for _, group in grouped):
        raise ValueError('cross-section rank changed sample rows')
    return ranked


def _cross_section_robust_zscore(
    frame: pd.DataFrame,
    features: Sequence[str],
    clip: float = 5.0,
) -> pd.DataFrame:
    """逐目标周计算 median/MAD；零 MAD 时使用同周稳健尺度回退。"""
    values = frame.loc[:, features].apply(pd.to_numeric, errors='coerce').astype(float)
    dates = frame['日期']
    grouped = values.groupby(dates, sort=False)
    median = grouped.transform('median')
    absolute_deviation = (values - median).abs()
    mad = absolute_deviation.groupby(dates, sort=False).transform('median')
    q25 = grouped.transform('quantile', q=0.25)
    q75 = grouped.transform('quantile', q=0.75)
    iqr_scale = (q75 - q25) / 1.349
    maximum_deviation = absolute_deviation.groupby(dates, sort=False).transform('max')
    scale = (1.4826 * mad).where(mad > 0.0, iqr_scale)
    scale = scale.where(scale > 0.0, maximum_deviation)
    scale = scale.where(scale > 0.0, 1.0)
    robust = ((values - median) / scale).clip(-clip, clip).astype(np.float32)
    if len(robust) != len(frame):
        raise ValueError('cross-section robust z-score changed sample rows')
    return robust


def build_variant_frame(
    frame: pd.DataFrame,
    base_features: Sequence[str],
    variant: str,
    stock_categories: Sequence[str],
) -> tuple[pd.DataFrame, list[str], bool]:
    """构造一个 E1 表示；返回特征、列名和是否启用原生类别处理。"""
    if variant not in E1_VARIANTS:
        raise ValueError(f'unsupported E1 variant: {variant}')
    feature_names = list(base_features)
    missing = set(feature_names) - set(frame.columns)
    if missing:
        raise ValueError(f'missing base features: {sorted(missing)}')
    if 'instrument' not in feature_names:
        raise ValueError('E1 baseline features must include instrument')

    if variant == 'raw_ordered_identity':
        return frame.loc[:, feature_names].astype(np.float32), feature_names, False

    non_identity = [name for name in feature_names if name != 'instrument']
    if variant == 'identity_disabled':
        return frame.loc[:, non_identity].astype(np.float32), non_identity, False

    if variant == 'cross_section_rank_identity_disabled':
        return _cross_section_rank(frame, non_identity), non_identity, False

    if variant == 'cross_section_robust_zscore_identity_disabled':
        return _cross_section_robust_zscore(frame, non_identity), non_identity, False

    categories = tuple(str(value) for value in stock_categories)
    unknown = sorted(set(frame['股票代码'].astype(str)) - set(categories))
    if unknown:
        raise ValueError(f'categorical identity contains unknown stocks: {unknown[:3]}')
    result = frame.loc[:, feature_names].astype(np.float32)
    result['instrument'] = pd.Categorical(
        frame['股票代码'].astype(str), categories=categories, ordered=False,
    )
    return result, feature_names, True


def _make_dmatrix(
    frame: pd.DataFrame,
    base_features: Sequence[str],
    variant: str,
    stock_categories: Sequence[str],
    min_group_size: int,
) -> tuple[xgb.DMatrix, list[int], list[str]]:
    grouped, groups = date_group_sizes(frame, min_group_size)
    values, feature_names, categorical = build_variant_frame(
        grouped, base_features, variant, stock_categories,
    )
    matrix = xgb.DMatrix(
        values,
        label=grouped['label'].to_numpy(np.float32),
        missing=np.nan,
        group=np.asarray(groups, np.uint32),
        enable_categorical=categorical,
    )
    if matrix.num_row() != len(grouped):
        raise ValueError('DMatrix changed shared sample rows')
    return matrix, groups, feature_names


def fit_nested_booster(
    inner_train: xgb.DMatrix,
    inner_validation: xgb.DMatrix,
    outer_train: xgb.DMatrix,
    outer_validation: xgb.DMatrix,
    params: dict,
    max_rounds: int,
) -> tuple[xgb.Booster, dict[str, object], dict[str, float]]:
    """仅用 inner validation 选轮，再在完整 outer-train 上重训。"""
    if max_rounds < 1:
        raise ValueError('max_rounds must be positive')
    if outer_validation is inner_validation or outer_validation is outer_train:
        raise ValueError('outer validation must be isolated from training matrices')
    train_params = dict(params)
    train_params['disable_default_eval_metric'] = 1
    history: dict[str, dict[str, list[float]]] = {}
    started = time.perf_counter()
    xgb.train(
        train_params,
        inner_train,
        num_boost_round=max_rounds,
        evals=[(inner_validation, 'inner_validation')],
        custom_metric=xgb_rank_return_metrics,
        maximize=True,
        evals_result=history,
        verbose_eval=False,
    )
    checkpoint_seconds = time.perf_counter() - started
    selection = choose_best_iteration(
        [float(value) for value in history['inner_validation']['top5_return']],
        'inner_validation_top5_return',
    )

    started = time.perf_counter()
    booster = xgb.train(
        train_params,
        outer_train,
        num_boost_round=int(selection['best_iteration']) + 1,
        evals=(),
        verbose_eval=False,
    )
    refit_seconds = time.perf_counter() - started
    return booster, selection, {
        'checkpoint_train_seconds': checkpoint_seconds,
        'outer_train_refit_seconds': refit_seconds,
        'train_seconds': checkpoint_seconds + refit_seconds,
    }


def _evaluate_outer(
    prediction: np.ndarray,
    outer_matrix: xgb.DMatrix,
    groups: list[int],
    dates: Sequence[pd.Timestamp],
    top_k: int,
) -> dict[str, object]:
    labels = outer_matrix.get_label()
    topk = topk_return_metrics(prediction, labels, groups, top_k=top_k)
    ptr = group_ptr_from_sizes(groups)
    weekly_rank_ic = [
        rank_ic_score(prediction[begin:end], labels[begin:end], np.asarray([0, end - begin]))
        for begin, end in zip(ptr[:-1], ptr[1:])
    ]
    weekly_returns = topk[f'pred_top{top_k}_group_returns']
    return {
        'validation_weeks': [pd.Timestamp(value).date().isoformat() for value in dates],
        'weekly_returns': [float(value) for value in weekly_returns],
        'mean_return': float(np.mean(weekly_returns)),
        'worst_return': float(np.min(weekly_returns)),
        'return_std': float(np.std(weekly_returns, ddof=0)),
        'positive_rate': float(np.mean(np.asarray(weekly_returns) > 0.0)),
        'rank_ic': float(np.mean(weekly_rank_ic)),
        'weekly_rank_ic': [float(value) for value in weekly_rank_ic],
        'universe_return': float(topk[f'benchmark_top{top_k}_return_avg']),
        'excess_return_diagnostic': float(topk[f'pred_top{top_k}_excess_return_avg']),
    }


def _fold_variant(
    frame: pd.DataFrame,
    fold: FoldSpec,
    base_features: Sequence[str],
    stock_categories: Sequence[str],
    variant: str,
    params: dict,
    benchmark: RepresentationBenchmarkConfig,
    min_group_size: int,
) -> dict[str, object]:
    inner_train, inner_validation, outer_validation = split_outer_inner(
        frame, fold, benchmark.inner_validation_weeks,
    )
    outer_train = pd.concat([inner_train, inner_validation], ignore_index=True).sort_values(
        ['日期', '股票代码'],
    )
    matrices = [
        _make_dmatrix(part, base_features, variant, stock_categories, min_group_size)
        for part in (inner_train, inner_validation, outer_train, outer_validation)
    ]
    d_inner_train, d_inner_validation, d_outer_train, d_outer_validation = (
        value[0] for value in matrices
    )
    outer_groups = matrices[-1][1]
    feature_names = matrices[-1][2]
    shared_rows = [matrix.num_row() for matrix in (
        d_inner_train, d_inner_validation, d_outer_train, d_outer_validation,
    )]
    expected_rows = [len(inner_train), len(inner_validation), len(outer_train), len(outer_validation)]
    if shared_rows != expected_rows:
        raise ValueError(f'{fold.name}/{variant} did not preserve shared rows')

    booster, selection, runtime = fit_nested_booster(
        d_inner_train, d_inner_validation, d_outer_train, d_outer_validation,
        params, benchmark.max_rounds,
    )
    started = time.perf_counter()
    prediction = booster.predict(d_outer_validation)
    predict_seconds = time.perf_counter() - started
    validation_dates = pd.DatetimeIndex(outer_validation['日期'].unique()).sort_values()
    model_bytes = len(booster.save_raw(raw_format='ubj'))
    return {
        'fold': fold.name,
        'checkpoint_train_scope_sha256': _frame_scope_hash(inner_train),
        'checkpoint_validation_scope_sha256': _frame_scope_hash(inner_validation),
        'outer_train_scope_sha256': _frame_scope_hash(outer_train),
        'sample_scope_sha256': _frame_scope_hash(outer_validation),
        'checkpoint': selection,
        'metrics': _evaluate_outer(
            prediction, d_outer_validation, outer_groups, validation_dates, benchmark.top_k,
        ),
        'runtime': {
            **runtime,
            'predict_seconds': predict_seconds,
            'model_bytes': model_bytes,
            'peak_vram_bytes': None,
            'peak_vram_note': 'XGBoost Python API does not expose allocator peak memory',
            'accelerator': benchmark.device,
        },
        'feature_count': len(feature_names),
        'categorical_identity': variant == 'categorical_identity',
    }


def _aggregate(results: Sequence[dict[str, object]]) -> dict[str, object]:
    weekly_returns = [
        value for result in results for value in result['metrics']['weekly_returns']
    ]
    weekly_rank_ic = [
        value for result in results for value in result['metrics']['weekly_rank_ic']
    ]
    return {
        'weekly_returns': weekly_returns,
        'mean_return': float(np.mean(weekly_returns)),
        'worst_return': float(np.min(weekly_returns)),
        'return_std': float(np.std(weekly_returns, ddof=0)),
        'positive_rate': float(np.mean(np.asarray(weekly_returns) > 0.0)),
        'rank_ic': float(np.mean(weekly_rank_ic)),
        'train_seconds': float(sum(result['runtime']['train_seconds'] for result in results)),
        'predict_seconds': float(sum(result['runtime']['predict_seconds'] for result in results)),
        'model_bytes': int(max(result['runtime']['model_bytes'] for result in results)),
    }


def _pareto_decisions(models: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    """相对 legacy baseline 生成可审计的 observed-dimension Pareto 决策。"""
    baseline_name = 'raw_ordered_identity'
    baseline = models[baseline_name]['aggregate']
    maximize = ('mean_return', 'worst_return', 'positive_rate', 'rank_ic')
    minimize = ('return_std', 'train_seconds', 'predict_seconds', 'model_bytes')
    decisions = {}
    for name, payload in models.items():
        candidate = payload['aggregate']
        regressions = [key for key in maximize if candidate[key] < baseline[key]]
        regressions += [key for key in minimize if candidate[key] > baseline[key]]
        strict = [key for key in maximize if candidate[key] > baseline[key]]
        strict += [key for key in minimize if candidate[key] < baseline[key]]
        observed_pareto = not regressions and bool(strict)
        decisions[name] = {
            'baseline': name == baseline_name,
            'observed_dimension_pareto_improvement': observed_pareto,
            'regressions': regressions,
            'strict_improvements': strict,
            'resource_complete': False,
            'resource_gap': ['peak_vram_bytes'],
            'promote': False,
            'reason': 'baseline' if name == baseline_name else (
                'resource evidence incomplete' if observed_pareto else 'not a complete Pareto improvement'
            ),
        }
    return decisions


def _prepare_shared_samples(project_config: dict):
    raw = load_market_data(project_config)
    plan = build_validation_plan(raw, project_config)
    folds = plan.validation_splits()
    if len(folds) != 4:
        raise ValueError('E1 validation plan must produce four outer folds')
    model_config = project_config['model_params']['xgb_rank_pairwise']
    input_window = int(model_config['input_window'])
    feature_num = feature_num_for_window(input_window, model_config['feature_type'])
    samples = build_stock_data_samples(
        raw,
        min(fold.train_start for fold in folds),
        max(fold.validation_end for fold in folds),
        input_window,
    )
    stock_categories = tuple(sorted(raw['股票代码'].unique()))
    stockid2idx = {stock_id: index for index, stock_id in enumerate(stock_categories)}
    frame, base_features = preprocess_stock_data_samples(samples, feature_num, stockid2idx)
    frame = frame.dropna(subset=['label']).sort_values(['日期', '股票代码'])
    frame, _ = date_group_sizes(frame, int(project_config['min_group_size']))
    return raw, folds, frame, base_features, stock_categories, feature_num


def run_representation_benchmark(
    benchmark: RepresentationBenchmarkConfig,
) -> dict[str, object]:
    """运行四折 E1；返回真实冻结数据上的可审计结果。"""
    np.random.seed(benchmark.seed)
    os.environ['PYTHONHASHSEED'] = str(benchmark.seed)
    project_config = copy.deepcopy(base_config)
    project_config['validation'] = {**project_config['validation'], 'mode': 'rolling_kfold'}
    raw, folds, frame, base_features, stock_categories, feature_num = _prepare_shared_samples(
        project_config,
    )
    if len(folds) != benchmark.outer_folds:
        raise ValueError('benchmark and validation plan outer folds differ')

    model_config = project_config['model_params']['xgb_rank_pairwise']
    params = {
        key: value for key, value in model_config.items()
        if key not in {'input_window', 'feature_type', 'num_boost_round'}
    }
    params.update({
        'seed': benchmark.seed,
        'nthread': benchmark.nthread,
        'device': benchmark.device,
        'tree_method': 'hist',
    })
    resource_contract = {
        'seed': benchmark.seed,
        'outer_folds': benchmark.outer_folds,
        'inner_validation_weeks': benchmark.inner_validation_weeks,
        'max_rounds': benchmark.max_rounds,
        'nthread': benchmark.nthread,
        'device': benchmark.device,
        'tree_method': params['tree_method'],
        'top_k': benchmark.top_k,
    }
    resource_hash = _canonical_hash(resource_contract)
    models = {}
    for variant in E1_VARIANTS:
        fold_results = [
            _fold_variant(
                frame, fold, base_features, stock_categories, variant, params,
                benchmark, int(project_config['min_group_size']),
            )
            for fold in folds
        ]
        models[variant] = {
            'resource_contract_sha256': resource_hash,
            'aggregate': _aggregate(fold_results),
            'folds': fold_results,
        }

    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root / project_config['data_path'] / project_config['data_manifest_file']).read_text(
            encoding='utf-8',
        ),
    )
    fold_contracts = []
    for fold in folds:
        inner_train, inner_validation, outer_validation = split_outer_inner(
            frame, fold, benchmark.inner_validation_weeks,
        )
        fold_contracts.append({
            'name': fold.name,
            'inner_train_weeks': int(inner_train['日期'].nunique()),
            'inner_validation_weeks': int(inner_validation['日期'].nunique()),
            'outer_validation_weeks': int(outer_validation['日期'].nunique()),
            'inner_train_rows': len(inner_train),
            'inner_validation_rows': len(inner_validation),
            'outer_train_rows': len(inner_train) + len(inner_validation),
            'outer_validation_rows': len(outer_validation),
            'inner_validation_end': inner_validation['日期'].max().date().isoformat(),
            'outer_validation_start': outer_validation['日期'].min().date().isoformat(),
            'inner_train_scope_sha256': _frame_scope_hash(inner_train),
            'inner_validation_scope_sha256': _frame_scope_hash(inner_validation),
            'outer_train_scope_sha256': _frame_scope_hash(pd.concat(
                [inner_train, inner_validation], ignore_index=True,
            ).sort_values(['日期', '股票代码'])),
            'outer_validation_scope_sha256': _frame_scope_hash(outer_validation),
        })
    return {
        'schema_version': 1,
        'experiment': 'e1_xgboost_representation_identity_nested_rolling',
        'data_kind': 'real_frozen_market',
        'synthetic': False,
        'effect_evidence': True,
        'claim_boundary': (
            'four-fold nested walk-forward evidence; each outer window is isolated from its own '
            'fold training, but earlier outer windows may enter later-fold history; not an independent holdout'
        ),
        'data': {
            'file': project_config['data_path'].removeprefix('./') + '/' + project_config['full_data_file'],
            'manifest': project_config['data_path'].removeprefix('./') + '/' + project_config['data_manifest_file'],
            'sha256': manifest['data']['sha256'],
            'rows': manifest['data']['rows'],
            'date_max': manifest['data']['date_max'],
            'loaded_rows': len(raw),
            'shared_sample_rows': len(frame),
            'shared_sample_sha256': _frame_scope_hash(frame),
        },
        'label_mode': project_config['label_mode'],
        'sample_calendar_policy': project_config['sample_calendar_policy'],
        'data_cutoff': project_config['data_cutoff'],
        'primary_metric': 'absolute_top5_return',
        'diagnostic_metrics': ['excess_return', 'rank_ic'],
        'feature_type': feature_num,
        'identity_contract': {
            'raw_ordered_identity': 'legacy numeric stock index; intentionally retained only as baseline',
            'identity_disabled': 'stock identity excluded',
            'cross_section_rank_identity_disabled': 'date-local average rank in [-0.5, 0.5]; identity excluded',
            'cross_section_robust_zscore_identity_disabled': (
                'date-local median/MAD z-score clipped to [-5, 5]; identity excluded'
            ),
            'categorical_identity': 'XGBoost native categorical split over canonical stock codes',
        },
        'nested_selection_contract': {
            'checkpoint_metric': 'inner_validation_top5_return',
            'refit': 'selected rounds on complete outer-train',
            'outer_validation': (
                'prediction and metrics only for the same fold; walk-forward history may include '
                'earlier folds in later-fold training'
            ),
        },
        'resource_contract': resource_contract,
        'resource_contract_sha256': resource_hash,
        'runtime_environment': {
            'python': platform.python_version(),
            'platform': platform.system(),
            'logical_cpu_count': os.cpu_count(),
            'xgboost': xgb.__version__,
        },
        'folds': fold_contracts,
        'models': models,
        'pareto_contract': {
            'baseline': 'raw_ordered_identity',
            'maximize': ['mean_return', 'worst_return', 'positive_rate', 'rank_ic'],
            'minimize': ['return_std', 'train_seconds', 'predict_seconds', 'model_bytes'],
            'required_but_unavailable': ['peak_vram_bytes'],
        },
        'decisions': _pareto_decisions(models),
        'git': _git_state(root),
        'config': asdict(benchmark),
        'production_config_changed': False,
    }


def write_report(output: Path, report: dict[str, object], root: Path | None = None) -> None:
    """仅把报告写入仓库相对路径。"""
    if output.is_absolute() or '..' in output.parts:
        raise ValueError('output must be a repository-relative path without parent traversal')
    target = (root or Path.cwd()) / output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8',
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--max-rounds', type=int, default=200)
    parser.add_argument('--inner-validation-weeks', type=int, default=4)
    parser.add_argument('--nthread', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    benchmark = RepresentationBenchmarkConfig(
        inner_validation_weeks=args.inner_validation_weeks,
        max_rounds=args.max_rounds,
        seed=args.seed,
        nthread=args.nthread,
        device=args.device,
    )
    report = run_representation_benchmark(benchmark)
    root = Path(__file__).resolve().parents[2]
    write_report(args.output, report, root=root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
