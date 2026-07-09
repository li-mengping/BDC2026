"""搜索 window=1 的组合策略，目标超过旧版 all_window+xsec 基准。"""

import argparse
import copy
import itertools
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from code.config import config as base_config
from code.features.baseline import FEATURE_COLUMNS, date_group_sizes, preprocess_stock_data_samples
from code.features.windows import BASE_FEATURE_COLUMNS as OLD_BASE_FEATURE_COLUMNS
from code.features.windows import feature_num_for_window
from code.models.xgboost.loss import rank_ic_score, topk_return_metrics, xgb_rank_return_metrics
from code.models.xgboost.model import choose_best_iteration
from code.utils.runtime_split import load_market_data
from code.utils.validation import build_validation_plan

from .xgb_window1_deep_scan import select_window_1_features
from .xgb_window_factor_scan import select_features


KEY_COLUMNS = ['sample_id', '日期', '股票代码', 'feature_date', 'target_week', 'label']
PRED_COLUMNS = ['candidate', 'fold', '日期', '股票代码', 'label', 'prediction']


@dataclass(frozen=True)
class FeatureSource:
    """一个候选策略可由多个 window=1 特征源拼接而成。"""

    family: str
    variant: str
    group: str = 'all'
    columns_file: str | None = None


@dataclass(frozen=True)
class Candidate:
    """单个可训练候选策略。"""

    name: str
    sources: tuple[FeatureSource, ...]
    rounds: int = 220
    params_patch: dict = field(default_factory=dict)


def set_seed(seed: int) -> None:
    """固定随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def read_columns(path: str | None) -> list[str]:
    """读取列名文件。"""
    if not path:
        return []
    return [line.strip() for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def build_candidates(args: argparse.Namespace) -> list[Candidate]:
    """声明本轮要比较的 window=1 策略。"""
    raw_finalists = 'output/window1_raw_finalist_columns.txt'
    xsec_finalists = 'output/window1_xsec_finalist_columns.txt'
    params_default = {}
    params_shallow = {'max_depth': 3, 'min_child_weight': 45.0, 'lambda': 25.0, 'eta': 0.03, 'colsample_bytree': 0.85}
    params_deep = {'max_depth': 6, 'min_child_weight': 18.0, 'lambda': 12.0, 'eta': 0.022, 'colsample_bytree': 0.78}
    params_stable = {'max_depth': 4, 'min_child_weight': 55.0, 'lambda': 30.0, 'eta': 0.02, 'subsample': 0.9}
    params_seed7 = {'seed': 7}
    params_seed202 = {'seed': 202}

    old_all_xsec = (FeatureSource('old', 'xsec', 'all'),)
    old_all_raw = (FeatureSource('old', 'raw', 'all'),)
    old_position_xsec = (FeatureSource('old', 'xsec', 'position'),)
    old_liquidity_xsec = (FeatureSource('old', 'xsec', 'liquidity'),)
    old_raw_position = (FeatureSource('old', 'raw', 'position'),)
    old_raw_liquidity = (FeatureSource('old', 'raw', 'liquidity'),)
    new_daily_raw = (FeatureSource('new', 'raw', 'daily_state'),)
    new_trend_raw = (FeatureSource('new', 'raw', 'trend_efficiency'),)
    new_position_raw = (FeatureSource('new', 'raw', 'position_breakout'),)
    new_reversal_xsec = (FeatureSource('new', 'xsec', 'reversal'),)
    new_liquidity_xsec = (FeatureSource('new', 'xsec', 'liquidity_impact'),)
    new_compact_raw = (FeatureSource('new', 'raw', 'columns', raw_finalists),)
    new_compact_xsec = (FeatureSource('new', 'xsec', 'columns', xsec_finalists),)
    hybrid_raw = old_all_xsec + new_compact_raw
    hybrid_xsec = old_all_xsec + new_compact_xsec
    hybrid_both = old_all_xsec + new_compact_raw + new_compact_xsec
    hybrid_groups = old_all_xsec + new_daily_raw + new_trend_raw + new_position_raw

    candidates = [
        Candidate('old_all_xsec_default', old_all_xsec, args.rounds, params_default),
        Candidate('old_all_xsec_shallow', old_all_xsec, args.rounds, params_shallow),
        Candidate('old_all_xsec_deep', old_all_xsec, args.rounds, params_deep),
        Candidate('old_all_xsec_stable', old_all_xsec, args.rounds, params_stable),
        Candidate('old_all_xsec_seed7', old_all_xsec, args.rounds, params_seed7),
        Candidate('old_all_xsec_seed202', old_all_xsec, args.rounds, params_seed202),
        Candidate('old_all_raw_default', old_all_raw, args.rounds, params_default),
        Candidate('old_position_xsec_default', old_position_xsec, args.rounds, params_default),
        Candidate('old_liquidity_xsec_default', old_liquidity_xsec, args.rounds, params_default),
        Candidate('old_raw_position_default', old_raw_position, args.rounds, params_default),
        Candidate('old_raw_liquidity_default', old_raw_liquidity, args.rounds, params_default),
        Candidate('new_daily_raw_default', new_daily_raw, args.rounds, params_default),
        Candidate('new_trend_raw_default', new_trend_raw, args.rounds, params_default),
        Candidate('new_position_raw_default', new_position_raw, args.rounds, params_default),
        Candidate('new_reversal_xsec_default', new_reversal_xsec, args.rounds, params_default),
        Candidate('new_liquidity_xsec_default', new_liquidity_xsec, args.rounds, params_default),
        Candidate('new_compact_raw_default', new_compact_raw, args.rounds, params_default),
        Candidate('new_compact_xsec_default', new_compact_xsec, args.rounds, params_default),
        Candidate('hybrid_old_new_raw_finalists', hybrid_raw, args.rounds, params_default),
        Candidate('hybrid_old_new_xsec_finalists', hybrid_xsec, args.rounds, params_default),
        Candidate('hybrid_old_new_all_finalists', hybrid_both, args.rounds, params_default),
        Candidate('hybrid_old_new_groups', hybrid_groups, args.rounds, params_default),
        Candidate('hybrid_old_new_all_finalists_stable', hybrid_both, args.rounds, params_stable),
        Candidate('hybrid_old_new_all_finalists_shallow', hybrid_both, args.rounds, params_shallow),
    ]
    stable_grid = {
        'old_all_xsec_grid_depth3_mcw55_l30': {'max_depth': 3, 'min_child_weight': 55.0, 'lambda': 30.0, 'eta': 0.02, 'subsample': 0.9},
        'old_all_xsec_grid_depth5_mcw55_l30': {'max_depth': 5, 'min_child_weight': 55.0, 'lambda': 30.0, 'eta': 0.02, 'subsample': 0.9},
        'old_all_xsec_grid_mcw45_l30': {'max_depth': 4, 'min_child_weight': 45.0, 'lambda': 30.0, 'eta': 0.02, 'subsample': 0.9},
        'old_all_xsec_grid_mcw70_l30': {'max_depth': 4, 'min_child_weight': 70.0, 'lambda': 30.0, 'eta': 0.02, 'subsample': 0.9},
        'old_all_xsec_grid_mcw55_l20': {'max_depth': 4, 'min_child_weight': 55.0, 'lambda': 20.0, 'eta': 0.02, 'subsample': 0.9},
        'old_all_xsec_grid_mcw55_l40': {'max_depth': 4, 'min_child_weight': 55.0, 'lambda': 40.0, 'eta': 0.02, 'subsample': 0.9},
        'old_all_xsec_grid_eta015': {'max_depth': 4, 'min_child_weight': 55.0, 'lambda': 30.0, 'eta': 0.015, 'subsample': 0.9},
        'old_all_xsec_grid_eta025': {'max_depth': 4, 'min_child_weight': 55.0, 'lambda': 30.0, 'eta': 0.025, 'subsample': 0.9},
        'old_all_xsec_grid_sub10': {'max_depth': 4, 'min_child_weight': 55.0, 'lambda': 30.0, 'eta': 0.02, 'subsample': 1.0},
        'old_all_xsec_grid_col085': {'max_depth': 4, 'min_child_weight': 55.0, 'lambda': 30.0, 'eta': 0.02, 'subsample': 0.9, 'colsample_bytree': 0.85},
        'old_all_xsec_grid_balanced': {'max_depth': 4, 'min_child_weight': 65.0, 'lambda': 35.0, 'eta': 0.018, 'subsample': 0.9},
        'old_all_xsec_grid_fast_regular': {'max_depth': 3, 'min_child_weight': 70.0, 'lambda': 40.0, 'eta': 0.025, 'subsample': 0.9},
        'old_all_xsec_grid_mcw50_l30': {'max_depth': 4, 'min_child_weight': 50.0, 'lambda': 30.0, 'eta': 0.02, 'subsample': 0.9},
        'old_all_xsec_grid_mcw60_l30': {'max_depth': 4, 'min_child_weight': 60.0, 'lambda': 30.0, 'eta': 0.02, 'subsample': 0.9},
        'old_all_xsec_grid_mcw55_l25': {'max_depth': 4, 'min_child_weight': 55.0, 'lambda': 25.0, 'eta': 0.02, 'subsample': 0.9},
        'old_all_xsec_grid_mcw55_l35': {'max_depth': 4, 'min_child_weight': 55.0, 'lambda': 35.0, 'eta': 0.02, 'subsample': 0.9},
        'old_all_xsec_grid_col065': {'max_depth': 4, 'min_child_weight': 55.0, 'lambda': 30.0, 'eta': 0.02, 'subsample': 0.9, 'colsample_bytree': 0.65},
        'old_all_xsec_grid_col070': {'max_depth': 4, 'min_child_weight': 55.0, 'lambda': 30.0, 'eta': 0.02, 'subsample': 0.9, 'colsample_bytree': 0.70},
        'old_all_xsec_grid_col080': {'max_depth': 4, 'min_child_weight': 55.0, 'lambda': 30.0, 'eta': 0.02, 'subsample': 0.9, 'colsample_bytree': 0.80},
        'old_all_xsec_grid_sub085': {'max_depth': 4, 'min_child_weight': 55.0, 'lambda': 30.0, 'eta': 0.02, 'subsample': 0.85},
        'old_all_xsec_grid_sub095': {'max_depth': 4, 'min_child_weight': 55.0, 'lambda': 30.0, 'eta': 0.02, 'subsample': 0.95},
        'old_all_xsec_grid_alpha0': {'max_depth': 4, 'min_child_weight': 55.0, 'lambda': 30.0, 'alpha': 0.0, 'eta': 0.02, 'subsample': 0.9},
        'old_all_xsec_grid_alpha05': {'max_depth': 4, 'min_child_weight': 55.0, 'lambda': 30.0, 'alpha': 0.5, 'eta': 0.02, 'subsample': 0.9},
        'old_all_xsec_grid_gamma01': {'max_depth': 4, 'min_child_weight': 55.0, 'lambda': 30.0, 'gamma': 0.1, 'eta': 0.02, 'subsample': 0.9},
        'old_all_xsec_grid_gamma05': {'max_depth': 4, 'min_child_weight': 55.0, 'lambda': 30.0, 'gamma': 0.5, 'eta': 0.02, 'subsample': 0.9},
        'old_all_xsec_grid_maxbin128': {'max_depth': 4, 'min_child_weight': 55.0, 'lambda': 30.0, 'eta': 0.02, 'subsample': 0.9, 'max_bin': 128},
        'old_all_xsec_grid_collevel08': {'max_depth': 4, 'min_child_weight': 55.0, 'lambda': 30.0, 'eta': 0.02, 'subsample': 0.9, 'colsample_bylevel': 0.8},
        'old_all_xsec_grid_colnode08': {'max_depth': 4, 'min_child_weight': 55.0, 'lambda': 30.0, 'eta': 0.02, 'subsample': 0.9, 'colsample_bynode': 0.8},
    }
    candidates.extend(
        Candidate(name, old_all_xsec, args.rounds, patch)
        for name, patch in stable_grid.items()
    )
    filtered_old_sources = {
        'old_all_xsec_filter_no_mkt_stable': FeatureSource('old', 'xsec', 'all_no_mkt'),
        'old_all_xsec_filter_no_base_stable': FeatureSource('old', 'xsec', 'all_no_base'),
        'old_all_xsec_filter_no_base_mkt_stable': FeatureSource('old', 'xsec', 'all_no_base_mkt'),
        'old_all_xsec_filter_no_xs_stable': FeatureSource('old', 'xsec', 'all_no_xs'),
        'old_all_xsec_filter_xs_mkt_only_stable': FeatureSource('old', 'xsec', 'xs_mkt_only'),
        'old_all_xsec_filter_xs_only_stable': FeatureSource('old', 'xsec', 'xs_only'),
    }
    candidates.extend(
        Candidate(name, (source,), args.rounds, params_stable)
        for name, source in filtered_old_sources.items()
    )
    if args.candidates:
        wanted = set(args.candidates)
        candidates = [candidate for candidate in candidates if candidate.name in wanted]
    return candidates


def source_feature_num(source: FeatureSource) -> str:
    """解析源特征名。"""
    if source.family == 'old':
        feature_type = 'window+xsec' if source.variant == 'xsec' else 'window'
    elif source.family == 'new':
        feature_type = 'window_1+xsec' if source.variant == 'xsec' else 'window_1'
    else:
        raise ValueError(f'unknown feature family: {source.family}')
    return feature_num_for_window(1, feature_type)


def select_source_features(source: FeatureSource, all_features: list[str]) -> list[str]:
    """按源声明选择特征列。"""
    if source.group == 'columns':
        selected = [column for column in read_columns(source.columns_file) if column in all_features]
        if not selected:
            raise ValueError(f'no selected columns for {source}')
        return selected
    if source.family == 'old':
        if source.group == 'all':
            return list(all_features)
        if source.group == 'all_no_mkt':
            return [column for column in all_features if not column.startswith('MKT_')]
        if source.group == 'all_no_base':
            base_columns = set(OLD_BASE_FEATURE_COLUMNS)
            return [column for column in all_features if column not in base_columns]
        if source.group == 'all_no_base_mkt':
            base_columns = set(OLD_BASE_FEATURE_COLUMNS)
            return [column for column in all_features if column not in base_columns and not column.startswith('MKT_')]
        if source.group == 'all_no_xs':
            return [column for column in all_features if not column.startswith('XS_')]
        if source.group == 'xs_mkt_only':
            return [column for column in all_features if column.startswith('XS_') or column.startswith('MKT_')]
        if source.group == 'xs_only':
            return [column for column in all_features if column.startswith('XS_')]
        return select_features(all_features, 1, source.group, source.variant == 'xsec')
    if source.family == 'new':
        if source.group == 'all':
            return list(all_features)
        return select_window_1_features(all_features, source.group, source.variant == 'xsec')
    raise ValueError(f'unknown source: {source}')


def prepare_feature_frame(
    dataset_cache: dict,
    plan,
    fold,
    source: FeatureSource,
    stockid2idx: dict[str, int],
    min_group_size: int,
    split: str,
) -> tuple[pd.DataFrame, list[str]]:
    """按源准备 train/validation 特征帧。"""
    feature_num = source_feature_num(source)
    key = (feature_num, fold.name, split)
    if key not in dataset_cache:
        samples = plan.get_train_samples(fold, 1) if split == 'train' else plan.get_validation_samples(fold, 1)
        frame, all_features = preprocess_stock_data_samples(samples, feature_num, stockid2idx)
        frame = frame.dropna(subset=['label']).sort_values(['日期', '股票代码']).reset_index(drop=True)
        dataset_cache[key] = (frame, all_features)
    frame, all_features = dataset_cache[key]
    features = select_source_features(source, all_features)
    filtered, groups = date_group_sizes(frame, min_group_size)
    if groups:
        return filtered.reset_index(drop=True), features
    raise ValueError(f'no valid groups for {source} {fold.name} {split}')


def build_candidate_frame(
    dataset_cache: dict,
    plan,
    fold,
    candidate: Candidate,
    stockid2idx: dict[str, int],
    min_group_size: int,
    split: str,
) -> tuple[pd.DataFrame, list[str]]:
    """把候选策略声明的多个源拼成一个训练帧。"""
    combined = None
    features: list[str] = []
    blocks = []
    for source in candidate.sources:
        frame, source_features = prepare_feature_frame(dataset_cache, plan, fold, source, stockid2idx, min_group_size, split)
        if combined is None:
            combined = frame[KEY_COLUMNS].copy()
        else:
            left_key = combined[KEY_COLUMNS].reset_index(drop=True)
            right_key = frame[KEY_COLUMNS].reset_index(drop=True)
            if not left_key.equals(right_key):
                raise ValueError(f'feature source rows do not align for {candidate.name} {fold.name} {split}: {source}')
        for column in source_features:
            if column in features:
                continue
            features.append(column)
        blocks.append(frame[[column for column in source_features if column in features]].copy())
    if combined is None or not features:
        raise ValueError(f'empty candidate: {candidate.name}')
    if blocks:
        feature_block = pd.concat(blocks, axis=1)
        feature_block = feature_block.loc[:, ~feature_block.columns.duplicated()]
        combined = pd.concat([combined, feature_block[features]], axis=1)
    return combined, features


def make_dmatrix(df: pd.DataFrame, features: list[str], min_group_size: int) -> tuple[xgb.DMatrix, list[int], pd.DataFrame]:
    """构造 DMatrix，并保留分组后的行。"""
    grouped_df, groups = date_group_sizes(df, min_group_size)
    x = grouped_df[features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    y = grouped_df['label'].to_numpy(dtype=np.float32)
    dmatrix = xgb.DMatrix(x, label=y, missing=np.nan, group=np.asarray(groups, dtype=np.uint32), feature_names=features)
    return dmatrix, groups, grouped_df.reset_index(drop=True)


def train_eval_candidate(
    params: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    rounds: int,
    min_group_size: int,
) -> tuple[dict, np.ndarray, pd.DataFrame]:
    """训练并返回指标、验证预测和验证行。"""
    dtrain, train_groups, _ = make_dmatrix(train_df, features, min_group_size)
    dval, val_groups, val_grouped = make_dmatrix(val_df, features, min_group_size)
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
    top5 = topk_return_metrics(pred, dval.get_label(), val_groups, top_k=5)
    top10 = topk_return_metrics(pred, dval.get_label(), val_groups, top_k=10)
    ptr = np.concatenate([[0], np.cumsum(np.asarray(val_groups, dtype=np.int64))])
    metrics = {
        'features': len(features),
        'rounds': rounds,
        'best_iteration': selection['best_iteration'],
        'best_selection_score': selection['score'],
        'validation_rank_ic': rank_ic_score(pred, dval.get_label(), ptr),
        'validation_universe_return': top5['benchmark_top5_return_avg'],
        'validation_top5_return': top5['pred_top5_return_avg'],
        'validation_top5_excess_return': top5['pred_top5_excess_return_avg'],
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
    return metrics, pred, val_grouped


def summarize_metrics(rows: list[dict], group_col: str) -> pd.DataFrame:
    """聚合候选或 ensemble 的 fold 指标。"""
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    summary = frame.groupby(group_col, sort=False).agg(
        fold_count=('fold', 'nunique'),
        validation_top5_excess_return_mean=('validation_top5_excess_return', 'mean'),
        validation_top5_excess_return_std=('validation_top5_excess_return', 'std'),
        worst_fold=('validation_top5_excess_return', 'min'),
        validation_top10_excess_return_mean=('validation_top10_excess_return', 'mean'),
        validation_rank_ic_mean=('validation_rank_ic', 'mean'),
        validation_top5_excess_positive_rate_mean=('validation_top5_excess_positive_rate', 'mean'),
        validation_top5_precision_mean=('validation_top5_precision', 'mean'),
        features_mean=('features', 'mean') if 'features' in frame.columns else ('fold', 'size'),
        best_iteration_mean=('best_iteration', 'mean') if 'best_iteration' in frame.columns else ('fold', 'size'),
    ).reset_index()
    return summary.sort_values(
        ['validation_top5_excess_return_mean', 'worst_fold', 'validation_top10_excess_return_mean'],
        ascending=[False, False, False],
    )


def normalize_by_date(frame: pd.DataFrame, pred_col: str, method: str) -> pd.Series:
    """按日期把候选预测归一化，便于跨模型融合。"""
    grouped = frame.groupby(['fold', '日期'], sort=False)[pred_col]
    if method == 'rank':
        return grouped.rank(pct=True, method='average')
    if method == 'z':
        mean = grouped.transform('mean')
        std = grouped.transform('std').replace(0.0, np.nan)
        return ((frame[pred_col] - mean) / std).fillna(0.0)
    raise ValueError(f'unknown normalize method: {method}')


def evaluate_prediction_frame(frame: pd.DataFrame, pred_col: str, name: str) -> list[dict]:
    """按 fold 评估一列预测。"""
    rows = []
    for fold, fold_df in frame.groupby('fold', sort=False):
        fold_df = fold_df.sort_values(['日期', '股票代码']).reset_index(drop=True)
        _, groups = date_group_sizes(fold_df, 1)
        pred = fold_df[pred_col].to_numpy(dtype=np.float64)
        label = fold_df['label'].to_numpy(dtype=np.float64)
        top5 = topk_return_metrics(pred, label, groups, top_k=5)
        top10 = topk_return_metrics(pred, label, groups, top_k=10)
        ptr = np.concatenate([[0], np.cumsum(np.asarray(groups, dtype=np.int64))])
        rows.append({
            'ensemble': name,
            'fold': fold,
            'validation_rank_ic': rank_ic_score(pred, label, ptr),
            'validation_top5_return': top5['pred_top5_return_avg'],
            'validation_top5_excess_return': top5['pred_top5_excess_return_avg'],
            'validation_top5_excess_return_min': top5['pred_top5_excess_return_min'],
            'validation_top5_excess_positive_rate': top5['pred_top5_excess_positive_rate'],
            'validation_top5_precision': top5['pred_top5_precision_avg'],
            'validation_top10_return': top10['pred_top10_return_avg'],
            'validation_top10_excess_return': top10['pred_top10_excess_return_avg'],
            'validation_top10_excess_return_min': top10['pred_top10_excess_return_min'],
            'validation_top10_excess_positive_rate': top10['pred_top10_excess_positive_rate'],
            'validation_top10_precision': top10['pred_top10_precision_avg'],
        })
    return rows


def build_wide_predictions(predictions: pd.DataFrame, candidates: list[str], method: str) -> pd.DataFrame:
    """把长表预测转成每个候选一列，并做日期内归一化。"""
    parts = []
    for candidate in candidates:
        part = predictions[predictions['candidate'].eq(candidate)].copy()
        if part.empty:
            continue
        part[candidate] = normalize_by_date(part, 'prediction', method)
        parts.append(part[['fold', '日期', '股票代码', 'label', candidate]])
    if not parts:
        raise ValueError('no predictions for ensemble')
    wide = parts[0]
    for part in parts[1:]:
        wide = wide.merge(part, on=['fold', '日期', '股票代码', 'label'], how='inner')
    return wide


def ensemble_search(predictions: pd.DataFrame, candidate_summary: pd.DataFrame, max_candidates: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """搜索简单 rank/zscore 平均和二元加权融合。"""
    ordered = candidate_summary['candidate'].head(max_candidates).tolist()
    rows = []
    summaries = []

    for method in ['rank', 'z']:
        wide = build_wide_predictions(predictions, ordered, method)
        for size in [2, 3, 4, 5, min(8, len(ordered))]:
            if size > len(ordered):
                continue
            cols = ordered[:size]
            name = f'{method}_mean_top{size}'
            wide[name] = wide[cols].mean(axis=1)
            rows.extend(evaluate_prediction_frame(wide[['fold', '日期', '股票代码', 'label', name]], name, name))

        for left, right in itertools.combinations(ordered[: min(10, len(ordered))], 2):
            cols = [left, right]
            pair = build_wide_predictions(predictions, cols, method)
            for weight in [0.2, 0.35, 0.5, 0.65, 0.8]:
                name = f'{method}_pair_{weight:.2f}_{left}__{right}'
                pair[name] = weight * pair[left] + (1.0 - weight) * pair[right]
                rows.extend(evaluate_prediction_frame(pair[['fold', '日期', '股票代码', 'label', name]], name, name))

    summary = summarize_metrics(rows, 'ensemble')
    if not summary.empty:
        summaries.append(summary)
    return pd.DataFrame(rows), pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()


def write_outputs(path: Path, rows: list[dict], predictions: pd.DataFrame, ensemble_rows: pd.DataFrame, ensemble_summary: pd.DataFrame) -> None:
    """保存搜索结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate_summary = summarize_metrics(rows, 'candidate')
    payload = {
        'experiment': 'xgb_window1_strategy_search',
        'rows': rows,
        'candidate_summary': candidate_summary.to_dict('records'),
        'ensemble_summary': ensemble_summary.to_dict('records'),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    pd.DataFrame(rows).to_csv(path.with_suffix('.rows.csv'), index=False)
    candidate_summary.to_csv(path.with_suffix('.summary.csv'), index=False)
    predictions.to_csv(path.with_suffix('.predictions.csv'), index=False)
    ensemble_rows.to_csv(path.with_suffix('.ensemble_rows.csv'), index=False)
    ensemble_summary.to_csv(path.with_suffix('.ensemble_summary.csv'), index=False)


def write_partial(path: Path, rows: list[dict], prediction_parts: list[pd.DataFrame]) -> None:
    """训练过程中保存已完成 fold，避免长实验被中断后丢进度。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path.with_suffix('.partial_rows.csv'), index=False)
    if prediction_parts:
        pd.concat(prediction_parts, ignore_index=True).to_csv(path.with_suffix('.partial_predictions.csv'), index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=220)
    parser.add_argument('--output', type=Path, default=Path('output/xgb_window1_strategy_search.json'))
    parser.add_argument('--candidates', nargs='*')
    parser.add_argument('--max-ensemble-candidates', type=int, default=14)
    args = parser.parse_args()

    config = copy.deepcopy(base_config)
    config['validation'] = {**config['validation'], 'mode': 'rolling_kfold'}
    params_base = {
        key: value
        for key, value in config['model_params']['xgb_rank_pairwise'].items()
        if key not in {'input_window', 'feature_type', 'num_boost_round'}
    }
    params_base['disable_default_eval_metric'] = 1
    set_seed(int(config['seed']))

    raw_df = load_market_data(config)
    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
    plan = build_validation_plan(raw_df, config)
    folds = plan.validation_splits()
    candidates = build_candidates(args)

    rows: list[dict] = []
    prediction_parts = []
    dataset_cache = {}
    started = time.time()

    for candidate in candidates:
        params = {**params_base, **candidate.params_patch}
        print(f'candidate={candidate.name} rounds={candidate.rounds} params_patch={candidate.params_patch}', flush=True)
        for fold in folds:
            print(f'prepare/train candidate={candidate.name} fold={fold.name}', flush=True)
            tic = time.time()
            train_df, features = build_candidate_frame(dataset_cache, plan, fold, candidate, stockid2idx, int(config['min_group_size']), 'train')
            val_df, _ = build_candidate_frame(dataset_cache, plan, fold, candidate, stockid2idx, int(config['min_group_size']), 'validation')
            metrics, pred, val_grouped = train_eval_candidate(params, train_df, val_df, features, candidate.rounds, int(config['min_group_size']))
            row = {
                'candidate': candidate.name,
                'fold': fold.name,
                'train_start': str(fold.train_start.date()),
                'train_end': str(fold.train_end.date()),
                'validation_start': str(fold.validation_start.date()),
                'validation_end': str(fold.validation_end.date()),
                **metrics,
                'seconds': time.time() - tic,
            }
            rows.append(row)
            pred_part = val_grouped[['日期', '股票代码', 'label']].copy()
            pred_part['candidate'] = candidate.name
            pred_part['fold'] = fold.name
            pred_part['prediction'] = pred
            prediction_parts.append(pred_part[PRED_COLUMNS])
            write_partial(args.output, rows, prediction_parts)
            print(
                f"done candidate={candidate.name} fold={fold.name} "
                f"top5_excess={row['validation_top5_excess_return']:.6f} "
                f"rank_ic={row['validation_rank_ic']:.6f} seconds={row['seconds']:.1f}",
                flush=True,
            )

    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame(columns=PRED_COLUMNS)
    candidate_summary = summarize_metrics(rows, 'candidate')
    print('candidate summary')
    print(candidate_summary.head(30).to_string(index=False))
    ensemble_rows, ensemble_summary = ensemble_search(predictions, candidate_summary, args.max_ensemble_candidates)
    print('ensemble summary')
    print(ensemble_summary.head(30).to_string(index=False))
    write_outputs(args.output, rows, predictions, ensemble_rows, ensemble_summary)
    print(f'elapsed_seconds={time.time() - started:.1f}')


if __name__ == '__main__':
    main()
