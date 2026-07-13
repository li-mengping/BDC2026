"""在冻结真实行情上公平比较 XGBoost、朴素动量和 E0/E2/E3 结构。

所有模型共享同一个 rolling fold、同一验证截面和第1/第5交易日开盘标签。神经模型
只读取目标周之前 12 个完整周（60 个交易日）；本模块不修改生产配置或模型产物。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

from code.config import config as base_config
from code.experiments.neural_objectives import pairwise_ranking_loss
from code.features.baseline import preprocess_stock_data_samples
from code.features.windows import feature_num_for_window
from code.models.neural import build_ranker
from code.models.xgboost.loss import rank_ic_score, topk_return_metrics, xgb_rank_return_metrics
from code.models.xgboost.model import choose_best_iteration
from code.utils.runtime_split import build_stock_data_samples, load_market_data
from code.utils.validation import build_validation_plan


HISTORY_WEEKS = 12
HISTORY_DAYS = 60
NEURAL_FAMILIES = (
    ('e0_clean_room_official_contract_transformer', 'e0_transformer'),
    ('e2_causal_tcn', 'e2_tcn'),
    ('e3_cross_stock_attention', 'e3_cross_stock'),
)
SEQUENCE_FEATURES = (
    'open_to_previous_close',
    'close_to_open',
    'high_to_open',
    'low_to_open',
    'close_to_previous_close',
    'log_volume_change',
    'log_amount_change',
    'turnover_fraction',
)
FEATURE_CONTRACT = {
    'history': '12 contiguous full Monday-Friday weeks / 60 rows, strictly before target week',
    'open_to_previous_close': 'open[t] / close[t-1] - 1; first row is 0',
    'close_to_open': 'close[t] / open[t] - 1',
    'high_to_open': 'high[t] / open[t] - 1',
    'low_to_open': 'low[t] / open[t] - 1',
    'close_to_previous_close': 'close[t] / close[t-1] - 1; first row is 0',
    'log_volume_change': 'log1p(volume[t]) - log1p(volume[t-1]); first row is 0',
    'log_amount_change': 'log1p(amount[t]) - log1p(amount[t-1]); first row is 0',
    'turnover_fraction': 'turnover[t] / 100',
    'normalization': 'fixed clipping to [-5, 5]; no validation or future statistics are fitted',
    'missing_non_price': 'within-sequence forward fill using past rows only; leading missing values become 0',
    'missing_price': 'rejected',
}


@dataclass(frozen=True)
class BenchmarkConfig:
    mode: str
    folds: int
    inner_validation_weeks: int
    neural_epochs: int
    xgb_rounds: int
    hidden_dim: int
    num_heads: int
    learning_rate: float
    seed: int = 42
    top_k: int = 5
    nthread: int = 8

    @classmethod
    def for_mode(cls, mode: str, seed: int = 42) -> 'BenchmarkConfig':
        if mode == 'smoke':
            return cls(mode, folds=1, inner_validation_weeks=4, neural_epochs=1, xgb_rounds=20,
                       hidden_dim=8, num_heads=2, learning_rate=0.003, seed=seed)
        if mode == 'full':
            return cls(mode, folds=4, inner_validation_weeks=4, neural_epochs=30, xgb_rounds=200,
                       hidden_dim=32, num_heads=4, learning_rate=0.001, seed=seed)
        raise ValueError("mode must be 'smoke' or 'full'")

    def __post_init__(self) -> None:
        if self.folds < 1 or self.inner_validation_weeks < 1:
            raise ValueError('folds and inner validation weeks must be positive')
        if self.neural_epochs < 1 or self.xgb_rounds < 1:
            raise ValueError('training budgets must be positive')
        if self.nthread < 1:
            raise ValueError('nthread must be positive')


@dataclass(frozen=True)
class CrossSection:
    target_week: pd.Timestamp
    stock_ids: tuple[str, ...]
    values: np.ndarray
    returns: np.ndarray


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _section_scope_hash(sections: Sequence[CrossSection]) -> str:
    payload = [
        {
            'target_week': section.target_week.date().isoformat(),
            'stock_ids': list(section.stock_ids),
            'returns': [float(value) for value in section.returns],
        }
        for section in sections
    ]
    return _canonical_hash(payload)


def split_outer_inner_sections(
    outer_train_sections: Sequence[CrossSection],
    inner_validation_weeks: int,
) -> tuple[tuple[CrossSection, ...], tuple[CrossSection, ...]]:
    """从 outer-train 尾部切出 inner validation，outer validation 不参与切分。"""
    if inner_validation_weeks < 1:
        raise ValueError('inner validation weeks must be positive')
    sections = tuple(outer_train_sections)
    dates = pd.DatetimeIndex(section.target_week for section in sections)
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError('outer-train sections must be unique and ordered')
    if len(sections) <= inner_validation_weeks:
        raise ValueError('outer-train has insufficient weeks for nested validation')
    inner_train = sections[:-inner_validation_weeks]
    inner_validation = sections[-inner_validation_weeks:]
    if inner_train[-1].target_week >= inner_validation[0].target_week:
        raise ValueError('nested sections are not strictly ordered')
    return inner_train, inner_validation


def _build_shared_xgb_feature_frame(
    raw_df: pd.DataFrame,
    folds: Sequence[object],
    stockid2idx: dict[str, int],
    project_config: dict,
) -> tuple[pd.DataFrame, list[str], str]:
    """覆盖全部 outer folds 一次性生成因果 XGBoost 特征。"""
    model_config = project_config['model_params']['xgb_rank_pairwise']
    input_window = int(model_config['input_window'])
    if input_window != HISTORY_WEEKS:
        raise ValueError('shared XGBoost baseline must use the benchmark 12-week history')
    feature_num = feature_num_for_window(input_window, model_config['feature_type'])
    samples = build_stock_data_samples(
        raw_df,
        min(pd.Timestamp(fold.train_start) for fold in folds),
        max(pd.Timestamp(fold.validation_end) for fold in folds),
        input_window,
    )
    frame, features = preprocess_stock_data_samples(samples, feature_num, stockid2idx)
    frame = frame.dropna(subset=['label']).sort_values(['日期', '股票代码']).reset_index(drop=True)
    frame['日期'] = pd.to_datetime(frame['日期'], errors='coerce').dt.normalize()
    if frame['日期'].isna().any() or frame.duplicated(['日期', '股票代码']).any():
        raise ValueError('shared XGBoost feature frame has invalid or duplicate sample keys')
    if frame['日期'].max() >= pd.Timestamp(project_config['data_cutoff']).normalize():
        raise ValueError('shared XGBoost feature frame reaches the competition target period')
    return frame, features, feature_num


def _xgb_feature_frame_hash(frame: pd.DataFrame, features: Sequence[str]) -> str:
    columns = ['日期', '股票代码', 'label', *features]
    work = frame.loc[:, columns].copy()
    work['日期'] = pd.to_datetime(work['日期']).dt.strftime('%Y-%m-%d')
    row_hashes = pd.util.hash_pandas_object(work, index=False).to_numpy(np.uint64)
    digest = hashlib.sha256('|'.join(columns).encode('utf-8'))
    digest.update(np.ascontiguousarray(row_hashes).tobytes())
    return digest.hexdigest()


def _xgb_scope_hash(frame: pd.DataFrame) -> str:
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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 1e-12):
        raise ValueError('sequence contains invalid price denominator')
    return numerator / denominator - 1.0


MARKET_COLUMNS = ('开盘', '收盘', '最高', '最低', '成交量', '成交额', '换手率')


def _week_market_matrix(week: object) -> np.ndarray:
    frame = week.to_frame().copy()
    if len(frame) != 5:
        raise ValueError('history contains incomplete week')
    missing = set(MARKET_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f'sequence missing columns: {sorted(missing)}')
    frame['日期'] = pd.to_datetime(frame['日期'], errors='coerce').dt.normalize()
    frame = frame.sort_values('日期')
    expected_dates = pd.date_range(pd.Timestamp(week.start_date), periods=5, freq='D')
    if frame['日期'].isna().any() or tuple(frame['日期']) != tuple(expected_dates):
        raise ValueError('history week is not a complete ordered Monday-Friday week')
    return frame.loc[:, MARKET_COLUMNS].apply(pd.to_numeric, errors='coerce').to_numpy(np.float64)


def _forward_fill_zero(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    indices = np.where(finite, np.arange(len(values)), 0)
    np.maximum.accumulate(indices, out=indices)
    filled = values[indices]
    filled[: int(np.argmax(finite)) if finite.any() else len(values)] = 0.0
    return filled if finite.any() else np.zeros_like(values)


def _sequence_features_from_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape != (HISTORY_DAYS, len(MARKET_COLUMNS)):
        raise ValueError(f'expected {HISTORY_DAYS} daily history rows')
    values = {name: matrix[:, index].copy() for index, name in enumerate(MARKET_COLUMNS)}
    if any(not np.isfinite(values[name]).all() for name in ('开盘', '收盘', '最高', '最低')):
        raise ValueError('sequence contains non-finite price values')
    for name in ('成交量', '成交额', '换手率'):
        values[name] = _forward_fill_zero(values[name])

    previous_close = np.roll(values['收盘'], 1)
    previous_close[0] = values['收盘'][0]
    log_volume = np.log1p(np.maximum(values['成交量'], 0.0))
    log_amount = np.log1p(np.maximum(values['成交额'], 0.0))
    volume_change = np.diff(log_volume, prepend=log_volume[0])
    amount_change = np.diff(log_amount, prepend=log_amount[0])
    result = np.column_stack((
        _safe_ratio(values['开盘'], previous_close),
        _safe_ratio(values['收盘'], values['开盘']),
        _safe_ratio(values['最高'], values['开盘']),
        _safe_ratio(values['最低'], values['开盘']),
        _safe_ratio(values['收盘'], previous_close),
        volume_change,
        amount_change,
        values['换手率'] / 100.0,
    ))
    result[0, (0, 4, 5, 6)] = 0.0
    if result.shape != (HISTORY_DAYS, len(SEQUENCE_FEATURES)) or not np.isfinite(result).all():
        raise ValueError('invalid transformed sequence')
    return np.clip(result, -5.0, 5.0).astype(np.float32)


def daily_sequence_features(history_weeks: Sequence[object]) -> np.ndarray:
    """把 12 个完整历史周转为无拟合状态的 60x8 日序列。"""
    if len(history_weeks) != HISTORY_WEEKS:
        raise ValueError(f'expected {HISTORY_WEEKS} history weeks')
    starts = pd.DatetimeIndex(pd.Timestamp(week.start_date).normalize() for week in history_weeks)
    if not starts.is_monotonic_increasing or starts.has_duplicates:
        raise ValueError('history weeks must be unique and ordered')
    matrix = np.concatenate([_week_market_matrix(week) for week in history_weeks], axis=0)
    return _sequence_features_from_matrix(matrix)


def build_cross_sections(samples: Sequence[object], min_group_size: int) -> tuple[CrossSection, ...]:
    """构造按目标周分组的真实神经输入，并检查历史严格早于标签周。"""
    grouped: dict[pd.Timestamp, list[object]] = {}
    for sample in samples:
        target = pd.Timestamp(sample.future_week.start_date).normalize()
        grouped.setdefault(target, []).append(sample)
    sections = []
    week_cache: dict[tuple[str, pd.Timestamp], np.ndarray] = {}

    def cached_sequence(sample: object) -> np.ndarray:
        matrices = []
        starts = []
        for week in sample.history_weeks:
            start = pd.Timestamp(week.start_date).normalize()
            starts.append(start)
            key = (sample.stock_id, start)
            if key not in week_cache:
                week_cache[key] = _week_market_matrix(week)
            matrices.append(week_cache[key])
        index = pd.DatetimeIndex(starts)
        if len(index) != HISTORY_WEEKS or not index.is_monotonic_increasing or index.has_duplicates:
            raise ValueError('history weeks must be 12 unique ordered weeks')
        return _sequence_features_from_matrix(np.concatenate(matrices, axis=0))

    for target, group in sorted(grouped.items()):
        rows = sorted(group, key=lambda item: item.stock_id)
        if len(rows) < min_group_size:
            continue
        if any(pd.Timestamp(item.history_weeks[-1].end_date).normalize() >= target for item in rows):
            raise ValueError('future leakage: history reaches target week')
        sections.append(CrossSection(
            target_week=target,
            stock_ids=tuple(item.stock_id for item in rows),
            values=np.stack([cached_sequence(item) for item in rows]),
            returns=np.asarray([item.future_return for item in rows], dtype=np.float32),
        ))
    if not sections:
        raise ValueError('no valid real cross-sections')
    return tuple(sections)


def _flatten_predictions(predictions: Sequence[np.ndarray], sections: Sequence[CrossSection]):
    if len(predictions) != len(sections):
        raise ValueError('prediction count does not match cross-sections')
    scores = np.concatenate([np.asarray(value, dtype=np.float64) for value in predictions])
    returns = np.concatenate([section.returns.astype(np.float64) for section in sections])
    groups = [len(section.stock_ids) for section in sections]
    if len(scores) != len(returns):
        raise ValueError('prediction rows do not match labels')
    return scores, returns, groups


def summarize_predictions(predictions: Sequence[np.ndarray], sections: Sequence[CrossSection], top_k: int = 5) -> dict:
    scores, returns, groups = _flatten_predictions(predictions, sections)
    topk = topk_return_metrics(scores, returns, groups, top_k=top_k)
    ptr = np.concatenate(([0], np.cumsum(groups)))
    weekly_rank_ic = []
    for begin, end in zip(ptr[:-1], ptr[1:]):
        weekly_rank_ic.append(rank_ic_score(scores[begin:end], returns[begin:end], np.asarray([0, end - begin])))
    weekly_returns = topk[f'pred_top{top_k}_group_returns']
    return {
        'weekly_returns': [float(value) for value in weekly_returns],
        'validation_weeks': [section.target_week.date().isoformat() for section in sections],
        'mean_return': float(np.mean(weekly_returns)),
        'worst_return': float(np.min(weekly_returns)),
        'return_std': float(np.std(weekly_returns, ddof=0)),
        'positive_rate': float(np.mean(np.asarray(weekly_returns) > 0.0)),
        'rank_ic': float(np.mean(weekly_rank_ic)),
        'weekly_rank_ic': [float(value) for value in weekly_rank_ic],
        'universe_return': float(topk[f'benchmark_top{top_k}_return_avg']),
        'excess_return_diagnostic': float(topk[f'pred_top{top_k}_excess_return_avg']),
    }


def _runtime_start(device) -> float:
    import torch

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def _runtime_stop(device, started: float) -> tuple[float, int]:
    import torch

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
    else:
        peak = 0
    return time.perf_counter() - started, peak


def _build_neural_model(family: str, benchmark: BenchmarkConfig, device):
    return build_ranker(
        family, feature_dim=len(SEQUENCE_FEATURES), hidden_dim=benchmark.hidden_dim,
        num_heads=benchmark.num_heads, dropout=0.0,
    ).to(device)


def _train_neural_epochs(model, sections: Sequence[CrossSection], epochs: int, benchmark: BenchmarkConfig) -> None:
    import torch

    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=benchmark.learning_rate)
    model.train()
    for _ in range(epochs):
        for section in sections:
            values = torch.from_numpy(section.values).unsqueeze(0).to(device)
            target = torch.from_numpy(section.returns).unsqueeze(0).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = pairwise_ranking_loss(model(values), target)
            loss.backward()
            optimizer.step()


def _predict_neural(model, sections: Sequence[CrossSection]) -> list[np.ndarray]:
    import torch

    model.eval()
    predictions = []
    with torch.no_grad():
        device = next(model.parameters()).device
        for section in sections:
            values = torch.from_numpy(section.values).unsqueeze(0).to(device)
            predictions.append(model(values).squeeze(0).cpu().numpy())
    return predictions


def _select_neural_epoch(
    family: str,
    inner_train_sections: Sequence[CrossSection],
    inner_validation_sections: Sequence[CrossSection],
    benchmark: BenchmarkConfig,
    device,
) -> tuple[dict[str, object], float, int]:
    _set_seed(benchmark.seed)
    model = _build_neural_model(family, benchmark, device)
    import torch

    optimizer = torch.optim.AdamW(model.parameters(), lr=benchmark.learning_rate)
    scores = []
    started = _runtime_start(device)
    for _ in range(benchmark.neural_epochs):
        model.train()
        for section in inner_train_sections:
            values = torch.from_numpy(section.values).unsqueeze(0).to(device)
            target = torch.from_numpy(section.returns).unsqueeze(0).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = pairwise_ranking_loss(model(values), target)
            loss.backward()
            optimizer.step()
        predictions = _predict_neural(model, inner_validation_sections)
        scores.append(float(summarize_predictions(
            predictions, inner_validation_sections, benchmark.top_k,
        )['mean_return']))
    elapsed, peak_vram = _runtime_stop(device, started)
    selection = choose_best_iteration(scores, 'inner_validation_top5_return')
    selection['best_epoch'] = int(selection['best_iteration']) + 1
    selection['metric_history'] = scores
    return selection, elapsed, peak_vram


def _neural_fold(
    family: str,
    inner_train_sections: Sequence[CrossSection],
    inner_validation_sections: Sequence[CrossSection],
    outer_train_sections: Sequence[CrossSection],
    outer_validation_sections: Sequence[CrossSection],
    benchmark: BenchmarkConfig,
) -> dict:
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    selection, checkpoint_seconds, checkpoint_peak = _select_neural_epoch(
        family, inner_train_sections, inner_validation_sections, benchmark, device,
    )

    _set_seed(benchmark.seed)
    model = _build_neural_model(family, benchmark, device)
    started = _runtime_start(device)
    _train_neural_epochs(model, outer_train_sections, int(selection['best_epoch']), benchmark)
    refit_seconds, refit_peak = _runtime_stop(device, started)

    started = _runtime_start(device)
    predictions = _predict_neural(model, outer_validation_sections)
    predict_seconds, predict_peak = _runtime_stop(device, started)
    model_bytes = int(sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()))
    return {
        'metrics': summarize_predictions(predictions, outer_validation_sections, benchmark.top_k),
        'checkpoint': selection,
        'runtime': {
            'checkpoint_train_seconds': checkpoint_seconds,
            'outer_train_refit_seconds': refit_seconds,
            'train_seconds': checkpoint_seconds + refit_seconds,
            'predict_seconds': predict_seconds,
            'peak_vram_bytes': max(checkpoint_peak, refit_peak, predict_peak),
            'model_bytes': model_bytes,
            'accelerator': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        },
    }


def _momentum_fold(validation_sections: Sequence[CrossSection], top_k: int) -> dict:
    started = time.perf_counter()
    # close_to_previous_close 的逐日复利是只使用历史 60 日的朴素动量信号。
    predictions = [np.prod(1.0 + section.values[:, :, 4], axis=1) - 1.0 for section in validation_sections]
    return {
        'metrics': summarize_predictions(predictions, validation_sections, top_k),
        'runtime': {
            'train_seconds': 0.0,
            'predict_seconds': time.perf_counter() - started,
            'peak_vram_bytes': 0,
            'model_bytes': 0,
            'accelerator': 'cpu',
        },
    }


def fit_nested_xgb_booster(
    inner_train: xgb.DMatrix,
    inner_validation: xgb.DMatrix,
    outer_train: xgb.DMatrix,
    params: dict,
    max_rounds: int,
) -> tuple[xgb.Booster, dict[str, object], dict[str, float]]:
    """仅用 inner validation 选轮，并从相同 seed 在完整 outer-train 重训。"""
    if max_rounds < 1:
        raise ValueError('max rounds must be positive')
    history: dict[str, dict[str, list[float]]] = {}
    started = time.perf_counter()
    xgb.train(
        dict(params), inner_train, num_boost_round=max_rounds,
        evals=[(inner_validation, 'inner_validation')], custom_metric=xgb_rank_return_metrics,
        maximize=True, evals_result=history, verbose_eval=False,
    )
    checkpoint_seconds = time.perf_counter() - started
    selection = choose_best_iteration(
        [float(value) for value in history['inner_validation']['top5_return']],
        'inner_validation_top5_return',
    )
    selection['selected_rounds'] = int(selection['best_iteration']) + 1

    started = time.perf_counter()
    booster = xgb.train(
        dict(params), outer_train, num_boost_round=int(selection['selected_rounds']),
        evals=(), verbose_eval=False,
    )
    refit_seconds = time.perf_counter() - started
    return booster, selection, {
        'checkpoint_train_seconds': checkpoint_seconds,
        'outer_train_refit_seconds': refit_seconds,
        'train_seconds': checkpoint_seconds + refit_seconds,
    }


def _matrix_from_shared_xgb_frame(
    shared_frame: pd.DataFrame,
    sections: Sequence[CrossSection],
    features: Sequence[str],
) -> tuple[pd.DataFrame, xgb.DMatrix, list[int]]:
    """仅按已验证 section 键切共享特征，不在 split 内重新计算 TA/EMA。"""
    expected_rows = []
    groups = []
    for section in sections:
        groups.append(len(section.stock_ids))
        expected_rows.extend(
            (section.target_week, stock_id, float(label))
            for stock_id, label in zip(section.stock_ids, section.returns)
        )
    expected = pd.DataFrame(expected_rows, columns=['日期', '股票代码', '_expected_label'])
    expected['_row_order'] = np.arange(len(expected), dtype=np.int64)
    if expected.empty or expected.duplicated(['日期', '股票代码']).any():
        raise ValueError('nested XGBoost section keys are empty or duplicated')

    available = shared_frame.loc[:, ['日期', '股票代码', 'label', *features]].copy()
    if available.duplicated(['日期', '股票代码']).any():
        raise ValueError('shared XGBoost feature keys are duplicated')
    sliced = expected.merge(
        available, on=['日期', '股票代码'], how='left', validate='one_to_one', indicator=True,
    ).sort_values('_row_order')
    if not sliced['_merge'].eq('both').all() or len(sliced) != len(expected):
        missing = int((sliced['_merge'] != 'both').sum())
        raise ValueError(f'shared XGBoost feature frame misses {missing} nested rows')
    if not np.allclose(
        sliced['label'].to_numpy(np.float32),
        sliced['_expected_label'].to_numpy(np.float32),
        atol=1e-7,
    ):
        raise ValueError('shared XGBoost labels differ from neural section labels')
    sliced = sliced.drop(columns=['_expected_label', '_row_order', '_merge']).reset_index(drop=True)
    matrix = xgb.DMatrix(
        sliced.loc[:, features].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32),
        label=sliced['label'].to_numpy(np.float32), missing=np.nan,
        group=np.asarray(groups, np.uint32), feature_names=list(features),
    )
    if matrix.num_row() != len(expected) or sum(groups) != len(expected):
        raise ValueError('shared XGBoost matrix changed nested section rows')
    return sliced, matrix, groups


def _xgb_fold(
    shared_feature_frame: pd.DataFrame,
    features: Sequence[str],
    inner_train_sections: Sequence[CrossSection],
    inner_validation_sections: Sequence[CrossSection],
    outer_train_sections: Sequence[CrossSection],
    outer_validation_sections: Sequence[CrossSection],
    project_config: dict,
    benchmark: BenchmarkConfig,
) -> dict:
    model_config = project_config['model_params']['xgb_rank_pairwise']
    prepared = [
        _matrix_from_shared_xgb_frame(shared_feature_frame, sections, features)
        for sections in (
            inner_train_sections, inner_validation_sections, outer_train_sections,
            outer_validation_sections,
        )
    ]
    inner_train_df, d_inner_train, _ = prepared[0]
    inner_validation_df, d_inner_validation, _ = prepared[1]
    outer_train_df, d_outer_train, _ = prepared[2]
    outer_validation_df, d_outer_validation, outer_validation_groups = prepared[3]
    params = {key: value for key, value in model_config.items()
              if key not in {'input_window', 'feature_type', 'num_boost_round'}}
    params['disable_default_eval_metric'] = 1
    params['seed'] = benchmark.seed
    params['nthread'] = benchmark.nthread
    booster, selection, runtime = fit_nested_xgb_booster(
        d_inner_train, d_inner_validation, d_outer_train, params, benchmark.xgb_rounds,
    )
    started = time.perf_counter()
    prediction = booster.predict(d_outer_validation)
    predict_seconds = time.perf_counter() - started
    offsets = np.concatenate(([0], np.cumsum(outer_validation_groups)))
    predictions = [prediction[begin:end] for begin, end in zip(offsets[:-1], offsets[1:])]
    raw_model = booster.save_raw(raw_format='ubj')
    return {
        'metrics': summarize_predictions(predictions, outer_validation_sections, benchmark.top_k),
        'checkpoint': selection,
        'runtime': {
            **runtime,
            'predict_seconds': predict_seconds,
            'peak_vram_bytes': None,
            'model_bytes': len(raw_model),
            'accelerator': str(params.get('device', 'cpu')),
            'peak_vram_note': 'XGBoost Python API does not expose allocator peak memory',
        },
        'feature_count': len(features),
        'sample_rows': {
            'inner_train': len(inner_train_df),
            'inner_validation': len(inner_validation_df),
            'outer_train': len(outer_train_df),
            'outer_validation': len(outer_validation_df),
        },
        'split_sha256': {
            'inner_train': _xgb_scope_hash(inner_train_df),
            'inner_validation': _xgb_scope_hash(inner_validation_df),
            'outer_train': _xgb_scope_hash(outer_train_df),
            'outer_validation': _xgb_scope_hash(outer_validation_df),
        },
    }


def _aggregate_folds(folds: Sequence[dict]) -> dict:
    weekly_returns = [value for fold in folds for value in fold['metrics']['weekly_returns']]
    rank_ics = [value for fold in folds for value in fold['metrics']['weekly_rank_ic']]
    peak_vram = [
        fold['runtime']['peak_vram_bytes'] for fold in folds
        if fold['runtime']['peak_vram_bytes'] is not None
    ]
    return {
        'weekly_returns': weekly_returns,
        'mean_return': float(np.mean(weekly_returns)),
        'worst_return': float(np.min(weekly_returns)),
        'return_std': float(np.std(weekly_returns, ddof=0)),
        'positive_rate': float(np.mean(np.asarray(weekly_returns) > 0.0)),
        'rank_ic': float(np.mean(rank_ics)),
        'train_seconds': float(sum(fold['runtime']['train_seconds'] for fold in folds)),
        'predict_seconds': float(sum(fold['runtime']['predict_seconds'] for fold in folds)),
        'peak_vram_bytes': int(max(peak_vram)) if len(peak_vram) == len(folds) else None,
        'model_bytes': int(max(fold['runtime']['model_bytes'] for fold in folds)),
    }


def _pareto_analysis(models: dict[str, dict[str, object]]) -> dict[str, object]:
    """给出观测维度 Pareto 前沿；资源缺失时一律不晋升。"""
    maximize = ('mean_return', 'worst_return', 'positive_rate', 'rank_ic')
    minimize = ('return_std', 'train_seconds', 'predict_seconds', 'model_bytes')
    model_names = tuple(models)
    baseline_name = 'xgboost_rank_pairwise'
    baseline = models[baseline_name]['aggregate']

    def dominates(left: dict, right: dict) -> bool:
        no_worse = all(left[key] >= right[key] for key in maximize)
        no_worse = no_worse and all(left[key] <= right[key] for key in minimize)
        strictly_better = any(left[key] > right[key] for key in maximize)
        strictly_better = strictly_better or any(left[key] < right[key] for key in minimize)
        return no_worse and strictly_better

    frontier = []
    decisions = {}
    for name in model_names:
        candidate = models[name]['aggregate']
        dominators = [
            other for other in model_names
            if other != name and dominates(models[other]['aggregate'], candidate)
        ]
        nondominated = not dominators
        if nondominated:
            frontier.append(name)
        resource_gap = [
            field for field in ('peak_vram_bytes',)
            if candidate.get(field) is None
        ]
        regressions = [key for key in maximize if candidate[key] < baseline[key]]
        regressions += [key for key in minimize if candidate[key] > baseline[key]]
        strict = [key for key in maximize if candidate[key] > baseline[key]]
        strict += [key for key in minimize if candidate[key] < baseline[key]]
        improves_baseline = name != baseline_name and not regressions and bool(strict)
        decisions[name] = {
            'nondominated_observed_dimensions': nondominated,
            'dominated_by': dominators,
            'resource_complete': not resource_gap,
            'resource_gap': resource_gap,
            'baseline': name == baseline_name,
            'baseline_regressions': regressions,
            'baseline_strict_improvements': strict,
            'complete_pareto_improvement_over_baseline': improves_baseline,
            'eligible_for_promotion': improves_baseline and not resource_gap,
            'promote': False,
            'reason': 'benchmark evidence never mutates production configuration',
        }
    return {
        'maximize': list(maximize),
        'minimize': list(minimize),
        'required_resource': ['peak_vram_bytes'],
        'baseline': baseline_name,
        'frontier_observed_dimensions': frontier,
        'models': decisions,
    }


def run_real_structure_benchmark(benchmark: BenchmarkConfig) -> dict[str, object]:
    """执行真实冻结行情 E0/E2/E3 对比；smoke 只跑最旧 rolling fold。"""
    _set_seed(benchmark.seed)
    root = Path(__file__).resolve().parents[2]
    project_config = copy.deepcopy(base_config)
    project_config['validation'] = {**project_config['validation'], 'mode': 'rolling_kfold'}
    if project_config['sample_calendar_policy'] != 'full_monday_friday':
        raise ValueError('real structure benchmark requires full_monday_friday policy')
    raw_df = load_market_data(project_config)
    plan = build_validation_plan(raw_df, project_config)
    all_folds = plan.validation_splits()
    folds = all_folds[:benchmark.folds]
    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {stock_id: index for index, stock_id in enumerate(stock_ids)}
    shared_xgb_frame, shared_xgb_features, xgb_feature_num = _build_shared_xgb_feature_frame(
        raw_df, folds, stockid2idx, project_config,
    )
    shared_xgb_sha256 = _xgb_feature_frame_hash(shared_xgb_frame, shared_xgb_features)
    manifest = json.loads((root / 'data' / project_config['data_manifest_file']).read_text(encoding='utf-8'))
    model_folds: dict[str, list[dict]] = {
        'xgboost_rank_pairwise': [], 'naive_60d_momentum': [],
        **{report_name: [] for report_name, _ in NEURAL_FAMILIES},
    }
    fold_records = []
    for fold in folds:
        outer_train_samples = plan.get_train_samples(fold, HISTORY_WEEKS)
        outer_validation_samples = plan.get_validation_samples(fold, HISTORY_WEEKS)
        outer_train_sections = build_cross_sections(
            outer_train_samples, int(project_config['min_group_size']),
        )
        outer_validation_sections = build_cross_sections(
            outer_validation_samples, int(project_config['min_group_size']),
        )
        inner_train_sections, inner_validation_sections = split_outer_inner_sections(
            outer_train_sections, benchmark.inner_validation_weeks,
        )
        fold_records.append({
            'name': fold.name,
            'outer_train_start': fold.train_start.date().isoformat(),
            'outer_train_end_exclusive': fold.train_end.date().isoformat(),
            'inner_train_end_inclusive': inner_train_sections[-1].target_week.date().isoformat(),
            'inner_validation_start': inner_validation_sections[0].target_week.date().isoformat(),
            'inner_validation_end_inclusive': inner_validation_sections[-1].target_week.date().isoformat(),
            'outer_validation_start': fold.validation_start.date().isoformat(),
            'outer_validation_end_exclusive': fold.validation_end.date().isoformat(),
            'inner_train_weeks': len(inner_train_sections),
            'inner_validation_weeks': len(inner_validation_sections),
            'outer_train_weeks': len(outer_train_sections),
            'outer_validation_weeks': len(outer_validation_sections),
            'inner_train_rows': int(sum(len(section.stock_ids) for section in inner_train_sections)),
            'inner_validation_rows': int(sum(
                len(section.stock_ids) for section in inner_validation_sections
            )),
            'outer_train_rows': int(sum(len(section.stock_ids) for section in outer_train_sections)),
            'outer_validation_rows': int(sum(
                len(section.stock_ids) for section in outer_validation_sections
            )),
            'split_sha256': {
                'inner_train': _section_scope_hash(inner_train_sections),
                'inner_validation': _section_scope_hash(inner_validation_sections),
                'outer_train': _section_scope_hash(outer_train_sections),
                'outer_validation': _section_scope_hash(outer_validation_sections),
            },
        })
        model_folds['naive_60d_momentum'].append(_momentum_fold(
            outer_validation_sections, benchmark.top_k,
        ))
        model_folds['xgboost_rank_pairwise'].append(_xgb_fold(
            shared_xgb_frame, shared_xgb_features, inner_train_sections,
            inner_validation_sections, outer_train_sections, outer_validation_sections,
            project_config, benchmark,
        ))
        for report_name, family in NEURAL_FAMILIES:
            model_folds[report_name].append(_neural_fold(
                family, inner_train_sections, inner_validation_sections,
                outer_train_sections, outer_validation_sections, benchmark,
            ))

    config_payload = {
        **asdict(benchmark),
        'history_weeks': HISTORY_WEEKS,
        'history_days': HISTORY_DAYS,
        'sequence_features': SEQUENCE_FEATURES,
        'label_mode': project_config['label_mode'],
        'sample_calendar_policy': project_config['sample_calendar_policy'],
        'data_cutoff': project_config['data_cutoff'],
    }
    models = {
        name: {'aggregate': _aggregate_folds(results), 'folds': results}
        for name, results in model_folds.items()
    }
    import torch

    return {
        'schema_version': 2,
        'data_kind': 'real_frozen_market',
        'synthetic': False,
        'effect_evidence': True,
        'mode': benchmark.mode,
        'claim_boundary': (
            'one-fold low-budget nested rolling smoke; directional only'
            if benchmark.mode == 'smoke'
            else 'four-fold nested walk-forward evidence; not an independent final holdout'
        ),
        'official_baseline_contract': {
            'source_id': 'EXT-001',
            'implementation': 'clean_room_from_public_structure_and_scoring_contract',
            'equivalence_boundary': 'not source, preprocessing, checkpoint, or published-score parity',
        },
        'data': {
            'file': 'data/stock_data.csv',
            'manifest': 'data/manifest.json',
            'sha256': manifest['data']['sha256'],
            'rows': manifest['data']['rows'],
            'date_max': manifest['data']['date_max'],
            'data_scope': 'manifest-verified rows strictly before 2026-06-29; no post-cutoff or synthetic rows',
            'non_price_missing_rows': {
                name: int(pd.to_numeric(raw_df[name], errors='coerce').isna().sum())
                for name in ('成交量', '成交额', '换手率')
            },
        },
        'git': _git_state(root),
        'config': config_payload,
        'config_sha256': _canonical_hash(config_payload),
        'feature_contract': FEATURE_CONTRACT,
        'xgboost_shared_feature_frame': {
            'construction': 'one causal preprocessing pass over the complete selected-fold interval',
            'feature_num': xgb_feature_num,
            'feature_count': len(shared_xgb_features),
            'rows': len(shared_xgb_frame),
            'target_start': shared_xgb_frame['日期'].min().date().isoformat(),
            'target_end': shared_xgb_frame['日期'].max().date().isoformat(),
            'sha256': shared_xgb_sha256,
            'split_policy': 'all nested matrices are exact (target_week, stock_id) key slices',
        },
        'nested_selection_contract': {
            'inner_validation': 'last four target weeks of each complete outer-train',
            'checkpoint_metric': 'absolute mean Top5 return on inner validation',
            'neural_refit': 'fresh model with the same seed; selected epochs on complete outer-train',
            'xgboost_refit': 'fresh booster with the same seed; selected rounds on complete outer-train',
            'outer_validation': 'same-fold prediction and metrics only; never checkpoint input',
            'naive_momentum': 'no training or checkpoint selection',
        },
        'runtime_environment': {
            'python': platform.python_version(),
            'platform': platform.system(),
            'logical_cpu_count': os.cpu_count(),
            'xgboost': xgb.__version__,
            'torch': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        },
        'folds': fold_records,
        'models': models,
        'pareto': _pareto_analysis(models),
        'production_config_changed': False,
    }


def write_report(output: Path, report: dict[str, object]) -> None:
    if output.is_absolute() or '..' in output.parts:
        raise ValueError('output must be a repository-relative path')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--output', type=Path,
        required=True,
    )
    args = parser.parse_args()
    report = run_real_structure_benchmark(BenchmarkConfig.for_mode(args.mode, seed=args.seed))
    write_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
