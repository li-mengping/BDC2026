from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioConfig:
    top_k_per_model: int = 10
    final_k: int = 5
    weight_strategy: str = 'equal'
    model_subset: tuple[str, ...] | None = None
    union_top_k_by_model: tuple[tuple[str, int], ...] | None = (
        ('xgb_rank_pairwise', 10),
    )
    recall_model: str | None = None
    recall_top_k: int | None = None
    rerank_model: str | None = None
    rerank_top_k: int | None = None
    covariance_lookback_days: int = 40
    covariance_shrinkage: float = 0.1
    covariance_ridge: float = 1e-6
    risk_aversion: float = 12.0
    selection_max_weight: float = 0.2
    final_min_weight: float = 0.10
    final_max_weight: float = 0.30
    mu_percentile_floor: float = 0.01
    mu_percentile_cap: float = 0.99
    optimizer_iterations: int = 1000
    optimizer_tolerance: float = 1e-10


DEFAULT_PORTFOLIO_CONFIG = PortfolioConfig()


def select_portfolio(
    scored: pd.DataFrame,
    raw_df: pd.DataFrame,
    target_date,
    train_returns,
    model_names: list[str],
    portfolio_config: PortfolioConfig = DEFAULT_PORTFOLIO_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    top10, candidates = build_candidate_pool(scored, model_names, portfolio_config)
    candidates = candidates.copy()
    candidates['mu'] = consensus_mu(
        candidates['consensus_score'].to_numpy(),
        train_returns,
        portfolio_config.mu_percentile_floor,
        portfolio_config.mu_percentile_cap,
    )

    stock_ids = candidates['stock_id'].tolist()
    covariance = estimate_weekly_covariance(
        raw_df,
        stock_ids,
        target_date,
        portfolio_config.covariance_lookback_days,
        portfolio_config.covariance_shrinkage,
        portfolio_config.covariance_ridge,
    )
    selection_max_weight = max(portfolio_config.selection_max_weight, 1.0 / len(candidates))
    weights = optimize_mean_variance(
        candidates['mu'].to_numpy(dtype=np.float64),
        covariance,
        portfolio_config.risk_aversion,
        selection_max_weight,
        portfolio_config.optimizer_iterations,
        portfolio_config.optimizer_tolerance,
    )

    candidates['candidate_index'] = np.arange(len(candidates))
    # 第一阶段权重只用于在候选池中筛 Top5，不是最终提交权重。
    candidates['selection_weight'] = weights
    candidates['selection_rank'] = rank_descending(candidates['selection_weight']).astype(int)
    candidates['portfolio_vol_contrib'] = covariance @ weights
    candidates = candidates.sort_values(
        ['selection_weight', 'consensus_score', 'best_model_rank', 'stock_id'],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    selected = candidates.head(portfolio_config.final_k).copy()
    selected_indices = selected['candidate_index'].to_numpy(dtype=np.int64)
    selected_covariance = covariance[np.ix_(selected_indices, selected_indices)]
    selected['final_weight'] = assign_final_weights(selected, selected_covariance, portfolio_config)
    selected['final_vol_contrib'] = selected_covariance @ selected['final_weight'].to_numpy(dtype=np.float64)
    selected = selected.sort_values(
        ['final_weight', 'selection_weight', 'consensus_score', 'best_model_rank', 'stock_id'],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    selected['final_weight'] = selected['final_weight'].round(10)
    selected['final_rank'] = np.arange(1, len(selected) + 1)
    candidates = candidates.drop(columns=['candidate_index'])
    selected = selected.drop(columns=['candidate_index'])
    return top10, candidates, selected


def build_candidate_pool(
    scored: pd.DataFrame,
    model_names: list[str],
    portfolio_config: PortfolioConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按配置构造候选池。"""
    if portfolio_config.union_top_k_by_model:
        return build_union_pool(scored, model_names, portfolio_config.union_top_k_by_model)

    if portfolio_config.recall_model or portfolio_config.rerank_model:
        return build_recall_rerank_pool(
            scored,
            model_names,
            portfolio_config.recall_model,
            portfolio_config.recall_top_k,
            portfolio_config.rerank_model,
            portfolio_config.rerank_top_k,
        )

    active_model_names = list(portfolio_config.model_subset or model_names)
    missing_models = [name for name in active_model_names if name not in model_names]
    if missing_models:
        raise ValueError(f'model_subset contains unknown models: {missing_models}')
    return build_consensus_pool(scored, active_model_names, portfolio_config.top_k_per_model)


def assign_final_weights(selected: pd.DataFrame, covariance: np.ndarray, portfolio_config: PortfolioConfig) -> np.ndarray:
    """按配置给最终 Top5 分配权重。"""
    strategy = portfolio_config.weight_strategy
    n = len(selected)
    if strategy == 'mean_variance':
        return optimize_mean_variance(
            selected['mu'].to_numpy(dtype=np.float64),
            covariance,
            portfolio_config.risk_aversion,
            portfolio_config.final_max_weight,
            portfolio_config.optimizer_iterations,
            portfolio_config.optimizer_tolerance,
            min_weight=portfolio_config.final_min_weight,
        )
    if strategy == 'equal':
        return project_bounded_simplex(
            np.full(n, 1.0 / n, dtype=np.float64),
            portfolio_config.final_min_weight,
            portfolio_config.final_max_weight,
        )
    if strategy == 'rank_linear':
        scores = np.arange(n, 0, -1, dtype=np.float64)
        return project_bounded_simplex(
            scores / scores.sum(),
            portfolio_config.final_min_weight,
            portfolio_config.final_max_weight,
        )
    if strategy == 'consensus_softmax':
        scores = selected['consensus_score'].to_numpy(dtype=np.float64)
        scores = scores - scores.max()
        weights = np.exp(scores * 3.0)
        return project_bounded_simplex(
            weights / weights.sum(),
            portfolio_config.final_min_weight,
            portfolio_config.final_max_weight,
        )
    raise ValueError(f'unsupported weight strategy: {strategy}')


def build_consensus_pool(
    scored: pd.DataFrame,
    model_names: list[str],
    top_k_per_model: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stock_col = '股票代码'
    score_cols = [f'{name}_score' for name in model_names]
    missing = [col for col in score_cols if col not in scored.columns]
    if missing:
        raise ValueError(f'missing model score columns: {missing}')
    if stock_col not in scored.columns:
        raise ValueError('missing 股票代码 column')

    work = scored.copy()
    work[stock_col] = work[stock_col].astype(str).str.zfill(6)
    work = work.sort_values(stock_col).reset_index(drop=True)
    if work[score_cols].isna().any().any():
        raise ValueError('model scores contain NaN')

    top_frames = []
    rank_norm_cols = []
    zscore_cols = []
    rank_cols = []
    for name in model_names:
        score_col = f'{name}_score'
        rank_col = f'{name}_rank'
        rank_norm_col = f'{name}_rank_norm'
        zscore_col = f'{name}_zscore'
        work[rank_col] = rank_descending(work[score_col]).astype(int)
        work[rank_norm_col] = rank_percentile(work[score_col])
        work[zscore_col] = zscore(work[score_col])
        rank_cols.append(rank_col)
        rank_norm_cols.append(rank_norm_col)
        zscore_cols.append(zscore_col)

        top = work.sort_values([rank_col, stock_col], ascending=[True, True]).head(top_k_per_model).copy()
        top['model'] = name
        top['model_rank'] = top[rank_col]
        top['score'] = top[score_col]
        top['model_rank_norm'] = top[rank_norm_col]
        top['model_zscore'] = top[zscore_col]
        top_frames.append(top[['model', 'model_rank', stock_col, 'score', 'model_rank_norm', 'model_zscore']])

    work['consensus_score'] = work[rank_norm_cols].mean(axis=1)
    work['consensus_zscore'] = work[zscore_cols].mean(axis=1)
    work['consensus_rank'] = rank_descending(work['consensus_score']).astype(int)
    work['consensus_percentile'] = rank_percentile(work['consensus_score'])

    top10 = pd.concat(top_frames, ignore_index=True).rename(columns={stock_col: 'stock_id'})
    top10 = top10.sort_values(['model', 'model_rank', 'stock_id']).reset_index(drop=True)
    top10_stock_ids = top10['stock_id'].drop_duplicates().tolist()

    candidates = work[work[stock_col].isin(top10_stock_ids)].copy().rename(columns={stock_col: 'stock_id'})
    source_models = top10.groupby('stock_id')['model'].agg(lambda values: ','.join(values))
    best_model_rank = top10.groupby('stock_id')['model_rank'].min()
    candidates['source_models'] = candidates['stock_id'].map(source_models)
    candidates['source_count'] = candidates['source_models'].str.count(',') + 1
    candidates['best_model_rank'] = candidates['stock_id'].map(best_model_rank).astype(int)

    ordered_cols = [
        'stock_id',
        'source_models',
        'source_count',
        'best_model_rank',
        'consensus_score',
        'consensus_zscore',
        'consensus_rank',
        'consensus_percentile',
        *score_cols,
        *rank_cols,
        *rank_norm_cols,
        *zscore_cols,
    ]
    candidates = candidates[ordered_cols].sort_values(
        ['consensus_score', 'source_count', 'best_model_rank', 'stock_id'],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    return top10, candidates


def build_union_pool(
    scored: pd.DataFrame,
    model_names: list[str],
    union_top_k_by_model: tuple[tuple[str, int], ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stock_col = '股票代码'
    if not union_top_k_by_model:
        raise ValueError('union_top_k_by_model cannot be empty')

    active_model_names = list(dict.fromkeys(name for name, _ in union_top_k_by_model))
    invalid_top_k = [(name, top_k) for name, top_k in union_top_k_by_model if top_k < 1]
    if invalid_top_k:
        raise ValueError(f'union topK values must be positive: {invalid_top_k}')

    missing_models = [name for name in active_model_names if name not in model_names]
    if missing_models:
        raise ValueError(f'union_top_k_by_model contains unknown models: {missing_models}')
    score_cols = [f'{name}_score' for name in active_model_names]
    missing_cols = [col for col in score_cols if col not in scored.columns]
    if missing_cols:
        raise ValueError(f'missing model score columns: {missing_cols}')
    if stock_col not in scored.columns:
        raise ValueError('missing 股票代码 column')

    work = scored.copy()
    work[stock_col] = work[stock_col].astype(str).str.zfill(6)
    work = work.sort_values(stock_col).reset_index(drop=True)
    if work[score_cols].isna().any().any():
        raise ValueError('model scores contain NaN')

    top_frames = []
    rank_cols = []
    rank_norm_cols = []
    zscore_cols = []
    for name in active_model_names:
        score_col = f'{name}_score'
        rank_col = f'{name}_rank'
        rank_norm_col = f'{name}_rank_norm'
        zscore_col = f'{name}_zscore'
        work[rank_col] = rank_descending(work[score_col]).astype(int)
        work[rank_norm_col] = rank_percentile(work[score_col])
        work[zscore_col] = zscore(work[score_col])
        rank_cols.append(rank_col)
        rank_norm_cols.append(rank_norm_col)
        zscore_cols.append(zscore_col)

    for name, top_k in union_top_k_by_model:
        rank_col = f'{name}_rank'
        rank_norm_col = f'{name}_rank_norm'
        zscore_col = f'{name}_zscore'
        score_col = f'{name}_score'
        top = work.sort_values([rank_col, stock_col], ascending=[True, True]).head(top_k).copy()
        top['model'] = name
        top['model_rank'] = top[rank_col]
        top['score'] = top[score_col]
        top['model_rank_norm'] = top[rank_norm_col]
        top['model_zscore'] = top[zscore_col]
        top_frames.append(top[['model', 'model_rank', stock_col, 'score', 'model_rank_norm', 'model_zscore']])

    top10 = pd.concat(top_frames, ignore_index=True).rename(columns={stock_col: 'stock_id'})
    top10 = top10.sort_values(['model', 'model_rank', 'stock_id']).reset_index(drop=True)
    top_stock_ids = top10['stock_id'].drop_duplicates().tolist()

    work['consensus_score'] = work[rank_norm_cols].mean(axis=1)
    work['consensus_zscore'] = work[zscore_cols].mean(axis=1)
    work['consensus_rank'] = rank_descending(work['consensus_score']).astype(int)
    work['consensus_percentile'] = rank_percentile(work['consensus_score'])

    candidates = work[work[stock_col].isin(top_stock_ids)].copy().rename(columns={stock_col: 'stock_id'})
    source_models = top10.groupby('stock_id')['model'].agg(lambda values: ','.join(values))
    best_model_rank = top10.groupby('stock_id')['model_rank'].min()
    candidates['source_models'] = candidates['stock_id'].map(source_models)
    candidates['source_count'] = candidates['source_models'].str.count(',') + 1
    candidates['best_model_rank'] = candidates['stock_id'].map(best_model_rank).astype(int)

    ordered_cols = [
        'stock_id',
        'source_models',
        'source_count',
        'best_model_rank',
        'consensus_score',
        'consensus_zscore',
        'consensus_rank',
        'consensus_percentile',
        *score_cols,
        *rank_cols,
        *rank_norm_cols,
        *zscore_cols,
    ]
    candidates = candidates[ordered_cols].sort_values(
        ['source_count', 'consensus_score', 'best_model_rank', 'stock_id'],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    return top10, candidates


def build_recall_rerank_pool(
    scored: pd.DataFrame,
    model_names: list[str],
    recall_model: str | None,
    recall_top_k: int | None,
    rerank_model: str | None,
    rerank_top_k: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stock_col = '股票代码'
    if not recall_model or not rerank_model:
        raise ValueError('recall_model and rerank_model must both be set')
    if not recall_top_k or not rerank_top_k:
        raise ValueError('recall_top_k and rerank_top_k must both be set')
    if recall_top_k < 1 or rerank_top_k < 1:
        raise ValueError('stage topK values must be positive')
    if rerank_top_k > recall_top_k:
        raise ValueError('rerank_top_k cannot exceed recall_top_k')

    used_models = list(dict.fromkeys([recall_model, rerank_model]))
    missing_models = [name for name in used_models if name not in model_names]
    if missing_models:
        raise ValueError(f'stage models contain unknown models: {missing_models}')
    score_cols = [f'{name}_score' for name in used_models]
    missing_cols = [col for col in score_cols if col not in scored.columns]
    if missing_cols:
        raise ValueError(f'missing model score columns: {missing_cols}')
    if stock_col not in scored.columns:
        raise ValueError('missing 股票代码 column')

    work = scored.copy()
    work[stock_col] = work[stock_col].astype(str).str.zfill(6)
    work = work.sort_values(stock_col).reset_index(drop=True)
    if work[score_cols].isna().any().any():
        raise ValueError('model scores contain NaN')

    rank_cols = []
    rank_norm_cols = []
    zscore_cols = []
    for name in used_models:
        score_col = f'{name}_score'
        rank_col = f'{name}_rank'
        rank_norm_col = f'{name}_rank_norm'
        zscore_col = f'{name}_zscore'
        work[rank_col] = rank_descending(work[score_col]).astype(int)
        work[rank_norm_col] = rank_percentile(work[score_col])
        work[zscore_col] = zscore(work[score_col])
        rank_cols.append(rank_col)
        rank_norm_cols.append(rank_norm_col)
        zscore_cols.append(zscore_col)

    recall_rank_col = f'{recall_model}_rank'
    recall_score_col = f'{recall_model}_score'
    recall_rank_norm_col = f'{recall_model}_rank_norm'
    rerank_score_col = f'{rerank_model}_score'
    rerank_rank_col = f'{rerank_model}_rank'
    rerank_rank_norm_col = f'{rerank_model}_rank_norm'
    rerank_zscore_col = f'{rerank_model}_zscore'

    recall_pool = work.sort_values([recall_rank_col, stock_col], ascending=[True, True]).head(recall_top_k).copy()
    reranked = (
        recall_pool
        .sort_values([rerank_score_col, stock_col], ascending=[False, True])
        .head(rerank_top_k)
        .copy()
    )
    reranked['stage_rank'] = np.arange(1, len(reranked) + 1)

    recall_report = recall_pool.copy()
    recall_report['model'] = f'{recall_model}_recall'
    recall_report['model_rank'] = recall_report[recall_rank_col]
    recall_report['score'] = recall_report[recall_score_col]
    recall_report['model_rank_norm'] = recall_report[recall_rank_norm_col]
    recall_report['model_zscore'] = recall_report[f'{recall_model}_zscore']

    rerank_report = reranked.copy()
    rerank_report['model'] = f'{rerank_model}_rerank'
    rerank_report['model_rank'] = rerank_report['stage_rank']
    rerank_report['score'] = rerank_report[rerank_score_col]
    rerank_report['model_rank_norm'] = rerank_report[rerank_rank_norm_col]
    rerank_report['model_zscore'] = rerank_report[rerank_zscore_col]

    top10 = pd.concat(
        [
            recall_report[['model', 'model_rank', stock_col, 'score', 'model_rank_norm', 'model_zscore']],
            rerank_report[['model', 'model_rank', stock_col, 'score', 'model_rank_norm', 'model_zscore']],
        ],
        ignore_index=True,
    ).rename(columns={stock_col: 'stock_id'})
    top10 = top10.sort_values(['model', 'model_rank', 'stock_id']).reset_index(drop=True)

    candidates = reranked.rename(columns={stock_col: 'stock_id'}).copy()
    candidates['source_models'] = f'{recall_model}_recall,{rerank_model}_rerank'
    candidates['source_count'] = 2
    candidates['best_model_rank'] = candidates['stage_rank'].astype(int)
    candidates['consensus_score'] = candidates[rerank_rank_norm_col]
    candidates['consensus_zscore'] = candidates[rerank_zscore_col]
    candidates['consensus_rank'] = candidates['stage_rank'].astype(int)
    candidates['consensus_percentile'] = candidates[rerank_rank_norm_col]

    ordered_cols = [
        'stock_id',
        'source_models',
        'source_count',
        'best_model_rank',
        'consensus_score',
        'consensus_zscore',
        'consensus_rank',
        'consensus_percentile',
        *score_cols,
        *rank_cols,
        *rank_norm_cols,
        *zscore_cols,
    ]
    candidates = candidates[ordered_cols].sort_values(
        ['consensus_score', 'best_model_rank', 'stock_id'],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return top10, candidates


def consensus_mu(consensus_percentile: np.ndarray, train_returns, percentile_floor: float, percentile_cap: float) -> np.ndarray:
    returns = np.asarray(train_returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 10:
        raise ValueError('not enough training returns to map consensus score to mu')
    percentiles = np.clip(np.asarray(consensus_percentile, dtype=np.float64), percentile_floor, percentile_cap)
    return np.quantile(returns, percentiles)


def estimate_weekly_covariance(
    raw_df: pd.DataFrame,
    stock_ids: list[str],
    target_date,
    lookback_days: int,
    shrinkage: float,
    ridge: float,
) -> np.ndarray:
    target_date = pd.Timestamp(target_date).normalize()
    stock_ids = [str(stock_id).zfill(6) for stock_id in stock_ids]
    history = raw_df[(raw_df['股票代码'].isin(stock_ids)) & (raw_df['日期'] < target_date)].copy()
    history = history.sort_values(['股票代码', '日期'])
    history['daily_return'] = history.groupby('股票代码')['收盘'].pct_change(fill_method=None)
    returns = history.pivot(index='日期', columns='股票代码', values='daily_return')
    returns = returns.reindex(columns=stock_ids).tail(lookback_days).dropna(axis=0, how='any')
    if len(returns) < 5:
        raise ValueError(f'not enough covariance history before {target_date.date()}')

    covariance = returns.cov().to_numpy(dtype=np.float64) * 5.0
    diagonal = np.diag(np.diag(covariance))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    covariance = covariance + np.eye(len(stock_ids), dtype=np.float64) * ridge
    return covariance


def optimize_mean_variance(
    mu: np.ndarray,
    covariance: np.ndarray,
    risk_aversion: float,
    max_weight: float,
    iterations: int,
    tolerance: float,
    min_weight: float = 0.0,
) -> np.ndarray:
    mu = np.asarray(mu, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    n = len(mu)
    if covariance.shape != (n, n):
        raise ValueError('covariance shape does not match mu')
    if min_weight < 0.0:
        raise ValueError('min_weight must be non-negative')
    if max_weight < min_weight:
        raise ValueError('max_weight must be at least min_weight')
    if n * min_weight > 1.0 + 1e-12:
        raise ValueError('min_weight is too large for the candidate count')
    if n * max_weight < 1.0 - 1e-12:
        raise ValueError('max_weight is too small for the candidate count')

    covariance = 0.5 * (covariance + covariance.T)
    eig_max = float(np.linalg.eigvalsh(covariance).max())
    step = 1.0 / max(risk_aversion * eig_max, 1e-3)
    weights = np.full(n, 1.0 / n, dtype=np.float64)
    for _ in range(iterations):
        gradient = risk_aversion * covariance @ weights - mu
        updated = project_bounded_simplex(weights - step * gradient, min_weight, max_weight)
        if np.linalg.norm(updated - weights, ord=1) < tolerance:
            weights = updated
            break
        weights = updated
    return weights


def project_bounded_simplex(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        raise ValueError('cannot project empty values')
    if lower < 0.0:
        raise ValueError('lower bound must be non-negative')
    if upper < lower:
        raise ValueError('upper bound must be at least lower bound')
    n = len(values)
    if n * lower > 1.0 + 1e-12:
        raise ValueError('lower bound is too large for simplex projection')
    if n * upper < 1.0 - 1e-12:
        raise ValueError('upper bound is too small for simplex projection')

    low = values.min() - upper
    high = values.max() - lower
    for _ in range(100):
        tau = 0.5 * (low + high)
        projected = np.clip(values - tau, lower, upper)
        if projected.sum() > 1.0:
            low = tau
        else:
            high = tau
    projected = np.clip(values - high, lower, upper)
    residual = 1.0 - projected.sum()
    if abs(residual) > 1e-12:
        room = upper - projected if residual > 0.0 else projected - lower
        adjustable = room > 1e-12
        if adjustable.any():
            adjustment = min(abs(residual), float(room[adjustable].sum()))
            projected[adjustable] += np.sign(residual) * adjustment * room[adjustable] / room[adjustable].sum()
    projected_sum = float(projected.sum())
    if projected_sum <= 0.0:
        raise ValueError('simplex projection produced zero weights')
    if abs(projected_sum - 1.0) > 1e-8:
        raise ValueError('simplex projection failed to sum to one')
    return projected


def rank_descending(values) -> np.ndarray:
    series = pd.Series(values)
    return series.rank(method='first', ascending=False).to_numpy(dtype=np.float64)


def rank_percentile(values) -> np.ndarray:
    ranks = rank_descending(values)
    if len(ranks) <= 1:
        return np.ones(len(ranks), dtype=np.float64)
    return 1.0 - (ranks - 1.0) / (len(ranks) - 1.0)


def zscore(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    std = values.std(ddof=0)
    if std <= 1e-12:
        return np.zeros(len(values), dtype=np.float64)
    return (values - values.mean()) / std
