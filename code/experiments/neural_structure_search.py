"""仅用于结构健全性检查的合成数据神经搜索工具。

本模块的 synthetic smoke 不能作为真实行情效果或模型提升证据。真实冻结行情比较
由 ``code.experiments.real_structure_benchmark`` 执行。
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from code.experiments.model_neutral import absolute_topk_return, build_experiment_evidence
from code.experiments.neural_objectives import resolve_objective
from code.models.neural import build_ranker


STRUCTURE_FAMILIES = ('e0_transformer', 'e2_tcn', 'e3_cross_stock')


@dataclass(frozen=True)
class SearchConfig:
    mode: str
    epochs: int
    learning_rate: float
    hidden_dim: int
    num_heads: int
    trials: int

    @classmethod
    def for_mode(cls, mode: str, trials: int | None = None) -> 'SearchConfig':
        defaults = {
            'smoke': cls('smoke', epochs=2, learning_rate=0.01, hidden_dim=8, num_heads=2, trials=1),
            'full': cls('full', epochs=80, learning_rate=0.001, hidden_dim=32, num_heads=4, trials=8),
        }
        if mode not in defaults:
            raise ValueError("mode must be 'smoke' or 'full'")
        selected = defaults[mode]
        requested = selected.trials if trials is None else int(trials)
        if not 1 <= requested <= 16:
            raise ValueError('E6 permits between 1 and 16 Optuna trials')
        return cls(**{**asdict(selected), 'trials': requested})


def ensemble_gate(
    baseline_oof_residuals: Sequence[float],
    candidate_oof_residuals: Sequence[float],
    threshold: float = 0.8,
) -> dict[str, object]:
    baseline = np.asarray(baseline_oof_residuals, dtype=np.float64)
    candidate = np.asarray(candidate_oof_residuals, dtype=np.float64)
    if baseline.shape != candidate.shape or baseline.ndim != 1 or len(baseline) < 3:
        raise ValueError('OOF residual vectors must have the same one-dimensional shape')
    correlation = float(np.corrcoef(baseline, candidate)[0, 1])
    if not np.isfinite(correlation):
        correlation = 1.0
    accepted = abs(correlation) < threshold
    return {
        'accepted': accepted,
        'correlation': correlation,
        'threshold': float(threshold),
        'status': 'eligible_for_rank_ensemble' if accepted else 'stopped_correlated_oof_residuals',
    }


def run_bounded_optuna(
    evaluate_fold: Callable[[object, object], float],
    folds: Sequence[object],
    baseline_params: Mapping[str, object],
    config: SearchConfig,
    seed: int,
):
    """E6 接口：基线先入队，至少完成两折后才允许剪枝。"""
    import optuna

    if len(folds) < 2:
        raise ValueError('Optuna search requires at least two rolling folds')

    def objective(trial: optuna.Trial) -> float:
        fold_scores = []
        for fold_index, fold in enumerate(folds):
            fold_scores.append(float(evaluate_fold(trial, fold)))
            trial.report(float(np.mean(fold_scores)), step=fold_index)
            if fold_index >= 1 and trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr('fold_scores', fold_scores)
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=1),
    )
    study.enqueue_trial(dict(baseline_params))
    study.optimize(objective, n_trials=config.trials, show_progress_bar=False)
    return study


def _synthetic_panel(seed: int):
    import torch

    generator = torch.Generator().manual_seed(seed)
    values = torch.randn((8, 6, 8, 4), generator=generator)
    returns = 0.5 * values[:, :, -1, 0] - 0.2 * values[:, :, -2, 1]
    returns += 0.01 * torch.randn(returns.shape, generator=generator)
    return values, returns


def run_synthetic_smoke(seed: int = 17, config: SearchConfig | None = None) -> dict[str, object]:
    """在确定性合成截面上做前后向检查，不产生真实效果结论。"""
    import torch

    selected = config or SearchConfig.for_mode('smoke')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    values, returns = _synthetic_panel(seed)
    values = values.to(device)
    returns = returns.to(device)
    reports = {}
    for family in STRUCTURE_FAMILIES:
        model = build_ranker(
            family,
            feature_dim=values.shape[-1],
            hidden_dim=selected.hidden_dim,
            num_heads=selected.num_heads,
            dropout=0.0,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=selected.learning_rate)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        objective = resolve_objective('top5_weighted')
        started = time.perf_counter()
        for _ in range(selected.epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(values), returns)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(values).cpu().numpy().reshape(-1)
        raw_returns = returns.detach().cpu().numpy().reshape(-1)
        score = absolute_topk_return(prediction, raw_returns, [6] * len(values), top_k=5)
        evidence = build_experiment_evidence(
            Path(f'evidence/{family}.json'),
            {'family': family, 'seed': seed, **asdict(selected)},
            model,
        )
        reports[family] = {
            'absolute_top5_return': score,
            'epochs': selected.epochs,
            'train_seconds': time.perf_counter() - started,
            'runtime': asdict(evidence.runtime),
            'config_sha256': evidence.config_sha256,
            'git_commit': evidence.git_commit,
        }
    return {
        'data_kind': 'synthetic',
        'effect_evidence': False,
        'claim_boundary': 'structure_smoke_only; must_not_be_used_as_market_performance_evidence',
        'mode': selected.mode,
        'config': asdict(selected),
        'primary_metric': 'absolute_top5_return',
        'models': reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--trials', type=int)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--output', type=Path, default=Path('evidence/synthetic-neural-structure-smoke.json'))
    args = parser.parse_args()
    if args.output.is_absolute() or '..' in args.output.parts:
        parser.error('--output must be repository-relative')
    config = SearchConfig.for_mode(args.mode, trials=args.trials)
    result = run_synthetic_smoke(seed=args.seed, config=config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
