"""所有神经结构共享的 E4 排序目标。"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _validate(scores: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if scores.shape != targets.shape or scores.ndim != 2:
        raise ValueError('scores and targets must be matching [batch, stocks] tensors')
    return scores, targets


def pairwise_ranking_loss(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    scores, targets = _validate(scores, targets)
    score_difference = scores.unsqueeze(2) - scores.unsqueeze(1)
    target_difference = targets.unsqueeze(2) - targets.unsqueeze(1)
    ordered_pairs = target_difference > 0
    if not ordered_pairs.any():
        return scores.sum() * 0.0
    return F.softplus(-score_difference[ordered_pairs]).mean()


def listmle_loss(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    scores, targets = _validate(scores, targets)
    order = torch.argsort(targets, dim=1, descending=True, stable=True)
    ordered_scores = torch.gather(scores, 1, order)
    denominators = torch.logcumsumexp(ordered_scores.flip(1), dim=1).flip(1)
    return (denominators - ordered_scores).mean()


def topk_weighted_loss(scores: torch.Tensor, targets: torch.Tensor, top_k: int = 5) -> torch.Tensor:
    scores, targets = _validate(scores, targets)
    count = min(int(top_k), scores.shape[1])
    if count < 1:
        raise ValueError('top_k must be positive')
    top_targets, top_indices = torch.topk(targets, count, dim=1, sorted=True)
    weights = torch.softmax(top_targets, dim=1)
    log_probabilities = torch.log_softmax(scores, dim=1)
    return -(weights * torch.gather(log_probabilities, 1, top_indices)).sum(dim=1).mean()


OBJECTIVES = {
    'pairwise': pairwise_ranking_loss,
    'listmle': listmle_loss,
    'top5_weighted': topk_weighted_loss,
}


def resolve_objective(name: str):
    if name not in OBJECTIVES:
        raise ValueError(f'unsupported neural objective: {name}')
    return OBJECTIVES[name]
