"""模型无关的实验指标与可移植证据记录。"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RuntimeEvidence:
    python: str
    platform: str
    torch: str
    accelerator: str
    peak_vram_bytes: int
    model_bytes: int


@dataclass(frozen=True)
class ExperimentEvidence:
    output_path: str
    config_sha256: str
    git_commit: str
    runtime: RuntimeEvidence

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _portable_relative_path(path: Path) -> str:
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('experiment output must be a repository-relative path')
    return path.as_posix()


def _git_commit() -> str:
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        check=False,
        capture_output=True,
        text=True,
        encoding='utf-8',
    )
    return result.stdout.strip() if result.returncode == 0 else 'unavailable'


def _model_size_bytes(model: object | None) -> int:
    if model is None or not hasattr(model, 'parameters'):
        return 0
    return int(sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()))


def _runtime_evidence(model: object | None) -> RuntimeEvidence:
    try:
        import torch

        parameter = next(model.parameters(), None) if model is not None and hasattr(model, 'parameters') else None
        device = parameter.device if parameter is not None else torch.device('cpu')
        accelerator = torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device)
        peak_vram = int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else 0
        torch_version = torch.__version__
    except ImportError:
        accelerator = 'unavailable'
        peak_vram = 0
        torch_version = 'unavailable'
    return RuntimeEvidence(
        python=platform.python_version(),
        platform=f'{sys.platform}-{platform.machine()}',
        torch=torch_version,
        accelerator=accelerator,
        peak_vram_bytes=peak_vram,
        model_bytes=_model_size_bytes(model),
    )


def build_experiment_evidence(
    output_path: Path,
    config: Mapping[str, object],
    model: object | None = None,
) -> ExperimentEvidence:
    canonical = json.dumps(config, ensure_ascii=True, sort_keys=True, separators=(',', ':'))
    return ExperimentEvidence(
        output_path=_portable_relative_path(output_path),
        config_sha256=hashlib.sha256(canonical.encode('utf-8')).hexdigest(),
        git_commit=_git_commit(),
        runtime=_runtime_evidence(model),
    )


def write_experiment_evidence(
    output_path: Path,
    config: Mapping[str, object],
    result: Mapping[str, object],
    model: object | None = None,
) -> ExperimentEvidence:
    evidence = build_experiment_evidence(output_path, config, model)
    destination = Path(evidence.output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({'evidence': evidence.as_dict(), 'config': config, 'result': result}, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    return evidence


def absolute_topk_return(
    predictions: np.ndarray,
    raw_returns: np.ndarray,
    groups: Sequence[int],
    top_k: int = 5,
) -> float:
    """计算各横截面预测 TopK 的等权原始收益均值。"""
    scores = np.asarray(predictions, dtype=np.float64)
    returns = np.asarray(raw_returns, dtype=np.float64)
    sizes = tuple(int(size) for size in groups)
    if scores.ndim != 1 or returns.shape != scores.shape or sum(sizes) != len(scores):
        raise ValueError('predictions, returns, and groups must describe the same rows')
    if top_k < 1 or any(size < top_k for size in sizes):
        raise ValueError('each group must contain at least top_k rows')
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    values = []
    for begin, end in zip(offsets[:-1], offsets[1:]):
        local = np.argsort(-scores[begin:end], kind='stable')[:top_k]
        values.append(float(returns[begin:end][local].mean()))
    return float(np.mean(values))


def pareto_improves(baseline: Mapping[str, float], candidate: Mapping[str, float]) -> bool:
    """对收益、稳定性、资源和产物维度执行保守 Pareto 门禁。"""
    maximize = ('mean_return', 'worst_return', 'positive_rate', 'rank_ic')
    minimize = ('return_std', 'train_seconds', 'predict_seconds', 'peak_vram_bytes', 'model_bytes')
    no_worse = all(candidate[key] >= baseline[key] for key in maximize)
    no_worse &= all(candidate[key] <= baseline[key] for key in minimize)
    strictly_better = any(candidate[key] > baseline[key] for key in maximize)
    strictly_better |= any(candidate[key] < baseline[key] for key in minimize)
    return bool(no_worse and strictly_better)
