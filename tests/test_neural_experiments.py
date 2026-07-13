from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from code.experiments.model_neutral import (
    ExperimentEvidence,
    absolute_topk_return,
    build_experiment_evidence,
)
from code.experiments.neural_objectives import (
    listmle_loss,
    pairwise_ranking_loss,
    topk_weighted_loss,
)
from code.experiments.neural_structure_search import (
    SearchConfig,
    ensemble_gate,
    run_bounded_optuna,
    run_synthetic_smoke,
)
from code.models.neural import build_ranker


def _inputs() -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randn(2, 6, 8, 4)


@pytest.mark.parametrize('family', ['e0_transformer', 'e2_tcn', 'e3_cross_stock'])
def test_neural_ranker_returns_one_score_per_stock(family: str) -> None:
    model = build_ranker(family, feature_dim=4, hidden_dim=8, num_heads=2, dropout=0.0)
    scores = model(_inputs())
    assert scores.shape == (2, 6)
    assert torch.isfinite(scores).all()


def test_cross_stock_attention_is_permutation_equivariant() -> None:
    model = build_ranker('e3_cross_stock', feature_dim=4, hidden_dim=8, num_heads=2, dropout=0.0)
    model.eval()
    values = _inputs()
    permutation = torch.tensor([3, 0, 5, 1, 4, 2])
    with torch.no_grad():
        expected = model(values)[:, permutation]
        actual = model(values[:, permutation])
    assert torch.allclose(actual, expected, atol=1e-6)


def test_tcn_sequence_is_causal() -> None:
    model = build_ranker('e2_tcn', feature_dim=4, hidden_dim=8, num_heads=2, dropout=0.0)
    model.eval()
    original = _inputs()
    changed = original.clone()
    changed[:, :, -1] += 100.0
    with torch.no_grad():
        before = model.encode_sequence(original)
        after = model.encode_sequence(changed)
    assert torch.allclose(before[:, :, :-1], after[:, :, :-1], atol=1e-6)


def test_absolute_top5_return_uses_raw_returns_not_excess() -> None:
    predictions = np.asarray([9, 8, 7, 6, 5, 4], dtype=np.float64)
    returns = np.asarray([0.10, 0.08, 0.06, 0.04, 0.02, 0.90], dtype=np.float64)
    assert absolute_topk_return(predictions, returns, [6], top_k=5) == pytest.approx(0.06)


def test_neural_objectives_are_finite_and_reward_correct_order() -> None:
    target = torch.tensor([[0.4, 0.2, -0.1, -0.2, -0.3, -0.4]])
    good = target * 3
    bad = -good
    for objective in (pairwise_ranking_loss, listmle_loss, topk_weighted_loss):
        good_loss = objective(good, target)
        bad_loss = objective(bad, target)
        assert torch.isfinite(good_loss)
        assert good_loss < bad_loss


def test_ensemble_gate_requires_low_oof_residual_correlation() -> None:
    baseline = np.arange(20, dtype=np.float64)
    assert ensemble_gate(baseline, baseline * 2, threshold=0.8)['accepted'] is False
    alternating = np.tile([1.0, -1.0], 10)
    assert ensemble_gate(baseline, alternating, threshold=0.8)['accepted'] is True


def test_search_modes_and_trial_budget_are_bounded() -> None:
    assert SearchConfig.for_mode('smoke').epochs < SearchConfig.for_mode('full').epochs
    assert SearchConfig.for_mode('full', trials=16).trials == 16
    with pytest.raises(ValueError, match='16'):
        SearchConfig.for_mode('full', trials=17)


def test_optuna_interface_enqueues_baseline_and_records_folds() -> None:
    pytest.importorskip('optuna')

    def evaluate_fold(trial, fold: float) -> float:
        width = trial.suggest_int('width', 2, 8)
        return float(fold) - abs(width - 4)

    study = run_bounded_optuna(
        evaluate_fold,
        folds=(1.0, 2.0),
        baseline_params={'width': 4},
        config=SearchConfig.for_mode('smoke'),
        seed=3,
    )
    assert study.trials[0].params['width'] == 4
    assert study.trials[0].user_attrs['fold_scores'] == [1.0, 2.0]


def test_evidence_rejects_absolute_output_and_records_provenance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='relative'):
        build_experiment_evidence(tmp_path / 'evidence.json', {'seed': 7})
    evidence = build_experiment_evidence(Path('evidence/neural.json'), {'seed': 7})
    assert isinstance(evidence, ExperimentEvidence)
    assert evidence.output_path == 'evidence/neural.json'
    assert len(evidence.config_sha256) == 64
    assert evidence.git_commit
    assert evidence.runtime.python
    assert evidence.runtime.model_bytes >= 0


def test_synthetic_smoke_trains_all_structure_families() -> None:
    report = run_synthetic_smoke(seed=11)
    assert set(report['models']) == {'e0_transformer', 'e2_tcn', 'e3_cross_stock'}
    assert report['primary_metric'] == 'absolute_top5_return'
    assert report['config']['mode'] == 'smoke'
    assert all(np.isfinite(item['absolute_top5_return']) for item in report['models'].values())
    assert all(item['epochs'] >= 1 for item in report['models'].values())
    assert all(item['runtime']['model_bytes'] > 0 for item in report['models'].values())
