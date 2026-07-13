from code.experiments.xgb_stock_identity_ablation import pareto_decision
from code.experiments.xgb_crossfit_checkpoint import select_round
from code.models.ranking_common import relevance_from_returns
import numpy as np


def test_structure_candidate_requires_full_pareto_improvement() -> None:
    baseline = {'mean': 0.03, 'worst': 0.01, 'std': 0.02}
    candidate = {'mean': 0.04, 'worst': 0.00, 'std': 0.01}

    decision = pareto_decision(baseline, candidate)

    assert decision['accepted'] is False
    assert decision['status'] == 'rejected_no_pareto_improvement'


def test_crossfit_round_does_not_read_held_out_fold() -> None:
    curves = {
        'fold_1': {'top5_excess_return': [100.0, -100.0]},
        'fold_2': {'top5_excess_return': [0.1, 0.2]},
        'fold_3': {'top5_excess_return': [0.1, 0.2]},
        'fold_4': {'top5_excess_return': [0.1, 0.2]},
    }

    selected = select_round(curves, held_out='fold_1')

    assert selected['round'] == 2
    assert 'fold_1' not in selected['selected_from_folds']


def test_relevance_preserves_weekly_return_order() -> None:
    returns = np.asarray([-0.1, 0.0, 0.2], dtype=np.float32)

    relevance = relevance_from_returns(returns, [3], max_relevance=30)

    assert relevance.tolist() == [0, 15, 30]
