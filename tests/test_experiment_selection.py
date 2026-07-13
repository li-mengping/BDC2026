from code.experiments.xgb_optuna import select_candidate


def test_select_candidate_prefers_pareto_stable_improvement():
    baseline = {'mean': 0.03, 'worst': 0.01, 'std': 0.02, 'seconds': 10.0}
    trials = [
        {'number': 1, 'mean': 0.032, 'worst': 0.009, 'std': 0.03, 'seconds': 10.0, 'params': {}},
        {'number': 2, 'mean': 0.031, 'worst': 0.015, 'std': 0.018, 'seconds': 11.0, 'params': {}},
    ]

    chosen = select_candidate(baseline, trials)

    assert chosen['number'] == 2
    assert chosen['improves_baseline'] is True
