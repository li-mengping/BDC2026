from code.experiments.xgb_portfolio_scan import select_by_validation


def _row(top_k: int, validation_mean: float, validation_worst: float, holdout_mean: float) -> dict:
    return {
        'validation': {
            'candidate_top_k': top_k,
            'excess_mean': validation_mean,
            'excess_worst': validation_worst,
            'excess_std': 0.01,
        },
        'holdout': {'excess_mean': holdout_mean},
    }


def test_portfolio_selection_does_not_use_holdout() -> None:
    rows = [
        _row(10, 0.02, -0.01, 0.50),
        _row(5, 0.03, -0.01, -0.50),
    ]

    selected = select_by_validation(rows)

    assert selected['validation']['candidate_top_k'] == 5
