import numpy as np
import pandas as pd

from code.portfolio import postprocess


def test_direct_equal_weight_top5_skips_covariance_and_optimizer(monkeypatch) -> None:
    scored = pd.DataFrame({
        '股票代码': [f'{index:06d}' for index in range(1, 8)],
        'xgb_rank_pairwise_score': [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    })

    def fail(*args, **kwargs):
        raise AssertionError('direct Top5 must not execute covariance or mean-variance code')

    monkeypatch.setattr(postprocess, 'estimate_weekly_covariance', fail)
    monkeypatch.setattr(postprocess, 'optimize_mean_variance', fail)

    _, candidates, selected = postprocess.select_portfolio(
        scored=scored,
        raw_df=pd.DataFrame(),
        target_date='2026-06-29',
        train_returns=np.asarray([], dtype=np.float64),
        model_names=['xgb_rank_pairwise'],
        portfolio_config=postprocess.DEFAULT_PORTFOLIO_CONFIG,
    )

    assert candidates['stock_id'].tolist() == [f'{index:06d}' for index in range(1, 6)]
    assert selected['stock_id'].tolist() == [f'{index:06d}' for index in range(1, 6)]
    assert selected['final_weight'].tolist() == [0.2] * 5
    assert {
        'mu', 'selection_weight', 'selection_rank', 'portfolio_vol_contrib',
    } <= set(candidates.columns)
    assert {'final_weight', 'final_vol_contrib', 'final_rank'} <= set(selected.columns)
