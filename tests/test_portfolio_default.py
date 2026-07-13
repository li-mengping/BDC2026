from code.portfolio.postprocess import DEFAULT_PORTFOLIO_CONFIG


def test_default_portfolio_uses_validated_direct_top5_pool():
    assert DEFAULT_PORTFOLIO_CONFIG.final_k == 5
    assert DEFAULT_PORTFOLIO_CONFIG.union_top_k_by_model == (('xgb_rank_pairwise', 5),)
    assert DEFAULT_PORTFOLIO_CONFIG.weight_strategy == 'equal'
