import pandas as pd

from code.utils.runtime_split import build_stock_data_samples, build_train_returns
from code.utils.stock import StockWeek, has_contiguous_history


def test_week_return_matches_competition_open_to_open_rule():
    frame = pd.DataFrame({
        '开盘': [100.0, 101.0, 102.0, 103.0, 110.0],
        '收盘': [101.0, 102.0, 103.0, 104.0, 150.0],
    })

    result = StockWeek('000001', pd.Timestamp('2026-06-01'), frame).get_return()

    assert result == 0.10


def test_stock_history_cannot_cross_a_missing_eligible_week() -> None:
    eligible_weeks = pd.to_datetime(['2026-01-05', '2026-01-12', '2026-01-19'])
    history = (pd.Timestamp('2026-01-05'),)

    assert has_contiguous_history(
        history,
        pd.Timestamp('2026-01-19'),
        eligible_weeks,
    ) is False


def test_training_samples_drop_stock_that_missed_an_eligible_market_week() -> None:
    rows = []
    for stock_id, monday_dates in {
        '000001': ['2026-01-05', '2026-01-19'],
        '000002': ['2026-01-05', '2026-01-12', '2026-01-19'],
    }.items():
        for monday in pd.to_datetime(monday_dates):
            for offset in range(5):
                rows.append({
                    '股票代码': stock_id,
                    '日期': monday + pd.Timedelta(days=offset),
                    '开盘': 10.0 + offset,
                    '收盘': 10.5 + offset,
                })
    frame = pd.DataFrame(rows)

    samples = build_stock_data_samples(
        frame,
        pd.Timestamp('2026-01-19'),
        pd.Timestamp('2026-01-26'),
        input_window=1,
    )

    assert [sample.stock_id for sample in samples] == ['000002']
    returns = build_train_returns(
        frame,
        pd.Timestamp('2026-01-19'),
        pd.Timestamp('2026-01-26'),
        input_window=1,
    )
    assert returns.tolist() == [0.4]
