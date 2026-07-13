import pandas as pd
import pytest

from code.config import config
from code.models.spine import EvaluationWindow, MetricReport, metric_schema, primary_portfolio_score
from code.models.xgboost.model import select_primary_checkpoint
from code.models.ranking_common import select_best_iteration
from code.utils.validation import build_validation_plan


def _market_dates(start: str, weeks: int) -> pd.DatetimeIndex:
    dates = []
    for monday in pd.date_range(start, periods=weeks, freq='7D'):
        dates.extend(monday + pd.to_timedelta(range(5), unit='D'))
    return pd.DatetimeIndex(dates)


def test_evaluation_return_uses_first_and_fifth_ordered_trading_day_open() -> None:
    frame = pd.DataFrame({
        '日期': pd.to_datetime([
            '2026-07-03', '2026-06-30', '2026-07-02', '2026-06-29', '2026-07-01',
        ]),
        '开盘': [110.0, 101.0, 103.0, 100.0, 102.0],
        '收盘': [1.0, 2.0, 3.0, 4.0, 5.0],
    })

    window = EvaluationWindow.from_frame(frame)

    assert window.return_value == pytest.approx(0.10)


def test_evaluation_return_does_not_depend_on_close() -> None:
    dates = pd.date_range('2026-06-29', periods=5, freq='D')
    first = EvaluationWindow.from_frame(pd.DataFrame({
        '日期': dates,
        '开盘': [100.0, 101.0, 102.0, 103.0, 110.0],
        '收盘': [0.0] * 5,
    }))
    second = EvaluationWindow.from_frame(pd.DataFrame({
        '日期': dates,
        '开盘': [100.0, 101.0, 102.0, 103.0, 110.0],
        '收盘': [9999.0] * 5,
    }))

    assert first.return_value == second.return_value


@pytest.mark.parametrize('dates', [
    pd.date_range('2026-06-29', periods=4, freq='D'),
    pd.to_datetime(['2026-06-29', '2026-06-30', '2026-07-01', '2026-07-01', '2026-07-03']),
])
def test_evaluation_window_requires_five_unique_trading_days(dates) -> None:
    frame = pd.DataFrame({'日期': dates, '开盘': range(len(dates))})

    with pytest.raises(ValueError, match='five unique'):
        EvaluationWindow.from_frame(frame)


def test_validation_requires_explicit_full_monday_friday_training_policy() -> None:
    dates = _market_dates('2025-01-06', 20)
    frame = pd.DataFrame({'日期': dates})
    local_config = {
        **config,
        'start_date': '2025-01-06',
        'test_date': '2025-05-26',
        'data_cutoff': '2026-06-29',
        'validation': {
            **config['validation'],
            'num_validation_weeks': 2,
            'num_test_weeks': 2,
        },
    }
    local_config.pop('sample_calendar_policy', None)

    with pytest.raises(ValueError, match='sample_calendar_policy'):
        build_validation_plan(frame, local_config)


def test_validation_rejects_rows_at_or_after_competition_cutoff() -> None:
    dates = _market_dates('2026-01-05', 26).append(pd.DatetimeIndex(['2026-06-29']))
    frame = pd.DataFrame({'日期': dates})
    local_config = {
        **config,
        'start_date': '2026-01-05',
        'test_date': '2026-06-29',
        'validation': {
            **config['validation'],
            'num_validation_weeks': 2,
            'num_test_weeks': 2,
        },
    }

    with pytest.raises(ValueError, match='data cutoff'):
        build_validation_plan(frame, local_config)


def test_training_calendar_excludes_a_four_day_holiday_week() -> None:
    dates = _market_dates('2025-01-06', 20)
    holiday_week = pd.to_datetime(['2025-05-26', '2025-05-27', '2025-05-28', '2025-05-29'])
    frame = pd.DataFrame({'日期': dates.append(pd.DatetimeIndex(holiday_week))})
    local_config = {
        **config,
        'start_date': '2025-01-06',
        'test_date': '2025-06-02',
        'validation': {
            **config['validation'],
            'num_validation_weeks': 2,
            'num_test_weeks': 2,
        },
    }

    plan = build_validation_plan(frame, local_config)

    assert pd.Timestamp('2025-05-26') not in plan.week_starts


def test_metric_report_declares_absolute_portfolio_return_as_primary() -> None:
    report = MetricReport(
        portfolio_return=0.04,
        universe_return=0.01,
        rank_ic=0.2,
    )

    assert report.primary_metric == 'portfolio_return'
    assert report.primary_value == pytest.approx(0.04)
    assert report.excess_return == pytest.approx(0.03)


def test_checkpoint_selection_reports_absolute_return_metric() -> None:
    predictions = [
        pd.Series([0.0, 1.0, 2.0, 3.0, 4.0]).to_numpy(),
        pd.Series([4.0, 3.0, 2.0, 1.0, 0.0]).to_numpy(),
    ]
    labels = pd.Series([-0.1, 0.0, 0.1, 0.2, 0.3]).to_numpy()

    selection, _ = select_best_iteration(
        lambda iteration: predictions[iteration],
        labels,
        [5],
        2,
        'test',
    )

    assert selection['metric'] == 'validation_top5_return'


def test_formal_xgboost_checkpoint_ignores_diagnostic_excess_curve() -> None:
    selection = select_primary_checkpoint({
        'validation': {
            'top5_return': [0.01, 0.03, 0.02],
            'top5_excess_return': [0.04, 0.01, 0.06],
        },
    })

    assert selection['best_iteration'] == 1
    assert selection['metric'] == 'validation_top5_return'


def test_metric_schema_keeps_excess_return_as_diagnostic_only() -> None:
    metrics = {
        'pred_top5_return_avg': 0.04,
        'pred_top5_excess_return_avg': 0.09,
    }

    assert primary_portfolio_score(metrics) == pytest.approx(0.04)
    assert metric_schema() == {
        'primary_metric': 'portfolio_return',
        'primary_metric_field': 'pred_top5_return_avg',
        'diagnostic_metrics': ('excess_return', 'rank_ic'),
    }
