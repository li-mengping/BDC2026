from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('torch')

from code.experiments.real_structure_benchmark import (
    BenchmarkConfig,
    CrossSection,
    SEQUENCE_FEATURES,
    _pareto_analysis,
    daily_sequence_features,
    fit_nested_xgb_booster,
    split_outer_inner_sections,
    summarize_predictions,
    write_report,
)
from code.utils.stock import StockWeek


def _history_weeks() -> tuple[StockWeek, ...]:
    weeks = []
    close = 10.0
    for week_index in range(12):
        start = pd.Timestamp('2025-01-06') + pd.Timedelta(days=7 * week_index)
        dates = pd.date_range(start, periods=5, freq='D')
        rows = []
        for day_index, date in enumerate(dates):
            open_price = close * 1.001
            close = open_price * 1.002
            rows.append({
                '股票代码': '000001', '日期': date, '开盘': open_price, '收盘': close,
                '最高': close * 1.001, '最低': open_price * 0.999,
                '成交量': 1000 + week_index * 10 + day_index,
                '成交额': 10000 + week_index * 100 + day_index,
                '换手率': 1.0 + day_index / 10,
            })
        weeks.append(StockWeek('000001', start, pd.DataFrame(rows)))
    return tuple(weeks)


def test_daily_sequence_is_60_days_and_does_not_fit_future_statistics() -> None:
    values = daily_sequence_features(_history_weeks())
    assert values.shape == (60, len(SEQUENCE_FEATURES))
    assert np.isfinite(values).all()
    assert values[0, [0, 4, 5, 6]].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert values[-1, 1] == pytest.approx(0.002, abs=1e-6)


def test_daily_sequence_rejects_incomplete_or_out_of_order_history() -> None:
    weeks = list(_history_weeks())
    with pytest.raises(ValueError, match='12'):
        daily_sequence_features(weeks[:-1])
    weeks[0] = StockWeek(weeks[0].stock_id, weeks[0].start_date, weeks[0].frame.iloc[::-1])
    # StockWeek itself allows an unsorted frame; the benchmark must normalize and verify exact dates.
    assert daily_sequence_features(weeks).shape[0] == 60
    swapped = list(_history_weeks())
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValueError, match='ordered'):
        daily_sequence_features(swapped)


def test_daily_sequence_imputes_non_price_fields_from_past_only() -> None:
    weeks = list(_history_weeks())
    frame = weeks[0].frame.copy()
    frame.loc[0, ['成交量', '成交额', '换手率']] = np.nan
    frame.loc[2, ['成交量', '成交额', '换手率']] = np.nan
    weeks[0] = StockWeek(weeks[0].stock_id, weeks[0].start_date, frame)
    values = daily_sequence_features(weeks)
    assert np.isfinite(values).all()
    assert values[0, 7] == 0.0
    assert values[2, 7] == values[1, 7]


def test_summary_uses_absolute_weekly_top5_return_and_rank_ic() -> None:
    sections = (
        CrossSection(pd.Timestamp('2025-04-07'), tuple(str(i) for i in range(6)),
                     np.zeros((6, 60, len(SEQUENCE_FEATURES)), np.float32),
                     np.asarray([0.10, 0.08, 0.06, 0.04, 0.02, -0.5], np.float32)),
    )
    summary = summarize_predictions((np.asarray([6, 5, 4, 3, 2, 1]),), sections)
    assert summary['weekly_returns'] == pytest.approx([0.06])
    assert summary['mean_return'] == pytest.approx(0.06)
    assert summary['rank_ic'] == pytest.approx(1.0)


def test_smoke_and_full_budgets_are_bounded() -> None:
    smoke = BenchmarkConfig.for_mode('smoke')
    full = BenchmarkConfig.for_mode('full')
    assert smoke.folds == 1 and smoke.neural_epochs == 1
    assert full.folds == 4 and full.neural_epochs > smoke.neural_epochs
    assert full.xgb_rounds >= smoke.xgb_rounds
    assert smoke.inner_validation_weeks == full.inner_validation_weeks == 4


def _section(target_week: str) -> CrossSection:
    return CrossSection(
        pd.Timestamp(target_week),
        tuple(str(index) for index in range(6)),
        np.zeros((6, 60, len(SEQUENCE_FEATURES)), np.float32),
        np.linspace(-0.03, 0.03, 6, dtype=np.float32),
    )


def test_nested_split_uses_last_four_outer_train_weeks() -> None:
    sections = tuple(_section(date) for date in pd.date_range('2025-01-06', periods=8, freq='7D'))
    inner_train, inner_validation = split_outer_inner_sections(sections, inner_validation_weeks=4)
    assert [section.target_week for section in inner_train] == [section.target_week for section in sections[:4]]
    assert [section.target_week for section in inner_validation] == [
        section.target_week for section in sections[-4:]
    ]
    assert inner_train[-1].target_week < inner_validation[0].target_week


def test_nested_split_rejects_insufficient_or_unordered_outer_train() -> None:
    sections = tuple(_section(date) for date in pd.date_range('2025-01-06', periods=4, freq='7D'))
    with pytest.raises(ValueError, match='insufficient'):
        split_outer_inner_sections(sections, inner_validation_weeks=4)
    with pytest.raises(ValueError, match='ordered'):
        split_outer_inner_sections(tuple(reversed(sections)) + (_section('2025-02-03'),), 4)


def test_xgboost_checkpoint_uses_inner_only_then_refits_outer_train(monkeypatch) -> None:
    import code.experiments.real_structure_benchmark as benchmark_module

    inner_train = object()
    inner_validation = object()
    outer_train = object()
    calls = []
    refit_booster = object()

    def fake_train(params, matrix, **kwargs):
        calls.append((matrix, kwargs.get('evals', ()), kwargs['num_boost_round']))
        if len(calls) == 1:
            kwargs['evals_result']['inner_validation'] = {'top5_return': [0.1, 0.3, 0.2]}
            return object()
        return refit_booster

    monkeypatch.setattr(benchmark_module.xgb, 'train', fake_train)
    booster, selection, runtime = fit_nested_xgb_booster(
        inner_train, inner_validation, outer_train, {'seed': 42}, max_rounds=3,
    )
    assert calls[0] == (inner_train, [(inner_validation, 'inner_validation')], 3)
    assert calls[1] == (outer_train, (), 2)
    assert booster is refit_booster
    assert selection['best_iteration'] == 1
    assert selection['selected_rounds'] == 2
    assert runtime['train_seconds'] == pytest.approx(
        runtime['checkpoint_train_seconds'] + runtime['outer_train_refit_seconds']
    )


def test_xgboost_four_nested_splits_slice_one_precomputed_feature_frame(monkeypatch) -> None:
    import code.experiments.real_structure_benchmark as benchmark_module

    sections = tuple(_section(date) for date in pd.date_range('2025-01-06', periods=4, freq='7D'))
    shared_frame = pd.DataFrame({'sentinel': [1]})
    seen_frames = []

    class FakeMatrix:
        pass

    class FakeBooster:
        def predict(self, matrix):
            return sections[-1].returns.copy()

        def save_raw(self, raw_format):
            return b'model'

    def fake_matrix(frame, selected_sections, features):
        seen_frames.append(frame)
        selected = selected_sections[0]
        selected_frame = pd.DataFrame({
            '日期': [selected.target_week] * len(selected.stock_ids),
            '股票代码': selected.stock_ids,
            'label': selected.returns,
        })
        return selected_frame, FakeMatrix(), [len(selected.stock_ids)]

    monkeypatch.setattr(benchmark_module, '_matrix_from_shared_xgb_frame', fake_matrix)
    monkeypatch.setattr(
        benchmark_module, 'fit_nested_xgb_booster',
        lambda *args: (FakeBooster(), {'best_iteration': 0, 'selected_rounds': 1}, {
            'checkpoint_train_seconds': 0.1,
            'outer_train_refit_seconds': 0.1,
            'train_seconds': 0.2,
        }),
    )
    result = benchmark_module._xgb_fold(
        shared_frame, ['f1'], sections[:1], sections[1:2], sections[2:3], sections[3:],
        {'model_params': {'xgb_rank_pairwise': {'device': 'cpu'}}},
        BenchmarkConfig.for_mode('smoke'),
    )
    assert seen_frames == [shared_frame] * 4
    assert result['checkpoint']['selected_rounds'] == 1


def test_neural_checkpoint_and_refit_keep_outer_validation_isolated(monkeypatch) -> None:
    import code.experiments.real_structure_benchmark as benchmark_module

    inner_train = (_section('2025-01-06'),)
    inner_validation = (_section('2025-01-13'),)
    outer_train = inner_train + inner_validation
    outer_validation = (_section('2025-01-20'),)
    calls = []

    class FakeParameter:
        def numel(self):
            return 1

        def element_size(self):
            return 4

    class FakeModel:
        def parameters(self):
            return iter((FakeParameter(),))

    def fake_select(family, train, validation, benchmark, device):
        calls.append(('select', family, train, validation))
        return {'best_iteration': 0, 'best_epoch': 1}, 0.1, 0

    def fake_train(model, sections, epochs, benchmark):
        calls.append(('refit', sections, epochs))

    def fake_predict(model, sections):
        calls.append(('predict', sections))
        return [section.returns.copy() for section in sections]

    monkeypatch.setattr(benchmark_module, '_select_neural_epoch', fake_select)
    monkeypatch.setattr(benchmark_module, '_build_neural_model', lambda *args: FakeModel())
    monkeypatch.setattr(benchmark_module, '_train_neural_epochs', fake_train)
    monkeypatch.setattr(benchmark_module, '_predict_neural', fake_predict)
    monkeypatch.setattr(benchmark_module, '_runtime_start', lambda device: 0.0)
    monkeypatch.setattr(benchmark_module, '_runtime_stop', lambda device, started: (0.2, 0))
    result = benchmark_module._neural_fold(
        'e0_transformer', inner_train, inner_validation, outer_train, outer_validation,
        BenchmarkConfig.for_mode('smoke'),
    )
    assert calls[0] == ('select', 'e0_transformer', inner_train, inner_validation)
    assert calls[1] == ('refit', outer_train, 1)
    assert calls[2] == ('predict', outer_validation)
    assert result['checkpoint']['best_epoch'] == 1


def test_pareto_analysis_records_observed_frontier_and_resource_gap() -> None:
    def aggregate(mean_return, worst_return, train_seconds, peak_vram_bytes):
        return {
            'mean_return': mean_return,
            'worst_return': worst_return,
            'positive_rate': 0.5,
            'rank_ic': 0.1,
            'return_std': 0.02,
            'train_seconds': train_seconds,
            'predict_seconds': 1.0,
            'model_bytes': 100,
            'peak_vram_bytes': peak_vram_bytes,
        }

    models = {
        'xgboost_rank_pairwise': {'aggregate': aggregate(0.02, -0.01, 5.0, None)},
        'candidate': {'aggregate': aggregate(0.03, 0.0, 4.0, 2048)},
        'dominated': {'aggregate': aggregate(0.01, -0.02, 6.0, 4096)},
    }
    analysis = _pareto_analysis(models)
    assert analysis['frontier_observed_dimensions'] == ['candidate']
    assert analysis['models']['candidate']['nondominated_observed_dimensions'] is True
    assert analysis['models']['candidate']['complete_pareto_improvement_over_baseline'] is True
    assert analysis['models']['candidate']['eligible_for_promotion'] is True
    assert analysis['models']['candidate']['promote'] is False
    assert analysis['models']['xgboost_rank_pairwise']['resource_gap'] == ['peak_vram_bytes']


def test_report_output_must_be_repository_relative(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='relative'):
        write_report(tmp_path / 'report.json', {'synthetic': False})


def test_real_benchmark_has_no_hardcoded_loop_or_official_parity_claim() -> None:
    source = (Path(__file__).resolve().parents[1] / 'code/experiments/real_structure_benchmark.py').read_text(
        encoding='utf-8',
    )
    assert 'loop-008' not in source
    assert 'official_like' not in source
    assert 'clean_room_official_contract' in source
    assert "required=True" in source
