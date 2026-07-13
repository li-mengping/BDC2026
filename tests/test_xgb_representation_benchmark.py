from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from code.experiments.xgb_representation_benchmark import (
    E1_VARIANTS,
    RepresentationBenchmarkConfig,
    build_variant_frame,
    fit_nested_booster,
    _pareto_decisions,
    parse_args,
    split_outer_inner,
    write_report,
)
from code.models.spine import FoldSpec


def _representation_frame() -> pd.DataFrame:
    return pd.DataFrame({
        '日期': pd.to_datetime([
            '2025-01-06', '2025-01-06', '2025-01-06',
            '2025-01-13', '2025-01-13', '2025-01-13',
        ]),
        '股票代码': ['000001', '000002', '000003'] * 2,
        'instrument': [0.0, 1.0, 2.0] * 2,
        'factor': [1.0, 2.0, 1000.0, 4.0, 5.0, 6.0],
        'factor_two': [9.0, 8.0, 7.0, 3.0, 2.0, 1.0],
        'label': [0.01, 0.02, 0.03, -0.01, 0.00, 0.01],
    })


def test_e1_declares_four_nested_outer_folds_and_required_variants() -> None:
    config = RepresentationBenchmarkConfig()
    assert config.outer_folds == 4
    assert config.inner_validation_weeks >= 1
    assert set(E1_VARIANTS) == {
        'raw_ordered_identity',
        'identity_disabled',
        'cross_section_rank_identity_disabled',
        'cross_section_robust_zscore_identity_disabled',
        'categorical_identity',
    }


def test_cross_section_rank_is_date_local_and_outlier_scale_invariant() -> None:
    frame = _representation_frame()
    ranked, feature_names, categorical = build_variant_frame(
        frame, ['instrument', 'factor', 'factor_two'],
        'cross_section_rank_identity_disabled', ('000001', '000002', '000003'),
    )
    scaled = frame.copy()
    scaled['factor'] = scaled['factor'] * 100_000 + 17
    ranked_scaled, _, _ = build_variant_frame(
        scaled, ['instrument', 'factor', 'factor_two'],
        'cross_section_rank_identity_disabled', ('000001', '000002', '000003'),
    )
    assert categorical is False
    assert 'instrument' not in feature_names
    assert np.allclose(ranked.to_numpy(), ranked_scaled.to_numpy(), equal_nan=True)
    assert ranked.loc[:2, 'factor'].tolist() == pytest.approx([-0.5, 0.0, 0.5])


def test_robust_zscore_is_week_local_affine_invariant_and_finite_with_zero_mad() -> None:
    frame = _representation_frame()
    frame['factor'] = [1.0, 1.0, 1e20, 4.0, 4.0, 4.0]
    robust, feature_names, categorical = build_variant_frame(
        frame, ['instrument', 'factor', 'factor_two'],
        'cross_section_robust_zscore_identity_disabled',
        ('000001', '000002', '000003'),
    )
    scaled = frame.copy()
    scaled['factor'] = scaled['factor'] * 23.0 + 91.0
    scaled['factor_two'] = scaled['factor_two'] * 7.0 - 13.0
    robust_scaled, _, _ = build_variant_frame(
        scaled, ['instrument', 'factor', 'factor_two'],
        'cross_section_robust_zscore_identity_disabled',
        ('000001', '000002', '000003'),
    )
    assert categorical is False
    assert 'instrument' not in feature_names
    assert np.isfinite(robust.to_numpy()).all()
    assert np.abs(robust.to_numpy()).max() <= 5.0
    assert np.allclose(robust.to_numpy(), robust_scaled.to_numpy(), atol=1e-6)
    assert robust.loc[3:5, 'factor'].tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_identity_variants_do_not_treat_categorical_id_as_continuous() -> None:
    frame = _representation_frame()
    raw, raw_features, raw_categorical = build_variant_frame(
        frame, ['instrument', 'factor'], 'raw_ordered_identity',
        ('000001', '000002', '000003'),
    )
    disabled, disabled_features, _ = build_variant_frame(
        frame, ['instrument', 'factor'], 'identity_disabled',
        ('000001', '000002', '000003'),
    )
    categorical, categorical_features, categorical_enabled = build_variant_frame(
        frame, ['instrument', 'factor'], 'categorical_identity',
        ('000001', '000002', '000003'),
    )
    assert raw_categorical is False and pd.api.types.is_float_dtype(raw['instrument'])
    assert raw_features == ['instrument', 'factor']
    assert disabled_features == ['factor'] and 'instrument' not in disabled
    assert categorical_enabled is True
    assert categorical_features == ['instrument', 'factor']
    assert isinstance(categorical['instrument'].dtype, pd.CategoricalDtype)
    dmatrix = xgb.DMatrix(
        categorical,
        label=frame['label'].to_numpy(np.float32),
        group=np.asarray([3, 3], np.uint32),
        enable_categorical=True,
    )
    assert dmatrix.feature_types[0] == 'c'
    booster = xgb.train(
        {'objective': 'rank:pairwise', 'tree_method': 'hist', 'seed': 42, 'nthread': 1},
        dmatrix,
        num_boost_round=1,
    )
    assert booster.predict(dmatrix).shape == (len(frame),)


def test_inner_checkpoint_window_is_strictly_inside_outer_training_period() -> None:
    dates = pd.date_range('2025-01-06', periods=12, freq='7D')
    frame = pd.DataFrame({
        '日期': np.repeat(dates, 5),
        '股票代码': [f'{index:06d}' for _ in dates for index in range(5)],
    })
    fold = FoldSpec(
        name='rolling_1',
        train_start=pd.Timestamp('2025-01-06'),
        train_end=pd.Timestamp('2025-03-03'),
        validation_start=pd.Timestamp('2025-03-03'),
        validation_end=pd.Timestamp('2025-03-31'),
    )
    inner_train, inner_validation, outer_validation = split_outer_inner(frame, fold, 2)
    assert inner_train['日期'].max() < inner_validation['日期'].min()
    assert inner_validation['日期'].max() < outer_validation['日期'].min()
    assert inner_validation['日期'].nunique() == 2
    assert outer_validation['日期'].min() == fold.validation_start


def test_outer_validation_is_never_passed_to_checkpoint_selection(monkeypatch) -> None:
    def matrix(label: float) -> xgb.DMatrix:
        return xgb.DMatrix(
            np.arange(12, dtype=np.float32).reshape(6, 2),
            label=np.full(6, label, np.float32),
            group=np.asarray([6], np.uint32),
        )

    inner_train = matrix(1.0)
    inner_validation = matrix(2.0)
    outer_train = matrix(3.0)
    outer_validation = matrix(99.0)
    calls = []

    class FakeBooster:
        def save_raw(self, raw_format='ubj'):
            return b'model'

    def fake_train(params, dtrain, num_boost_round, evals=(), **kwargs):
        calls.append((dtrain, tuple(evals), num_boost_round))
        if kwargs.get('evals_result') is not None:
            kwargs['evals_result']['inner_validation'] = {'top5_return': [0.1, 0.3, 0.2]}
        return FakeBooster()

    monkeypatch.setattr('code.experiments.xgb_representation_benchmark.xgb.train', fake_train)
    _, selection, _ = fit_nested_booster(
        inner_train, inner_validation, outer_train, outer_validation,
        {'objective': 'rank:pairwise'}, max_rounds=3,
    )
    assert selection['best_iteration'] == 1
    assert len(calls) == 2
    assert calls[0][1] == ((inner_validation, 'inner_validation'),)
    assert calls[1][1] == ()
    assert calls[1][0] is outer_train
    assert all(candidate is not outer_validation for _, evals, _ in calls for candidate, _ in evals)
    assert calls[1][2] == selection['best_iteration'] + 1


def test_output_is_required_and_repository_relative(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_args([])
    assert parse_args(['--output', 'temp/e1.json']).output == Path('temp/e1.json')
    with pytest.raises(ValueError, match='relative'):
        write_report(tmp_path / 'e1.json', {})
    with pytest.raises(ValueError, match='relative'):
        write_report(Path('temp/../outside.json'), {})


def test_e1_source_is_portable_and_does_not_bind_loop_or_production_config() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / 'code/experiments/xgb_representation_benchmark.py'
    ).read_text(encoding='utf-8')
    assert 'loop-00' not in source
    assert 'C:\\Users' not in source
    assert "required=True" in source
    assert 'production_config_changed' in source


def test_pareto_decision_rejects_mean_regression_and_incomplete_resources() -> None:
    def payload(mean: float, worst: float) -> dict:
        return {'aggregate': {
            'mean_return': mean, 'worst_return': worst, 'positive_rate': 0.5, 'rank_ic': 0.1,
            'return_std': 0.2, 'train_seconds': 1.0, 'predict_seconds': 0.1, 'model_bytes': 100,
        }}
    models = {
        'raw_ordered_identity': payload(0.03, -0.02),
        'identity_disabled': payload(0.02, -0.01),
    }
    decisions = _pareto_decisions(models)
    assert decisions['identity_disabled']['promote'] is False
    assert 'mean_return' in decisions['identity_disabled']['regressions']
    assert decisions['identity_disabled']['resource_complete'] is False
