from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from code.models.spine import (
    FoldSpec,
    MarketPanel,
    ModelArtifact,
    RankDataset,
    RankerAdapter,
    StockIdentityMode,
    StockIdentitySpec,
)


def test_market_panel_normalizes_keys_without_mutating_source() -> None:
    source = pd.DataFrame({
        '股票代码': [1, 1],
        '日期': ['2026-01-06', '2026-01-05'],
        '开盘': [11.0, 10.0],
    })

    panel = MarketPanel.from_frame(source)

    assert panel.frame['股票代码'].tolist() == ['000001', '000001']
    assert panel.frame['日期'].tolist() == list(pd.to_datetime(['2026-01-05', '2026-01-06']))
    assert source['股票代码'].tolist() == [1, 1]


def test_fold_spec_rejects_overlapping_train_and_validation() -> None:
    with pytest.raises(ValueError, match='before validation'):
        FoldSpec(
            name='bad',
            train_start=pd.Timestamp('2025-01-01'),
            train_end=pd.Timestamp('2025-03-01'),
            validation_start=pd.Timestamp('2025-02-01'),
            validation_end=pd.Timestamp('2025-04-01'),
        )


def test_rank_dataset_checks_group_and_row_contract() -> None:
    with pytest.raises(ValueError, match='group sizes'):
        RankDataset(
            frame=pd.DataFrame({'label': [0.1, 0.2]}),
            x=np.ones((2, 1), dtype=np.float32),
            raw_label=np.asarray([0.1, 0.2]),
            relevance=np.asarray([0, 1]),
            groups=(1,),
        )


def test_stock_identity_contract_never_encodes_category_as_continuous_float() -> None:
    mapping = {'000001': 0, '000002': 1}

    disabled = StockIdentitySpec(StockIdentityMode.DISABLED).encode(['000001'], mapping)
    categorical = StockIdentitySpec(StockIdentityMode.CATEGORICAL).encode(['000001'], mapping)
    embedding = StockIdentitySpec(StockIdentityMode.EMBEDDING).encode(['000002'], mapping)

    assert disabled is None
    assert categorical.dtype.name == 'category'
    assert embedding.dtype == np.int64


def test_model_artifact_and_adapter_are_model_neutral() -> None:
    artifact = ModelArtifact(
        family='dummy',
        path=Path('model/dummy.bin'),
        feature_names=('f1',),
        metadata={'seed': 42},
    )

    class DummyRanker:
        def train(self) -> float:
            return 0.0

        def predict(self) -> None:
            return None

    assert artifact.family == 'dummy'
    assert isinstance(DummyRanker(), RankerAdapter)
