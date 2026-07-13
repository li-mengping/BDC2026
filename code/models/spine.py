"""模型中立的训练与评估契约，供树模型和后续神经排序器共同复用。"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketPanel:
    """规范化并按股票、日期排序的行情面板。"""

    frame: pd.DataFrame

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> 'MarketPanel':
        required = {'股票代码', '日期', '开盘'}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f'market panel missing columns: {sorted(missing)}')
        normalized = frame.copy()
        normalized['股票代码'] = normalized['股票代码'].astype(str).str.extract(r'(\d+)', expand=False).str.zfill(6)
        normalized['日期'] = pd.to_datetime(normalized['日期'], errors='coerce').dt.normalize()
        if normalized[['股票代码', '日期']].isna().any().any():
            raise ValueError('market panel contains invalid stock id or date')
        if normalized.duplicated(['股票代码', '日期']).any():
            raise ValueError('market panel contains duplicate stock-date keys')
        normalized = normalized.sort_values(['股票代码', '日期']).reset_index(drop=True)
        return cls(normalized)


@dataclass(frozen=True)
class EvaluationWindow:
    """比赛评分窗口：五个有序评估交易日的开盘价。"""

    dates: tuple[pd.Timestamp, ...]
    opens: tuple[float, ...]

    def __post_init__(self) -> None:
        dates = tuple(pd.Timestamp(value).normalize() for value in self.dates)
        opens = tuple(float(value) for value in self.opens)
        if len(dates) != 5 or len(set(dates)) != 5:
            raise ValueError('evaluation window requires five unique trading days')
        if tuple(sorted(dates)) != dates:
            raise ValueError('evaluation trading days must be ordered')
        if len(opens) != 5 or not np.isfinite(opens).all():
            raise ValueError('evaluation window requires five finite open prices')
        if opens[0] <= 1e-12:
            raise ValueError('evaluation first-day open must be positive')
        object.__setattr__(self, 'dates', dates)
        object.__setattr__(self, 'opens', opens)

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        date_column: str = '日期',
        open_column: str = '开盘',
    ) -> 'EvaluationWindow':
        missing = {date_column, open_column} - set(frame.columns)
        if missing:
            raise ValueError(f'evaluation window missing columns: {sorted(missing)}')
        ordered = frame[[date_column, open_column]].copy()
        ordered[date_column] = pd.to_datetime(ordered[date_column], errors='coerce').dt.normalize()
        if ordered[date_column].isna().any():
            raise ValueError('evaluation window contains invalid trading days')
        ordered = ordered.sort_values(date_column).reset_index(drop=True)
        return cls(tuple(ordered[date_column]), tuple(ordered[open_column]))

    @property
    def return_value(self) -> float:
        """第1个评估交易日开盘至第5个评估交易日开盘的收益。"""
        return (self.opens[4] - self.opens[0]) / self.opens[0]


@dataclass(frozen=True)
class FoldSpec:
    """左闭右开的时序训练/验证切分。"""

    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp

    def __post_init__(self) -> None:
        for field_name in ('train_start', 'train_end', 'validation_start', 'validation_end'):
            object.__setattr__(self, field_name, pd.Timestamp(getattr(self, field_name)).normalize())
        if self.train_start >= self.train_end:
            raise ValueError('train_start must be before train_end')
        if self.train_end > self.validation_start:
            raise ValueError('training must end before validation starts')
        if self.validation_start >= self.validation_end:
            raise ValueError('validation_start must be before validation_end')


@dataclass(frozen=True)
class RankDataset:
    """按评估截面分组的模型中立排序数据集。"""

    frame: pd.DataFrame
    x: np.ndarray
    raw_label: np.ndarray
    relevance: np.ndarray
    groups: Sequence[int]

    def __post_init__(self) -> None:
        x = np.asarray(self.x)
        raw_label = np.asarray(self.raw_label)
        relevance = np.asarray(self.relevance)
        groups = tuple(int(value) for value in self.groups)
        rows = len(raw_label)
        if x.ndim != 2 or len(x) != rows or len(relevance) != rows or len(self.frame) != rows:
            raise ValueError('rank dataset row counts do not match')
        if not groups or any(value < 1 for value in groups) or sum(groups) != rows:
            raise ValueError('rank dataset group sizes do not match rows')
        object.__setattr__(self, 'x', x)
        object.__setattr__(self, 'raw_label', raw_label)
        object.__setattr__(self, 'relevance', relevance)
        object.__setattr__(self, 'groups', groups)


@dataclass(frozen=True)
class MetricReport:
    """以比赛绝对组合收益为主、超额收益仅作诊断的指标报告。"""

    portfolio_return: float
    universe_return: float
    rank_ic: float
    primary_metric: str = 'portfolio_return'

    @property
    def primary_value(self) -> float:
        return float(self.portfolio_return)

    @property
    def excess_return(self) -> float:
        return float(self.portfolio_return - self.universe_return)


def primary_portfolio_score(topk_metrics: Mapping[str, object]) -> float:
    """从 TopK 指标中读取正式选型使用的绝对组合收益。"""
    return float(topk_metrics['pred_top5_return_avg'])


def metric_schema() -> dict[str, object]:
    """写入 artifact 的稳定指标语义，避免把超额收益误作比赛主目标。"""
    return {
        'primary_metric': 'portfolio_return',
        'primary_metric_field': 'pred_top5_return_avg',
        'diagnostic_metrics': ('excess_return', 'rank_ic'),
    }


@dataclass(frozen=True)
class ModelArtifact:
    """可由任意排序模型 adapter 产出的最小 artifact 描述。"""

    family: str
    path: Path
    feature_names: tuple[str, ...]
    metadata: Mapping[str, object]


class StockIdentityMode(str, Enum):
    DISABLED = 'disabled'
    CATEGORICAL = 'categorical'
    EMBEDDING = 'embedding'


@dataclass(frozen=True)
class StockIdentitySpec:
    """股票身份编码契约；类别编号不能隐式作为连续浮点特征。"""

    mode: StockIdentityMode

    def encode(self, stock_ids: Sequence[str], mapping: Mapping[str, int]):
        normalized = [str(value).zfill(6) for value in stock_ids]
        unknown = [value for value in normalized if value not in mapping]
        if unknown:
            raise ValueError(f'unknown stock ids: {unknown[:5]}')
        if self.mode is StockIdentityMode.DISABLED:
            return None
        indices = [int(mapping[value]) for value in normalized]
        if self.mode is StockIdentityMode.CATEGORICAL:
            return pd.Series(pd.Categorical(indices, categories=sorted(set(mapping.values()))))
        if self.mode is StockIdentityMode.EMBEDDING:
            return np.asarray(indices, dtype=np.int64)
        raise ValueError(f'unsupported stock identity mode: {self.mode!r}')


@runtime_checkable
class RankerAdapter(Protocol):
    """模型族对训练和预测入口的最小结构化协议。"""

    def train(self) -> float:
        ...

    def predict(self) -> None:
        ...
