from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StockWeek:
    """一只股票的完整自然周行情，周一到周五缺一则不能构造。"""

    stock_id: str
    start_date: pd.Timestamp
    frame: pd.DataFrame

    def __post_init__(self) -> None:
        stock_id = str(self.stock_id)
        start_date = pd.Timestamp(self.start_date)
        frame = self.frame
        if len(frame) != 5:
            raise ValueError(f'incomplete stock week: {stock_id} {start_date.date()}')

        object.__setattr__(self, 'stock_id', stock_id)
        object.__setattr__(self, 'start_date', start_date)
        object.__setattr__(self, 'frame', frame)

    @property
    def end_date(self) -> pd.Timestamp:
        """自然周周五。"""
        return self.start_date + pd.Timedelta(days=4)

    def get_return(self) -> float:
        """计算周一开盘到周五收盘收益。"""
        start_open = float(self.frame.iloc[0]['开盘'])
        end_close = float(self.frame.iloc[-1]['收盘'])
        if start_open <= 1e-12:
            raise ValueError(f'{self.stock_id} invalid week open')
        return (end_close - start_open) / start_open

    def to_frame(self) -> pd.DataFrame:
        """返回周内行情副本。"""
        return self.frame.copy()


@dataclass(frozen=True)
class StockData:
    """一条周频样本，历史周用于预测未来周。"""

    stock_id: str
    history_weeks: tuple[StockWeek, ...]
    future_week: StockWeek

    @property
    def future_return(self) -> float:
        """未来周收益标签。"""
        return self.future_week.get_return()


def week_start(date) -> pd.Timestamp:
    """返回自然周周一。"""
    date = normalize_date(date)
    return date - pd.Timedelta(days=date.weekday())


def complete_week_starts(dates) -> pd.DatetimeIndex:
    """从日期序列提取周一到周五齐全的自然周起点。"""
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    grouped: dict[pd.Timestamp, set[pd.Timestamp]] = {}
    for date in dates:
        grouped.setdefault(week_start(date), set()).add(pd.Timestamp(date).normalize())

    starts = [
        start_date
        for start_date, week_dates in grouped.items()
        if week_dates == {start_date + pd.Timedelta(days=i) for i in range(5)}
    ]
    return pd.DatetimeIndex(sorted(starts))


def normalize_stock_id_series(series: pd.Series) -> pd.Series:
    """批量规范股票代码，遇到非法值直接失败。"""
    ids = series.astype(str).str.extract(r'(\d+)', expand=False)
    if ids.isna().any():
        raise ValueError('invalid stock id series')
    return ids.str.zfill(6)


def normalize_date(value) -> pd.Timestamp:
    """日期统一为 pandas normalize 后的 Timestamp。"""
    date = pd.to_datetime(value, errors='coerce')
    if pd.isna(date):
        raise ValueError(f'invalid date: {value}')
    return pd.Timestamp(date).normalize()


def normalize_date_series(series: pd.Series) -> pd.Series:
    """批量规范日期，遇到非法值直接失败。"""
    dates = pd.to_datetime(series, errors='coerce').dt.normalize()
    if dates.isna().any():
        raise ValueError('invalid date series')
    return dates
