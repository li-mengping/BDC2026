from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .stock import (
    StockData,
    StockWeek,
    complete_week_starts,
    normalize_date,
    normalize_date_series,
    normalize_stock_id_series,
    week_start,
)


REQUIRED_COLUMNS = {
    '股票代码',
    '日期',
    '开盘',
    '收盘',
    '最高',
    '最低',
    '成交量',
    '成交额',
    '振幅',
    '涨跌额',
    '换手率',
    '涨跌幅',
}


def load_market_data(config: dict) -> pd.DataFrame:
    """读取行情数据，并做最小字段校验。"""
    path = Path(config['data_path']) / config['full_data_file']
    df = pd.read_csv(path, dtype={'股票代码': str})
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f'missing columns: {sorted(missing)}')

    # 统一主键格式，后续所有模块只处理规范数据。
    df['股票代码'] = normalize_stock_id_series(df['股票代码'])
    df['日期'] = normalize_date_series(df['日期'])
    return df.sort_values(['股票代码', '日期']).reset_index(drop=True)


@dataclass
class RuntimeSplit:
    """只保存运行范围，实际样本由 get 方法按需生成。"""

    df: pd.DataFrame
    start_date: pd.Timestamp
    test_date: pd.Timestamp
    input_window: int
    num_validation_weeks: int
    num_test_weeks: int
    week_starts: pd.DatetimeIndex
    validation_start_pos: int
    holdout_test_start_pos: int
    validation_input_pos: int
    holdout_test_input_pos: int
    test_input_pos: int
    target_pos: int

    @property
    def validation_target_start(self) -> pd.Timestamp:
        """validation 目标周起点。"""
        return pd.Timestamp(self.week_starts[self.validation_start_pos]).normalize()

    @property
    def validation_input_start(self) -> pd.Timestamp:
        """validation 输入窗口起点。"""
        return pd.Timestamp(self.week_starts[self.validation_input_pos]).normalize()

    @property
    def test_input_start(self) -> pd.Timestamp:
        """未来预测输入窗口起点。"""
        return pd.Timestamp(self.week_starts[self.test_input_pos]).normalize()

    @property
    def holdout_test_target_start(self) -> pd.Timestamp:
        """holdout test 目标周起点。"""
        return pd.Timestamp(self.week_starts[self.holdout_test_start_pos]).normalize()

    @property
    def holdout_test_input_start(self) -> pd.Timestamp:
        """holdout test 输入窗口起点。"""
        return pd.Timestamp(self.week_starts[self.holdout_test_input_pos]).normalize()

    def get_train_samples(self) -> tuple[StockData, ...]:
        """训练 StockData。"""
        return self.get_stock_data(self.start_date, self.validation_target_start)

    def get_validation_samples(self) -> tuple[StockData, ...]:
        """验证 StockData。"""
        return self.get_stock_data(self.validation_target_start, self.holdout_test_target_start)

    def get_test_samples(self) -> tuple[StockData, ...]:
        """holdout test StockData。"""
        return self.get_stock_data(self.holdout_test_target_start, self.test_date)

    def get_prediction_samples(self) -> tuple[tuple[str, tuple[StockWeek, ...]], ...]:
        """预测历史窗口。"""
        return build_prediction_samples(
            self.df,
            self.test_input_start,
            self.test_date,
            self.input_window,
        )

    def get_train_returns(self) -> np.ndarray:
        """训练区间未来周收益分布，不构造训练 StockData。"""
        returns = build_train_returns(
            self.df,
            self.start_date,
            self.validation_target_start,
            self.input_window,
        )
        if len(returns) == 0:
            raise ValueError('empty train returns')
        return returns

    def get_stock_data(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> tuple[StockData, ...]:
        """按目标周范围生成 StockData。"""
        return build_stock_data_samples(self.df, start_date, end_date, self.input_window)


def split_runtime_data(df: pd.DataFrame, config: dict) -> RuntimeSplit:
    """只计算运行范围，不提前构造样本。"""
    start_date = normalize_date(config['start_date'])
    test_date = week_start(config['test_date'])
    input_window = int(config['input_window'])
    num_validation_weeks = int(config['num_validation_weeks'])
    num_test_weeks = int(config['num_test_weeks'])
    if input_window < 1 or num_validation_weeks < 1 or num_test_weeks < 1:
        raise ValueError('input_window, num_validation_weeks and num_test_weeks must be positive')

    week_starts = complete_week_starts(df['日期'].unique())
    if len(week_starts) == 0:
        raise ValueError('no complete Monday-Friday trading weeks')

    target_pos = int(week_starts.searchsorted(test_date, side='left'))
    holdout_test_start_pos = target_pos - num_test_weeks
    validation_start_pos = holdout_test_start_pos - num_validation_weeks
    validation_input_pos = validation_start_pos - input_window
    holdout_test_input_pos = holdout_test_start_pos - input_window
    test_input_pos = target_pos - input_window
    if validation_input_pos < 0 or holdout_test_input_pos < 0 or test_input_pos < 0:
        raise ValueError('not enough history before test_date')

    return RuntimeSplit(
        df=df,
        start_date=start_date,
        test_date=test_date,
        input_window=input_window,
        num_validation_weeks=num_validation_weeks,
        num_test_weeks=num_test_weeks,
        week_starts=week_starts,
        validation_start_pos=validation_start_pos,
        holdout_test_start_pos=holdout_test_start_pos,
        validation_input_pos=validation_input_pos,
        holdout_test_input_pos=holdout_test_input_pos,
        test_input_pos=test_input_pos,
        target_pos=target_pos,
    )


def build_prediction_samples(
    df: pd.DataFrame,
    input_start: pd.Timestamp,
    test_date: pd.Timestamp,
    input_window: int,
) -> tuple[tuple[str, tuple[StockWeek, ...]], ...]:
    """只在预测输入范围内构造 StockWeek 对象。"""
    frame = df[(df['日期'] >= input_start) & (df['日期'] < test_date)].copy()
    return prediction_samples_from_weeks(build_stock_weeks(frame), input_start, test_date, input_window)


def prediction_samples_from_weeks(
    stock_weeks: dict[str, tuple[StockWeek, ...]],
    input_start: pd.Timestamp,
    test_date: pd.Timestamp,
    input_window: int,
) -> tuple[tuple[str, tuple[StockWeek, ...]], ...]:
    """从已构造的周对象中切出预测窗口。"""
    samples = []
    for stock_id, weeks in stock_weeks.items():
        history_weeks = tuple(week for week in weeks if input_start <= week.start_date < test_date)
        if len(history_weeks) == input_window:
            samples.append((stock_id, history_weeks))

    if not samples:
        raise ValueError('empty prediction samples')
    samples.sort(key=lambda sample: sample[0])
    return tuple(samples)


def stock_data_from_weeks(
    stock_weeks: dict[str, tuple[StockWeek, ...]],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    input_window: int,
) -> tuple[StockData, ...]:
    """从临时周对象生成训练/验证 StockData。"""
    samples = []
    for stock_id, weeks in stock_weeks.items():
        for idx in range(input_window, len(weeks)):
            future_week = weeks[idx]
            if start_date <= future_week.start_date < end_date:
                history_weeks = weeks[idx - input_window:idx]
                samples.append(StockData(stock_id, history_weeks, future_week))

    if not samples:
        raise ValueError('empty stock data samples')
    samples.sort(key=lambda sample: (sample.future_week.start_date, sample.stock_id))
    return tuple(samples)


def build_stock_data_samples(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    input_window: int,
) -> tuple[StockData, ...]:
    """只把样本区间及其历史窗口转换成 StockData。"""
    week_index = complete_stock_week_index(df[df['日期'] < end_date])
    selected_keys = select_sample_week_keys(week_index, start_date, end_date, input_window)
    if not selected_keys:
        raise ValueError('empty stock data samples')

    key_df = pd.DataFrame(selected_keys, columns=['股票代码', '_week_start'])
    frame = add_week_start(df[df['日期'] < end_date])
    frame = frame.merge(key_df, on=['股票代码', '_week_start'], how='inner')
    stock_weeks = build_stock_weeks(frame)
    return stock_data_from_weeks(stock_weeks, start_date, end_date, input_window)


def select_sample_week_keys(
    week_index: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    input_window: int,
) -> set[tuple[str, pd.Timestamp]]:
    """选择目标周和对应历史窗口的周键。"""
    selected: set[tuple[str, pd.Timestamp]] = set()
    for stock_id, stock_weeks in week_index.groupby('股票代码', sort=True):
        week_starts = tuple(stock_weeks['week_start'])
        for idx, week_start_date in enumerate(week_starts):
            if start_date <= week_start_date < end_date and idx >= input_window:
                for history_idx in range(idx - input_window, idx + 1):
                    selected.add((stock_id, week_starts[history_idx]))
    return selected


def build_stock_weeks(df: pd.DataFrame) -> dict[str, tuple[StockWeek, ...]]:
    """把给定 DataFrame 范围转换为完整自然周对象。"""
    frame = add_week_start(df)
    if frame.empty:
        return {}

    group_cols = ['股票代码', '_week_start']
    week_stats = frame.groupby(group_cols, sort=True).agg(
        row_count=('日期', 'size'),
        date_count=('日期', 'nunique'),
        weekday_count=('_weekday', 'nunique'),
        first_weekday=('_weekday', 'min'),
        last_weekday=('_weekday', 'max'),
    )
    complete_keys = week_stats[
        (week_stats['row_count'] == 5)
        & (week_stats['date_count'] == 5)
        & (week_stats['weekday_count'] == 5)
        & (week_stats['first_weekday'] == 0)
        & (week_stats['last_weekday'] == 4)
    ].reset_index()[group_cols]
    if complete_keys.empty:
        return {}

    complete_frame = frame.merge(complete_keys, on=group_cols, how='inner', sort=False)
    data_cols = [col for col in complete_frame.columns if col not in {'_weekday', '_week_start'}]
    week_frames = complete_frame[data_cols]
    stock_ids = complete_frame['股票代码'].to_numpy(copy=False)
    week_starts = complete_frame['_week_start'].to_numpy(copy=False)

    stock_weeks = {}
    for offset in range(0, len(complete_frame), 5):
        week_df = week_frames.iloc[offset:offset + 5]
        if len(week_df) != 5:
            raise ValueError('invalid complete stock week chunk')

        stock_id = stock_ids[offset]
        start_date = week_starts[offset]
        week = StockWeek(stock_id, start_date, week_df)
        stock_weeks.setdefault(stock_id, []).append(week)
    return {stock_id: tuple(weeks) for stock_id, weeks in stock_weeks.items()}


def build_train_returns(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    input_window: int,
) -> np.ndarray:
    """按训练目标周范围快速计算未来周收益分布。"""
    complete_weeks = complete_stock_week_index(df)
    selected = complete_weeks[
        (complete_weeks['week_index'] >= input_window)
        & (complete_weeks['week_start'] >= start_date)
        & (complete_weeks['week_start'] < end_date)
    ].copy()
    if (selected['start_open'] <= 1e-12).any():
        bad = selected.loc[selected['start_open'] <= 1e-12].iloc[0]
        raise ValueError(f"{bad['股票代码']} invalid week open")
    returns = (
        selected['end_close'].astype(float) - selected['start_open'].astype(float)
    ) / selected['start_open'].astype(float)
    return returns.to_numpy(dtype=np.float64)


def complete_stock_week_index(df: pd.DataFrame) -> pd.DataFrame:
    """按股票列出完整自然周及周收益所需端点。"""
    frame = df[['股票代码', '日期', '开盘', '收盘']].copy()
    frame = add_week_start(frame)
    frame = frame.sort_values(['股票代码', '日期'])

    group_cols = ['股票代码', '_week_start']
    grouped = frame.groupby(group_cols, sort=True)
    week_stats = grouped.agg(
        row_count=('日期', 'size'),
        date_count=('日期', 'nunique'),
        weekday_count=('_weekday', 'nunique'),
        first_weekday=('_weekday', 'min'),
        last_weekday=('_weekday', 'max'),
    )
    complete_weeks = week_stats[
        (week_stats['row_count'] == 5)
        & (week_stats['date_count'] == 5)
        & (week_stats['weekday_count'] == 5)
        & (week_stats['first_weekday'] == 0)
        & (week_stats['last_weekday'] == 4)
    ].copy()

    monday_open = (
        frame[frame['_weekday'] == 0]
        .groupby(group_cols, sort=True)['开盘']
        .first()
        .rename('start_open')
    )
    friday_close = (
        frame[frame['_weekday'] == 4]
        .groupby(group_cols, sort=True)['收盘']
        .last()
        .rename('end_close')
    )
    complete_weeks = complete_weeks.join(monday_open).join(friday_close).dropna(subset=['start_open', 'end_close'])
    complete_weeks['week_index'] = complete_weeks.groupby(level=0).cumcount()
    complete_weeks = complete_weeks.reset_index().rename(columns={'_week_start': 'week_start'})
    return complete_weeks[['股票代码', 'week_start', 'week_index', 'start_open', 'end_close']]


def add_week_start(df: pd.DataFrame) -> pd.DataFrame:
    """追加自然周键，只保留周一到周五。"""
    frame = df.copy()
    frame['_weekday'] = frame['日期'].dt.weekday
    frame = frame[frame['_weekday'].between(0, 4)].copy()
    frame['_week_start'] = frame['日期'] - pd.to_timedelta(frame['_weekday'], unit='D')
    return frame
