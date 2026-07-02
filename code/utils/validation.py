from dataclasses import dataclass

import numpy as np
import pandas as pd

from .runtime_split import build_prediction_samples, build_stock_data_samples, build_train_returns
from .stock import StockData, StockWeek, complete_week_starts, normalize_date, week_start


@dataclass(frozen=True)
class ValidationFold:
    """一个训练验证时间块。"""

    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


@dataclass(frozen=True)
class ValidationPlan:
    """统一管理 holdout 和 rolling_kfold 验证切分。"""

    df: pd.DataFrame
    start_date: pd.Timestamp
    test_date: pd.Timestamp
    week_starts: pd.DatetimeIndex
    target_pos: int
    validation_start_pos: int
    holdout_test_start_pos: int
    validation_config: dict

    @property
    def validation_target_start(self) -> pd.Timestamp:
        """holdout 验证目标周起点。"""
        return pd.Timestamp(self.week_starts[self.validation_start_pos]).normalize()

    @property
    def holdout_test_target_start(self) -> pd.Timestamp:
        """holdout 测试目标周起点。"""
        return pd.Timestamp(self.week_starts[self.holdout_test_start_pos]).normalize()

    @property
    def num_validation_weeks(self) -> int:
        """holdout 验证周数。"""
        return int(self.validation_config['num_validation_weeks'])

    @property
    def num_test_weeks(self) -> int:
        """holdout 测试周数。"""
        return int(self.validation_config['num_test_weeks'])

    def holdout_split(self) -> ValidationFold:
        """正式训练使用的 train/validation/test 前两段。"""
        return ValidationFold(
            name='holdout',
            train_start=self.start_date,
            train_end=self.validation_target_start,
            validation_start=self.validation_target_start,
            validation_end=self.holdout_test_target_start,
        )

    def rolling_splits(self) -> tuple[ValidationFold, ...]:
        """从 holdout test 之前向前滚动生成 k fold 验证。"""
        folds = int(self.validation_config['rolling_folds'])
        validation_weeks = int(self.validation_config['rolling_validation_weeks'])
        gap_weeks = int(self.validation_config['rolling_gap_weeks'])
        min_train_weeks = int(self.validation_config['rolling_min_train_weeks'])
        start_pos = int(self.week_starts.searchsorted(self.start_date, side='left'))
        result = []

        for fold_idx in range(folds):
            validation_end_pos = self.holdout_test_start_pos - fold_idx * validation_weeks
            validation_start_pos = validation_end_pos - validation_weeks
            train_end_pos = validation_start_pos - gap_weeks
            if train_end_pos - start_pos < min_train_weeks:
                raise ValueError('rolling fold train weeks less than rolling_min_train_weeks')
            result.append(ValidationFold(
                name='rolling',
                train_start=pd.Timestamp(self.week_starts[start_pos]).normalize(),
                train_end=pd.Timestamp(self.week_starts[train_end_pos]).normalize(),
                validation_start=pd.Timestamp(self.week_starts[validation_start_pos]).normalize(),
                validation_end=pd.Timestamp(self.week_starts[validation_end_pos]).normalize(),
            ))

        return tuple(
            ValidationFold(
                name=f'rolling_{idx}',
                train_start=fold.train_start,
                train_end=fold.train_end,
                validation_start=fold.validation_start,
                validation_end=fold.validation_end,
            )
            for idx, fold in enumerate(sorted(result, key=lambda item: item.validation_start), start=1)
        )

    def validation_splits(self) -> tuple[ValidationFold, ...]:
        """按配置返回当前验证模式的切分。"""
        mode = self.validation_config['mode']
        if mode == 'holdout':
            return (self.holdout_split(),)
        if mode == 'rolling_kfold':
            return self.rolling_splits()
        raise ValueError(f'unsupported validation mode: {mode}')

    def input_start(self, target_date, input_window: int) -> pd.Timestamp:
        """按目标周和模型窗口返回输入窗口起点。"""
        target_pos = int(self.week_starts.searchsorted(week_start(target_date), side='left'))
        input_pos = target_pos - int(input_window)
        if input_pos < 0:
            raise ValueError(f'not enough history for input_window={input_window}')
        return pd.Timestamp(self.week_starts[input_pos]).normalize()

    def get_train_samples(self, fold: ValidationFold, input_window: int) -> tuple[StockData, ...]:
        """构造 fold 训练样本。"""
        return build_stock_data_samples(self.df, fold.train_start, fold.train_end, input_window)

    def get_validation_samples(self, fold: ValidationFold, input_window: int) -> tuple[StockData, ...]:
        """构造 fold 验证样本。"""
        return build_stock_data_samples(self.df, fold.validation_start, fold.validation_end, input_window)

    def get_test_samples(self, input_window: int) -> tuple[StockData, ...]:
        """构造 holdout test 样本。"""
        return build_stock_data_samples(self.df, self.holdout_test_target_start, self.test_date, input_window)

    def get_prediction_samples(self, input_window: int) -> tuple[tuple[str, tuple[StockWeek, ...]], ...]:
        """构造未来预测输入样本。"""
        return build_prediction_samples(
            self.df,
            self.input_start(self.test_date, input_window),
            self.test_date,
            input_window,
        )

    def get_train_returns(self, fold: ValidationFold, input_window: int) -> np.ndarray:
        """构造训练标签收益分布。"""
        returns = build_train_returns(self.df, fold.train_start, fold.train_end, input_window)
        if len(returns) == 0:
            raise ValueError('empty train returns')
        return returns


def build_validation_plan(df: pd.DataFrame, config: dict) -> ValidationPlan:
    """按配置生成唯一验证计划。"""
    start_date = normalize_date(config['start_date'])
    test_date = week_start(config['test_date'])
    validation_config = config['validation']
    num_validation_weeks = int(validation_config['num_validation_weeks'])
    num_test_weeks = int(validation_config['num_test_weeks'])
    if num_validation_weeks < 1 or num_test_weeks < 1:
        raise ValueError('num_validation_weeks and num_test_weeks must be positive')

    week_starts = complete_week_starts(df['日期'].unique())
    if len(week_starts) == 0:
        raise ValueError('no complete Monday-Friday trading weeks')

    target_pos = int(week_starts.searchsorted(test_date, side='left'))
    holdout_test_start_pos = target_pos - num_test_weeks
    validation_start_pos = holdout_test_start_pos - num_validation_weeks
    if validation_start_pos < 0 or holdout_test_start_pos < 0:
        raise ValueError('not enough complete weeks before test_date')

    return ValidationPlan(
        df=df,
        start_date=start_date,
        test_date=test_date,
        week_starts=week_starts,
        target_pos=target_pos,
        validation_start_pos=validation_start_pos,
        holdout_test_start_pos=holdout_test_start_pos,
        validation_config=validation_config,
    )
