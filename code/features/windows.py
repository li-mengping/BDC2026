"""短窗口因子工程；`baseline.py` 通过本模块注册窗口特征。"""

from functools import partial

import numpy as np
import pandas as pd

from .window_1 import WINDOW_1_FEATURE_COLUMNS, WINDOW_1_FEATURE_ENGINEERS, WINDOW_1_FEATURE_NUM


EPS = 1e-12

BASE_FEATURE_COLUMNS = [
    'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
]

DAILY_FEATURE_COLUMNS = [
    'WF_RET_1',
    'WF_GAP',
    'WF_BODY',
    'WF_RANGE',
    'WF_CLV',
    'WF_VWAP_DIST',
    'WF_LOG_AMOUNT',
]

ROLLING_FEATURE_TEMPLATES = [
    'WF_RET_{w}',
    'WF_REV_{w}',
    'WF_RET_MEAN_{w}',
    'WF_RET_IR_{w}',
    'WF_VOL_{w}',
    'WF_DOWNSIDE_VOL_{w}',
    'WF_UP_RATE_{w}',
    'WF_TURN_MEAN_{w}',
    'WF_TURN_Z_{w}',
    'WF_AMOUNT_MEAN_{w}',
    'WF_VOLUME_RATIO_{w}',
    'WF_RANGE_MEAN_{w}',
    'WF_CLV_MEAN_{w}',
    'WF_GAP_MEAN_{w}',
    'WF_HIGH_DIST_{w}',
    'WF_LOW_DIST_{w}',
    'WF_RSV_{w}',
]

WINDOW_SPECS = {
    # 每个输入窗口只绑定自己的自然日频窗口，实验只比较窗口和因子本身。
    'window_01': {'input_weeks': 1, 'windows': (2, 3, 5)},
    'window_02': {'input_weeks': 2, 'windows': (3, 5, 10)},
    'window_04': {'input_weeks': 4, 'windows': (5, 10, 20)},
    'window_08': {'input_weeks': 8, 'windows': (10, 20, 40)},
    'window_12': {'input_weeks': 12, 'windows': (20, 40, 60)},
}

STATIC_FEATURE_TYPES = {'39', '158+39'}
EXTRA_WINDOW_FEATURE_TYPES = {WINDOW_1_FEATURE_NUM}


def window_feature_columns(feature_num: str) -> list[str]:
    """按声明式窗口配置生成特征列名。"""
    columns = [*BASE_FEATURE_COLUMNS, *DAILY_FEATURE_COLUMNS]
    for w in WINDOW_SPECS[feature_num]['windows']:
        columns.extend(template.format(w=w) for template in ROLLING_FEATURE_TEMPLATES)
    return columns


def engineer_window_features(df: pd.DataFrame, feature_num: str) -> pd.DataFrame:
    """按短窗口配置计算日频滚动因子。"""
    df = df.copy()
    open_ = df['开盘'].astype(float)
    high = df['最高'].astype(float)
    low = df['最低'].astype(float)
    close = df['收盘'].astype(float)
    volume = df['成交量'].astype(float)
    amount = df['成交额'].astype(float)
    turnover = df['换手率'].astype(float)

    prev_close = close.shift(1)
    ret_1 = close.pct_change(1, fill_method=None)
    high_low = high - low
    vwap = amount / (volume + EPS)
    body = (close - open_) / (open_ + EPS)
    range_ = high_low / (open_ + EPS)
    gap = open_ / (prev_close + EPS) - 1.0
    clv = (2.0 * close - high - low) / (high_low + EPS)
    log_amount = np.log1p(amount)
    downside = ret_1.mask(ret_1 > 0.0, 0.0)

    features = {
        # 单日状态保留价格压力和量能状态，供每个窗口共用。
        'WF_RET_1': ret_1,
        'WF_GAP': gap,
        'WF_BODY': body,
        'WF_RANGE': range_,
        'WF_CLV': clv,
        'WF_VWAP_DIST': close / (vwap + EPS) - 1.0,
        'WF_LOG_AMOUNT': log_amount,
    }

    for w in WINDOW_SPECS[feature_num]['windows']:
        ret_w = close.pct_change(w, fill_method=None)
        ret_mean = ret_1.rolling(w).mean()
        ret_std = ret_1.rolling(w).std()
        turn_mean = turnover.rolling(w).mean()
        turn_std = turnover.rolling(w).std()
        volume_mean = volume.rolling(w).mean()
        rolling_high = high.rolling(w).max()
        rolling_low = low.rolling(w).min()

        # 短周期只保留反转、强弱、波动、流动性和位置五类信息。
        features[f'WF_RET_{w}'] = ret_w
        features[f'WF_REV_{w}'] = -ret_w
        features[f'WF_RET_MEAN_{w}'] = ret_mean
        features[f'WF_RET_IR_{w}'] = ret_mean / (ret_std + EPS)
        features[f'WF_VOL_{w}'] = ret_std
        features[f'WF_DOWNSIDE_VOL_{w}'] = downside.rolling(w).std()
        features[f'WF_UP_RATE_{w}'] = (ret_1 > 0.0).rolling(w).mean()
        features[f'WF_TURN_MEAN_{w}'] = turn_mean
        features[f'WF_TURN_Z_{w}'] = (turnover - turn_mean) / (turn_std + EPS)
        features[f'WF_AMOUNT_MEAN_{w}'] = log_amount.rolling(w).mean()
        features[f'WF_VOLUME_RATIO_{w}'] = volume / (volume_mean + EPS)
        features[f'WF_RANGE_MEAN_{w}'] = range_.rolling(w).mean()
        features[f'WF_CLV_MEAN_{w}'] = clv.rolling(w).mean()
        features[f'WF_GAP_MEAN_{w}'] = gap.rolling(w).mean()
        features[f'WF_HIGH_DIST_{w}'] = close / (rolling_high + EPS) - 1.0
        features[f'WF_LOW_DIST_{w}'] = close / (rolling_low + EPS) - 1.0
        features[f'WF_RSV_{w}'] = (close - rolling_low) / (rolling_high - rolling_low + EPS)

    feature_df = pd.DataFrame(features, index=df.index)
    return pd.concat([df, feature_df], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def window_input_weeks(feature_num: str) -> int:
    """返回窗口因子所需历史周数。"""
    return int(WINDOW_SPECS[feature_num]['input_weeks'])


def bound_window_feature(input_window: int) -> str:
    """把抽象 window 特征绑定到具体 input_window。"""
    window_feature_num = f'window_{int(input_window):02d}'
    if window_feature_num not in WINDOW_SPECS:
        raise ValueError(f'unsupported input_window for short window factors: {input_window}')
    return window_feature_num


def assert_input_window(input_window: int, feature_num: str) -> None:
    """显式窗口名必须和模型输入窗口一致。"""
    expected_window = window_input_weeks(feature_num)
    if int(input_window) != expected_window:
        raise ValueError(f'{feature_num} requires input_window={expected_window}')


def feature_num_for_window(input_window: int, feature_type: str) -> str:
    """把模型配置里的 input_window 和 feature_type 解析成特征配置名。"""
    if feature_type in STATIC_FEATURE_TYPES:
        return feature_type

    if feature_type in EXTRA_WINDOW_FEATURE_TYPES:
        if int(input_window) != 1:
            raise ValueError(f'{feature_type} requires input_window=1')
        return feature_type

    if feature_type == 'window':
        return bound_window_feature(input_window)

    if feature_type.endswith('+xsec'):
        base_feature_type = feature_type.removesuffix('+xsec')
        if base_feature_type in STATIC_FEATURE_TYPES:
            return feature_type
        if base_feature_type in EXTRA_WINDOW_FEATURE_TYPES:
            if int(input_window) != 1:
                raise ValueError(f'{feature_type} requires input_window=1')
            return feature_type
        if base_feature_type == 'window':
            return f'{bound_window_feature(input_window)}+xsec'
        if base_feature_type in WINDOW_SPECS:
            assert_input_window(input_window, base_feature_type)
            return feature_type

    if feature_type in WINDOW_SPECS:
        assert_input_window(input_window, feature_type)
        return feature_type

    raise ValueError(f'unsupported feature_type: {feature_type}')


WINDOW_FEATURE_COLUMNS = {
    feature_num: window_feature_columns(feature_num)
    for feature_num in WINDOW_SPECS
}
WINDOW_FEATURE_COLUMNS.update(WINDOW_1_FEATURE_COLUMNS)

WINDOW_FEATURE_ENGINEERS = {
    # `baseline.py` 需要单参数函数，这里绑定 feature_num。
    feature_num: partial(engineer_window_features, feature_num=feature_num)
    for feature_num in WINDOW_SPECS
}
WINDOW_FEATURE_ENGINEERS.update(WINDOW_1_FEATURE_ENGINEERS)
