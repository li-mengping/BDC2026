"""窗口因子工程；通过 `WINDOW_FEATURE_ENGINEERS` 按 feature_num 生成日频特征。"""

from functools import partial

import numpy as np
import pandas as pd
import talib


# 窗口因子只看目标周之前的历史日线，避免未来信息泄漏。
EPS = 1e-12

BASE_FEATURE_COLUMNS = [
    'instrument',
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
]

DAILY_FEATURE_COLUMNS = [
    'WF_CLV',
    'WF_BODY',
    'WF_RANGE',
    'WF_UPPER_SHADOW',
    'WF_LOWER_SHADOW',
    'WF_GAP',
    'WF_VWAP_DIST',
    'WF_LOG_AMOUNT',
]

FAMILY_COLUMNS = {
    'short_reversal': (
        'WF_REV_{w}',
        'WF_LAST_RET_{w}',
        'WF_UP_DAYS_{w}',
        'WF_DOWN_DAYS_{w}',
    ),
    'extreme': (
        'WF_MAXRET_{w}',
        'WF_MINRET_{w}',
        'WF_RET_SKEW_{w}',
        'WF_TAIL_RATIO_{w}',
    ),
    'momentum': (
        'WF_RET_{w}',
        'WF_RET_MEAN_{w}',
        'WF_RET_IR_{w}',
    ),
    'slow_momentum': (
        'WF_MOM_SKIP5_{w}',
        'WF_MOM_STABLE_{w}',
        'WF_POS_RET_RATIO_{w}',
    ),
    'quality_momentum': (
        'WF_MOM_EFF_{w}',
        'WF_MOM_VOLADJ_{w}',
        'WF_MOM_TURNADJ_{w}',
        'WF_MOM_MAXADJ_{w}',
    ),
    'acceleration': (
        'WF_RET_ACCEL_{w}',
        'WF_TURN_ACCEL_{w}',
    ),
    'trend': (
        'WF_MA_DIST_{w}',
        'WF_SLOPE_{w}',
        'WF_TREND_R2_{w}',
    ),
    'risk': (
        'WF_VOL_{w}',
        'WF_DOWNSIDE_VOL_{w}',
        'WF_MDD_{w}',
        'WF_AMP_MEAN_{w}',
    ),
    'liquidity': (
        'WF_TURN_MEAN_{w}',
        'WF_TURN_STD_{w}',
        'WF_AMOUNT_MEAN_{w}',
        'WF_ILLIQ_{w}',
        'WF_VOLUME_RATIO_{w}',
    ),
    'volume_price': (
        'WF_UP_VOLUME_RATIO_{w}',
        'WF_RET_VOLUME_CORR_{w}',
        'WF_RET_TURN_CORR_{w}',
    ),
    'breakout': (
        'WF_HIGH_DIST_{w}',
        'WF_LOW_DIST_{w}',
        'WF_RSV_{w}',
        'WF_REBOUND_{w}',
    ),
    'price_pressure': (
        'WF_CLV_MEAN_{w}',
        'WF_BODY_MEAN_{w}',
        'WF_GAP_MEAN_{w}',
        'WF_RANGE_RATIO_{w}',
    ),
}

WINDOW_SPECS = {
    'window_01': {
        'input_weeks': 1,
        'windows': (2, 3, 5),
        'families': ('short_reversal', 'extreme', 'risk', 'liquidity', 'volume_price'),
    },
    'window_02': {
        'input_weeks': 2,
        'windows': (3, 5, 10),
        'families': ('short_reversal', 'extreme', 'momentum', 'acceleration', 'risk', 'liquidity', 'volume_price'),
    },
    'window_04': {
        'input_weeks': 4,
        'windows': (5, 10, 20),
        'families': ('short_reversal', 'extreme', 'momentum', 'quality_momentum', 'risk', 'liquidity', 'volume_price', 'breakout'),
    },
    'window_08': {
        'input_weeks': 8,
        'windows': (10, 20, 40),
        'families': ('momentum', 'slow_momentum', 'quality_momentum', 'acceleration', 'trend', 'risk', 'liquidity', 'volume_price', 'breakout'),
    },
    'window_12': {
        'input_weeks': 12,
        'windows': (20, 40, 60),
        'families': ('momentum', 'slow_momentum', 'quality_momentum', 'trend', 'risk', 'liquidity', 'volume_price', 'breakout'),
    },
    'window_focus': {
        'input_weeks': 12,
        'windows': (5, 10, 20, 40, 60),
        'families': ('momentum', 'quality_momentum', 'trend', 'risk', 'liquidity', 'breakout', 'price_pressure'),
    },
    'window_cycle': {
        'input_weeks': 12,
        'windows': (5, 15, 30, 60),
        'families': ('momentum', 'quality_momentum', 'trend', 'risk', 'liquidity', 'breakout', 'price_pressure'),
    },
}

COMPOSED_WINDOW_SPECS = {
    'window_multi': {
        'sources': ('window_01', 'window_02', 'window_04', 'window_08', 'window_12'),
        'cross': False,
    },
    'window_multi_cross': {
        'sources': ('window_01', 'window_02', 'window_04', 'window_08', 'window_12'),
        'cross': True,
    },
}

CROSS_WINDOW_SPECS = (
    ('CX_RET_DIFF_5_20', 'diff', 'WF_RET_5', 'WF_RET_20'),
    ('CX_RET_DIFF_10_40', 'diff', 'WF_RET_10', 'WF_RET_40'),
    ('CX_RET_DIFF_20_60', 'diff', 'WF_RET_20', 'WF_RET_60'),
    ('CX_IR_DIFF_5_20', 'diff', 'WF_RET_IR_5', 'WF_RET_IR_20'),
    ('CX_IR_DIFF_20_60', 'diff', 'WF_RET_IR_20', 'WF_RET_IR_60'),
    ('CX_VOL_RATIO_5_20', 'ratio', 'WF_VOL_5', 'WF_VOL_20'),
    ('CX_VOL_RATIO_20_60', 'ratio', 'WF_VOL_20', 'WF_VOL_60'),
    ('CX_RET_VOL_5_20', 'ratio', 'WF_RET_5', 'WF_VOL_20'),
    ('CX_RET_VOL_20_60', 'ratio', 'WF_RET_20', 'WF_VOL_60'),
    ('CX_REV_RET_3_10', 'product', 'WF_REV_3', 'WF_RET_10'),
    ('CX_REV_RET_5_20', 'product', 'WF_REV_5', 'WF_RET_20'),
    ('CX_RSV_RET_20', 'product', 'WF_RSV_20', 'WF_RET_20'),
    ('CX_RSV_RET_60', 'product', 'WF_RSV_60', 'WF_RET_60'),
    ('CX_TREND_RET_40', 'product', 'WF_TREND_R2_40', 'WF_RET_40'),
    ('CX_TREND_RET_60', 'product', 'WF_TREND_R2_60', 'WF_RET_60'),
    ('CX_SLOPE_R2_40', 'product', 'WF_SLOPE_40', 'WF_TREND_R2_40'),
    ('CX_SLOPE_R2_60', 'product', 'WF_SLOPE_60', 'WF_TREND_R2_60'),
    ('CX_RET_TURN_ACCEL_20', 'product', 'WF_RET_20', 'WF_TURN_ACCEL_20'),
    ('CX_RET_TURN_ACCEL_40', 'product', 'WF_RET_40', 'WF_TURN_ACCEL_40'),
    ('CX_RET_ILLIQ_20', 'product', 'WF_RET_20', 'WF_ILLIQ_20'),
    ('CX_RET_ILLIQ_60', 'product', 'WF_RET_60', 'WF_ILLIQ_60'),
    ('CX_MDD_VOL_20', 'product', 'WF_MDD_20', 'WF_VOL_20'),
    ('CX_MDD_VOL_60', 'product', 'WF_MDD_60', 'WF_VOL_60'),
)
CROSS_WINDOW_COLUMNS = [name for name, _, _, _ in CROSS_WINDOW_SPECS]

WINDOW_BOUND_FEATURE_TYPES = {
    'window': '{window_feature_num}',
    '39+window': '39+{window_feature_num}',
    '158+39+window': '158+39+{window_feature_num}',
}

STATIC_FEATURE_TYPES = {
    '39',
    '158+39',
}


def rolling_time_rsqr(values: np.ndarray) -> float:
    """滚动窗口内价格和时间趋势相关性的 R^2。"""
    y = np.asarray(values, dtype=np.float64)
    if len(y) < 2 or np.isnan(y).any():
        return np.nan
    x = np.arange(len(y), dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
    if denom <= EPS:
        return np.nan
    corr = float(np.sum(x * y) / denom)
    return corr * corr


def dedupe_columns(columns: list[str]) -> list[str]:
    """按首次出现顺序去重。"""
    result = []
    seen = set()
    for column in columns:
        if column not in seen:
            result.append(column)
            seen.add(column)
    return result


def rolling_windows(feature_num: str) -> tuple[int, ...]:
    """返回输入窗口内可观察的日频窗口。"""
    spec = WINDOW_SPECS[feature_num]
    max_days = int(spec['input_weeks']) * 5
    return tuple(w for w in spec['windows'] if 1 < w <= max_days)


def feature_columns_for_spec(feature_num: str) -> list[str]:
    """返回单个窗口配置的特征列。"""
    spec = WINDOW_SPECS[feature_num]
    columns = list(BASE_FEATURE_COLUMNS) + list(DAILY_FEATURE_COLUMNS)
    for w in rolling_windows(feature_num):
        for family in spec['families']:
            columns.extend(name.format(w=w) for name in FAMILY_COLUMNS[family])
    return columns


def feature_columns_for_composed(feature_num: str) -> list[str]:
    """返回多个窗口配置拼接后的特征列。"""
    spec = COMPOSED_WINDOW_SPECS[feature_num]
    columns = list(BASE_FEATURE_COLUMNS) + list(DAILY_FEATURE_COLUMNS)
    for source in spec['sources']:
        columns.extend(
            column
            for column in feature_columns_for_spec(source)
            if column not in BASE_FEATURE_COLUMNS and column not in DAILY_FEATURE_COLUMNS
        )
    if spec['cross']:
        columns.extend(CROSS_WINDOW_COLUMNS)
    return dedupe_columns(columns)


def rolling_feature_columns(feature_num: str) -> list[str]:
    """统一返回窗口特征列。"""
    if feature_num in WINDOW_SPECS:
        return feature_columns_for_spec(feature_num)
    return feature_columns_for_composed(feature_num)


def engineer_window_spec(df: pd.DataFrame, feature_num: str) -> pd.DataFrame:
    """计算单只股票的窗口自适应因子。"""
    spec = WINDOW_SPECS[feature_num]
    df = df.copy()
    open_ = df['开盘'].astype(float)
    high = df['最高'].astype(float)
    low = df['最低'].astype(float)
    close = df['收盘'].astype(float)
    volume = df['成交量'].astype(float)
    amount = df['成交额'].astype(float)
    turnover = df['换手率'].astype(float)
    amplitude = df['振幅'].astype(float)

    prev_close = close.shift(1)
    ret_1 = close.pct_change(1, fill_method=None)
    volume_change = volume.pct_change(1, fill_method=None)
    vwap = amount / (volume + EPS)
    high_low = high - low
    body_high = pd.concat([open_, close], axis=1).max(axis=1)
    body_low = pd.concat([open_, close], axis=1).min(axis=1)
    clv = (2.0 * close - high - low) / (high_low + EPS)
    body = (close - open_) / (open_ + EPS)
    range_ = high_low / (open_ + EPS)
    gap = open_ / (prev_close + EPS) - 1.0
    log_amount = np.log1p(amount)

    features: dict[str, pd.Series] = {
        # 日内形态和价格压力，作为最后交易日状态特征。
        'WF_CLV': clv,
        'WF_BODY': body,
        'WF_RANGE': range_,
        'WF_UPPER_SHADOW': (high - body_high) / (open_ + EPS),
        'WF_LOWER_SHADOW': (body_low - low) / (open_ + EPS),
        'WF_GAP': gap,
        'WF_VWAP_DIST': close / (vwap + EPS) - 1.0,
        'WF_LOG_AMOUNT': log_amount,
    }

    for w in rolling_windows(feature_num):
        families = spec['families']
        ret_w = close.pct_change(w, fill_method=None)
        ret_mean = ret_1.rolling(w).mean()
        ret_std = ret_1.rolling(w).std()
        max_ret = ret_1.rolling(w).max()
        min_ret = ret_1.rolling(w).min()
        rolling_high = high.rolling(w).max()
        rolling_low = low.rolling(w).min()
        close_high = close.rolling(w).max()
        turn_mean = turnover.rolling(w).mean()
        volume_sum = volume.rolling(w).sum()

        if 'short_reversal' in families:
            features[f'WF_REV_{w}'] = -ret_w
            features[f'WF_LAST_RET_{w}'] = ret_1.rolling(w).sum()
            features[f'WF_UP_DAYS_{w}'] = (ret_1 > 0.0).rolling(w).mean()
            features[f'WF_DOWN_DAYS_{w}'] = (ret_1 < 0.0).rolling(w).mean()

        if 'extreme' in families:
            features[f'WF_MAXRET_{w}'] = max_ret
            features[f'WF_MINRET_{w}'] = min_ret
            features[f'WF_RET_SKEW_{w}'] = ret_1.rolling(w).skew()
            features[f'WF_TAIL_RATIO_{w}'] = max_ret / (min_ret.abs() + EPS)

        if 'momentum' in families:
            features[f'WF_RET_{w}'] = ret_w
            features[f'WF_RET_MEAN_{w}'] = ret_mean
            features[f'WF_RET_IR_{w}'] = ret_mean / (ret_std + EPS)

        if 'slow_momentum' in families:
            skip = 5
            half = max(skip, w // 2)
            recent = close.shift(skip) / (close.shift(half) + EPS) - 1.0
            older = close.shift(half) / (close.shift(w) + EPS) - 1.0
            features[f'WF_MOM_SKIP5_{w}'] = close.shift(skip) / (close.shift(w) + EPS) - 1.0
            features[f'WF_MOM_STABLE_{w}'] = recent - older
            features[f'WF_POS_RET_RATIO_{w}'] = (ret_1 > 0.0).rolling(w).mean()

        if 'quality_momentum' in families:
            ret_path = ret_1.abs().rolling(w).sum()
            turn_cv = turnover.rolling(w).std() / (turn_mean + EPS)
            features[f'WF_MOM_EFF_{w}'] = ret_w / (ret_path + EPS)
            features[f'WF_MOM_VOLADJ_{w}'] = ret_w / (ret_std * np.sqrt(w) + EPS)
            features[f'WF_MOM_TURNADJ_{w}'] = ret_w / (1.0 + turn_cv)
            features[f'WF_MOM_MAXADJ_{w}'] = ret_w - max_ret.clip(lower=0.0)

        if 'acceleration' in families:
            half = max(2, w // 2)
            recent_ret = close / (close.shift(half) + EPS) - 1.0
            prior_ret = close.shift(half) / (close.shift(w) + EPS) - 1.0
            recent_turn = turnover.rolling(half).mean()
            prior_turn = turnover.shift(half).rolling(half).mean()
            features[f'WF_RET_ACCEL_{w}'] = recent_ret - prior_ret
            features[f'WF_TURN_ACCEL_{w}'] = recent_turn / (prior_turn + EPS) - 1.0

        if 'trend' in families:
            slope = talib.LINEARREG_SLOPE(close, timeperiod=w)
            features[f'WF_MA_DIST_{w}'] = close / (talib.SMA(close, timeperiod=w) + EPS) - 1.0
            features[f'WF_SLOPE_{w}'] = slope / (close + EPS)
            features[f'WF_TREND_R2_{w}'] = close.rolling(w).apply(rolling_time_rsqr, raw=True)

        if 'risk' in families:
            downside = ret_1.mask(ret_1 > 0.0, 0.0)
            features[f'WF_VOL_{w}'] = ret_std
            features[f'WF_DOWNSIDE_VOL_{w}'] = downside.rolling(w).std()
            features[f'WF_MDD_{w}'] = close / (close_high + EPS) - 1.0
            features[f'WF_AMP_MEAN_{w}'] = amplitude.rolling(w).mean()

        if 'liquidity' in families:
            amount_unit = amount / 1e8
            volume_mean = volume.rolling(w).mean()
            features[f'WF_TURN_MEAN_{w}'] = turn_mean
            features[f'WF_TURN_STD_{w}'] = turnover.rolling(w).std()
            features[f'WF_AMOUNT_MEAN_{w}'] = log_amount.rolling(w).mean()
            features[f'WF_ILLIQ_{w}'] = (ret_1.abs() / (amount_unit + EPS)).rolling(w).mean()
            features[f'WF_VOLUME_RATIO_{w}'] = volume / (volume_mean + EPS)

        if 'volume_price' in families:
            up_volume = volume.where(ret_1 > 0.0, 0.0)
            features[f'WF_UP_VOLUME_RATIO_{w}'] = up_volume.rolling(w).sum() / (volume_sum + EPS)
            features[f'WF_RET_VOLUME_CORR_{w}'] = ret_1.rolling(w).corr(volume_change)
            features[f'WF_RET_TURN_CORR_{w}'] = ret_1.rolling(w).corr(turnover)

        if 'breakout' in families:
            features[f'WF_HIGH_DIST_{w}'] = close / (rolling_high + EPS) - 1.0
            features[f'WF_LOW_DIST_{w}'] = close / (rolling_low + EPS) - 1.0
            features[f'WF_RSV_{w}'] = (close - rolling_low) / (rolling_high - rolling_low + EPS)
            features[f'WF_REBOUND_{w}'] = close / (close.rolling(w).min() + EPS) - 1.0

        if 'price_pressure' in families:
            features[f'WF_CLV_MEAN_{w}'] = clv.rolling(w).mean()
            features[f'WF_BODY_MEAN_{w}'] = body.rolling(w).mean()
            features[f'WF_GAP_MEAN_{w}'] = gap.rolling(w).mean()
            features[f'WF_RANGE_RATIO_{w}'] = range_ / (range_.rolling(w).mean() + EPS)

    feature_df = pd.DataFrame(features, index=df.index)
    return pd.concat([df, feature_df], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_cross_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """生成少量跨窗口交互，保留现有 baseline 的可比较性。"""
    missing = sorted({col for _, _, left, right in CROSS_WINDOW_SPECS for col in (left, right) if col not in df.columns})
    if missing:
        raise ValueError(f'missing cross feature inputs: {missing[:10]}')

    features = {}
    for name, op, left_col, right_col in CROSS_WINDOW_SPECS:
        left = df[left_col].astype(float)
        right = df[right_col].astype(float)
        if op == 'diff':
            features[name] = left - right
        elif op == 'ratio':
            features[name] = left / (right.abs() + EPS)
        elif op == 'product':
            features[name] = left * right
        else:
            raise ValueError(f'unsupported cross feature op: {op}')

    feature_df = pd.DataFrame(features, index=df.index)
    return pd.concat([df, feature_df], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def engineer_composed_window_features(df: pd.DataFrame, feature_num: str) -> pd.DataFrame:
    """依次计算多个窗口配置，去重后拼接。"""
    frames = [df.copy()]
    seen = set(df.columns)
    spec = COMPOSED_WINDOW_SPECS[feature_num]
    for source in spec['sources']:
        engineered = engineer_window_spec(df, source)
        new_columns = [
            column
            for column in feature_columns_for_spec(source)
            if column != 'instrument' and column in engineered.columns and column not in seen
        ]
        if new_columns:
            frames.append(engineered[new_columns])
            seen.update(new_columns)

    result = pd.concat(frames, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if spec['cross']:
        return add_cross_window_features(result)
    return result


def engineer_window_features(df: pd.DataFrame, feature_num: str) -> pd.DataFrame:
    """按特征配置名调用对应窗口因子。"""
    if feature_num in WINDOW_SPECS:
        return engineer_window_spec(df, feature_num)
    if feature_num in COMPOSED_WINDOW_SPECS:
        return engineer_composed_window_features(df, feature_num)
    raise ValueError(f'unsupported window feature_num: {feature_num}')


def window_input_weeks(feature_num: str) -> int:
    """返回窗口特征需要的输入周数。"""
    if feature_num in WINDOW_SPECS:
        return int(WINDOW_SPECS[feature_num]['input_weeks'])
    spec = COMPOSED_WINDOW_SPECS[feature_num]
    return max(window_input_weeks(source) for source in spec['sources'])


def split_window_feature_type(feature_type: str) -> tuple[str, str] | None:
    """从 feature_type 中解析窗口特征名。"""
    if feature_type in WINDOW_SPECS or feature_type in COMPOSED_WINDOW_SPECS:
        return feature_type, feature_type
    for prefix in ('39+', '158+39+'):
        if feature_type.startswith(prefix):
            window_feature_num = feature_type.removeprefix(prefix)
            if window_feature_num in WINDOW_SPECS or window_feature_num in COMPOSED_WINDOW_SPECS:
                return feature_type, window_feature_num
    return None


def feature_num_for_window(input_window: int, feature_type: str) -> str:
    """把模型配置里的输入周数和特征类型绑定为特征配置名。"""
    window = int(input_window)
    suffix = '+xsec' if feature_type.endswith('+xsec') else ''
    base_type = feature_type.removesuffix('+xsec')

    if base_type in WINDOW_BOUND_FEATURE_TYPES:
        window_feature_num = f'window_{window:02d}'
        if window_feature_num not in WINDOW_SPECS:
            raise ValueError(f'unsupported input_window for window factors: {input_window}')
        return WINDOW_BOUND_FEATURE_TYPES[base_type].format(window_feature_num=window_feature_num) + suffix

    parsed = split_window_feature_type(base_type)
    if parsed:
        resolved_type, window_feature_num = parsed
        expected_window = window_input_weeks(window_feature_num)
        if window != expected_window:
            raise ValueError(f'{feature_type} requires input_window={expected_window}')
        return resolved_type + suffix

    if base_type in STATIC_FEATURE_TYPES:
        return base_type + suffix

    raise ValueError(f'unsupported feature_type: {feature_type}')


WINDOW_FEATURE_NAMES = (*WINDOW_SPECS.keys(), *COMPOSED_WINDOW_SPECS.keys())

WINDOW_FEATURE_COLUMNS = {
    feature_num: rolling_feature_columns(feature_num)
    for feature_num in WINDOW_FEATURE_NAMES
}

WINDOW_FEATURE_ENGINEERS = {
    # 绑定 feature_num，供 baseline.py 按配置直接调用对应窗口因子。
    feature_num: partial(engineer_window_features, feature_num=feature_num)
    for feature_num in WINDOW_FEATURE_NAMES
}
