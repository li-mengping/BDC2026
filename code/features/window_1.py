"""一周输入专用因子；所有特征只依赖最近 5 个交易日。"""

from collections.abc import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12
WINDOW_1_FEATURE_NUM = 'window_1'
WINDOW_1_WINDOWS = (2, 3, 5)

BASE_FEATURE_COLUMNS = [
    'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
]

DAILY_FEATURE_COLUMNS = [
    'W1_RET_1',
    'W1_REV_1',
    'W1_GAP',
    'W1_INTRADAY',
    'W1_BODY',
    'W1_BODY_ABS',
    'W1_RANGE',
    'W1_TRUE_RANGE',
    'W1_UPPER_SHADOW',
    'W1_LOWER_SHADOW',
    'W1_SHADOW_BALANCE',
    'W1_CLV',
    'W1_VWAP_DIST',
    'W1_TURN',
    'W1_LOG_AMOUNT',
    'W1_AMOUNT_CHG',
    'W1_VOLUME_CHG',
    'W1_SIGNED_TURN',
    'W1_AMIHUD',
    'W1_PARKINSON',
    'W1_GARMAN_KLASS',
    'W1_ROGERS_SATCHELL',
    'W1_GAP_FOLLOW',
]

ROLLING_FEATURE_TEMPLATES = [
    'W1_RET_{w}',
    'W1_REV_{w}',
    'W1_RET_MEAN_{w}',
    'W1_RET_STD_{w}',
    'W1_RET_IR_{w}',
    'W1_RET_MIN_{w}',
    'W1_RET_MAX_{w}',
    'W1_RET_SKEW_{w}',
    'W1_UP_RATE_{w}',
    'W1_DOWN_RATE_{w}',
    'W1_MAX_ABS_RET_{w}',
    'W1_MAX_RET_SHARE_{w}',
    'W1_CLOSE_MA_DIST_{w}',
    'W1_TREND_SLOPE_{w}',
    'W1_TREND_R2_{w}',
    'W1_EFFICIENCY_{w}',
    'W1_CHOPPINESS_{w}',
    'W1_HIGH_DIST_{w}',
    'W1_LOW_DIST_{w}',
    'W1_RSV_{w}',
    'W1_DRAWDOWN_{w}',
    'W1_RANGE_MEAN_{w}',
    'W1_TRUE_RANGE_MEAN_{w}',
    'W1_PARKINSON_MEAN_{w}',
    'W1_GARMAN_KLASS_MEAN_{w}',
    'W1_DOWNSIDE_VOL_{w}',
    'W1_UPSIDE_VOL_{w}',
    'W1_VOL_ASYM_{w}',
    'W1_TURN_MEAN_{w}',
    'W1_TURN_Z_{w}',
    'W1_AMOUNT_MEAN_{w}',
    'W1_AMOUNT_Z_{w}',
    'W1_VOLUME_RATIO_{w}',
    'W1_AMIHUD_MEAN_{w}',
    'W1_AMIHUD_Z_{w}',
    'W1_SIGNED_TURN_SUM_{w}',
    'W1_SIGNED_AMOUNT_SUM_{w}',
    'W1_MONEY_FLOW_{w}',
    'W1_RET_TURN_CORR_{w}',
    'W1_RET_AMOUNT_CORR_{w}',
    'W1_BODY_MEAN_{w}',
    'W1_BODY_ABS_MEAN_{w}',
    'W1_CLV_MEAN_{w}',
    'W1_GAP_MEAN_{w}',
    'W1_GAP_ABS_MEAN_{w}',
    'W1_GAP_FOLLOW_MEAN_{w}',
    'W1_SHADOW_BALANCE_MEAN_{w}',
    'W1_LAST_RET_RANK_{w}',
    'W1_LAST_TURN_RANK_{w}',
]

WINDOW_1_FEATURE_GROUPS = {
    'base': tuple(BASE_FEATURE_COLUMNS),
    'daily_state': tuple(DAILY_FEATURE_COLUMNS),
    'reversal': (
        'W1_RET_1', 'W1_REV_1', 'W1_RET_{w}', 'W1_REV_{w}', 'W1_RET_MIN_{w}', 'W1_RET_MAX_{w}',
        'W1_LAST_RET_RANK_{w}',
    ),
    'trend_efficiency': (
        'W1_RET_1', 'W1_RET_{w}', 'W1_CLOSE_MA_DIST_{w}', 'W1_TREND_SLOPE_{w}', 'W1_TREND_R2_{w}',
        'W1_EFFICIENCY_{w}', 'W1_CHOPPINESS_{w}', 'W1_UP_RATE_{w}', 'W1_DOWN_RATE_{w}',
    ),
    'range_volatility': (
        'W1_RANGE', 'W1_TRUE_RANGE', 'W1_PARKINSON', 'W1_GARMAN_KLASS', 'W1_ROGERS_SATCHELL',
        'W1_RET_STD_{w}', 'W1_RANGE_MEAN_{w}', 'W1_TRUE_RANGE_MEAN_{w}', 'W1_PARKINSON_MEAN_{w}',
        'W1_GARMAN_KLASS_MEAN_{w}', 'W1_DOWNSIDE_VOL_{w}', 'W1_UPSIDE_VOL_{w}', 'W1_VOL_ASYM_{w}',
    ),
    'liquidity_impact': (
        'W1_TURN', 'W1_LOG_AMOUNT', 'W1_AMOUNT_CHG', 'W1_VOLUME_CHG', 'W1_AMIHUD',
        'W1_TURN_MEAN_{w}', 'W1_TURN_Z_{w}', 'W1_AMOUNT_MEAN_{w}', 'W1_AMOUNT_Z_{w}',
        'W1_VOLUME_RATIO_{w}', 'W1_AMIHUD_MEAN_{w}', 'W1_AMIHUD_Z_{w}', 'W1_LAST_TURN_RANK_{w}',
    ),
    'money_flow': (
        'W1_CLV', 'W1_VWAP_DIST', 'W1_SIGNED_TURN', 'W1_SIGNED_TURN_SUM_{w}',
        'W1_SIGNED_AMOUNT_SUM_{w}', 'W1_MONEY_FLOW_{w}', 'W1_RET_TURN_CORR_{w}', 'W1_RET_AMOUNT_CORR_{w}',
    ),
    'position_breakout': (
        'W1_CLV', 'W1_VWAP_DIST', 'W1_HIGH_DIST_{w}', 'W1_LOW_DIST_{w}', 'W1_RSV_{w}',
        'W1_DRAWDOWN_{w}', 'W1_CLV_MEAN_{w}',
    ),
    'lottery_tail': (
        'W1_RET_MAX_{w}', 'W1_RET_MIN_{w}', 'W1_MAX_ABS_RET_{w}', 'W1_MAX_RET_SHARE_{w}',
        'W1_RET_SKEW_{w}', 'W1_DRAWDOWN_{w}',
    ),
    'gap_candle': (
        'W1_GAP', 'W1_INTRADAY', 'W1_BODY', 'W1_BODY_ABS', 'W1_UPPER_SHADOW', 'W1_LOWER_SHADOW',
        'W1_SHADOW_BALANCE', 'W1_GAP_FOLLOW', 'W1_BODY_MEAN_{w}', 'W1_BODY_ABS_MEAN_{w}',
        'W1_GAP_MEAN_{w}', 'W1_GAP_ABS_MEAN_{w}', 'W1_GAP_FOLLOW_MEAN_{w}',
        'W1_SHADOW_BALANCE_MEAN_{w}',
    ),
}


def window_1_feature_columns() -> list[str]:
    """声明 window_1 模块全部特征列。"""
    columns = [*BASE_FEATURE_COLUMNS, *DAILY_FEATURE_COLUMNS]
    for window in WINDOW_1_WINDOWS:
        columns.extend(template.format(w=window) for template in ROLLING_FEATURE_TEMPLATES)
    return list(dict.fromkeys(columns))


WINDOW_1_FEATURE_COLUMNS = {
    WINDOW_1_FEATURE_NUM: window_1_feature_columns(),
}


def window_1_group_columns(group_name: str) -> list[str]:
    """返回某个一周因子族的原始列。"""
    if group_name == 'all_window_1':
        return WINDOW_1_FEATURE_COLUMNS[WINDOW_1_FEATURE_NUM]
    if group_name not in WINDOW_1_FEATURE_GROUPS:
        raise ValueError(f'unknown window_1 factor group: {group_name}')

    columns = list(BASE_FEATURE_COLUMNS)
    for template in WINDOW_1_FEATURE_GROUPS[group_name]:
        if '{w}' in template:
            columns.extend(template.format(w=window) for window in WINDOW_1_WINDOWS)
        else:
            columns.append(template)
    return [column for column in dict.fromkeys(columns) if column in WINDOW_1_FEATURE_COLUMNS[WINDOW_1_FEATURE_NUM]]


def safe_div(numerator, denominator):
    """带极小值保护的除法。"""
    return numerator / (denominator + EPS)


def rolling_rank_last(values: np.ndarray) -> float:
    """窗口最后一个值在窗口内的百分位 rank。"""
    if len(values) == 0 or np.isnan(values[-1]):
        return np.nan
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.nan
    return float((valid <= values[-1]).mean())


def rolling_slope(values: np.ndarray) -> float:
    """短窗口线性趋势斜率。"""
    y = np.asarray(values, dtype=np.float64)
    if len(y) < 2 or np.isnan(y).any():
        return np.nan
    x = np.arange(len(y), dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sum(x ** 2)
    if denom <= EPS:
        return np.nan
    return float(np.sum(x * y) / denom)


def rolling_r2(values: np.ndarray) -> float:
    """短窗口线性趋势解释度。"""
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


def signed_sum(signal: pd.Series, weight: pd.Series, window: int) -> pd.Series:
    """按收益方向聚合成交权重，近似买卖压力。"""
    signed = np.sign(signal).fillna(0.0) * weight
    return safe_div(signed.rolling(window).sum(), weight.abs().rolling(window).sum())


def rolling_corr(left: pd.Series, right: pd.Series, window: int) -> pd.Series:
    """小窗口相关系数。"""
    return left.rolling(window).corr(right).replace([np.inf, -np.inf], np.nan)


def engineer_window_1_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算一周输入专用因子。"""
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
    gap = open_ / (prev_close + EPS) - 1.0
    intraday = close / (open_ + EPS) - 1.0
    high_low = high - low
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    body = (close - open_) / (open_ + EPS)
    body_abs = body.abs()
    range_ = high_low / (open_ + EPS)
    upper_shadow = (high - pd.concat([open_, close], axis=1).max(axis=1)) / (open_ + EPS)
    lower_shadow = (pd.concat([open_, close], axis=1).min(axis=1) - low) / (open_ + EPS)
    shadow_balance = lower_shadow - upper_shadow
    clv = (2.0 * close - high - low) / (high_low + EPS)
    vwap = amount / (volume + EPS)
    vwap_dist = close / (vwap + EPS) - 1.0
    log_amount = np.log1p(amount)
    amount_chg = log_amount.diff()
    volume_chg = np.log1p(volume).diff()
    signed_turn = np.sign(ret_1).fillna(0.0) * turnover
    amihud = ret_1.abs() / (amount + EPS)
    log_hl = np.log((high + EPS) / (low + EPS))
    log_co = np.log((close + EPS) / (open_ + EPS))
    log_ho = np.log((high + EPS) / (open_ + EPS))
    log_lo = np.log((low + EPS) / (open_ + EPS))
    log_hc = np.log((high + EPS) / (close + EPS))
    log_lc = np.log((low + EPS) / (close + EPS))
    parkinson = log_hl ** 2 / (4.0 * np.log(2.0))
    garman_klass = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2
    rogers_satchell = log_ho * log_hc + log_lo * log_lc
    gap_follow = gap * intraday

    features = {
        'W1_RET_1': ret_1,
        'W1_REV_1': -ret_1,
        'W1_GAP': gap,
        'W1_INTRADAY': intraday,
        'W1_BODY': body,
        'W1_BODY_ABS': body_abs,
        'W1_RANGE': range_,
        'W1_TRUE_RANGE': safe_div(true_range, prev_close),
        'W1_UPPER_SHADOW': upper_shadow,
        'W1_LOWER_SHADOW': lower_shadow,
        'W1_SHADOW_BALANCE': shadow_balance,
        'W1_CLV': clv,
        'W1_VWAP_DIST': vwap_dist,
        'W1_TURN': turnover,
        'W1_LOG_AMOUNT': log_amount,
        'W1_AMOUNT_CHG': amount_chg,
        'W1_VOLUME_CHG': volume_chg,
        'W1_SIGNED_TURN': signed_turn,
        'W1_AMIHUD': amihud,
        'W1_PARKINSON': parkinson,
        'W1_GARMAN_KLASS': garman_klass.clip(lower=0.0),
        'W1_ROGERS_SATCHELL': rogers_satchell.clip(lower=0.0),
        'W1_GAP_FOLLOW': gap_follow,
    }

    downside = ret_1.mask(ret_1 > 0.0, 0.0)
    upside = ret_1.mask(ret_1 < 0.0, 0.0)
    abs_ret_sum = ret_1.abs()
    for window in WINDOW_1_WINDOWS:
        ret_w = close.pct_change(window, fill_method=None)
        ret_mean = ret_1.rolling(window).mean()
        ret_std = ret_1.rolling(window).std()
        ret_min = ret_1.rolling(window).min()
        ret_max = ret_1.rolling(window).max()
        close_ma = close.rolling(window).mean()
        trend_base = np.log(close + EPS)
        rolling_high = high.rolling(window).max()
        rolling_low = low.rolling(window).min()
        high_dist = close / (rolling_high + EPS) - 1.0
        low_dist = close / (rolling_low + EPS) - 1.0
        rsv = (close - rolling_low) / (rolling_high - rolling_low + EPS)
        amount_mean = log_amount.rolling(window).mean()
        amount_std = log_amount.rolling(window).std()
        turnover_mean = turnover.rolling(window).mean()
        turnover_std = turnover.rolling(window).std()
        volume_mean = volume.rolling(window).mean()
        amihud_mean = amihud.rolling(window).mean()
        amihud_std = amihud.rolling(window).std()
        path_length = ret_1.abs().rolling(window).sum()

        features[f'W1_RET_{window}'] = ret_w
        features[f'W1_REV_{window}'] = -ret_w
        features[f'W1_RET_MEAN_{window}'] = ret_mean
        features[f'W1_RET_STD_{window}'] = ret_std
        features[f'W1_RET_IR_{window}'] = ret_mean / (ret_std + EPS)
        features[f'W1_RET_MIN_{window}'] = ret_min
        features[f'W1_RET_MAX_{window}'] = ret_max
        features[f'W1_RET_SKEW_{window}'] = ret_1.rolling(window).skew()
        features[f'W1_UP_RATE_{window}'] = (ret_1 > 0.0).rolling(window).mean()
        features[f'W1_DOWN_RATE_{window}'] = (ret_1 < 0.0).rolling(window).mean()
        features[f'W1_MAX_ABS_RET_{window}'] = ret_1.abs().rolling(window).max()
        features[f'W1_MAX_RET_SHARE_{window}'] = ret_max / (abs_ret_sum.rolling(window).sum() + EPS)
        features[f'W1_CLOSE_MA_DIST_{window}'] = close / (close_ma + EPS) - 1.0
        features[f'W1_TREND_SLOPE_{window}'] = trend_base.rolling(window).apply(rolling_slope, raw=True)
        features[f'W1_TREND_R2_{window}'] = trend_base.rolling(window).apply(rolling_r2, raw=True)
        features[f'W1_EFFICIENCY_{window}'] = ret_w.abs() / (path_length + EPS)
        features[f'W1_CHOPPINESS_{window}'] = path_length / (ret_w.abs() + EPS)
        features[f'W1_HIGH_DIST_{window}'] = high_dist
        features[f'W1_LOW_DIST_{window}'] = low_dist
        features[f'W1_RSV_{window}'] = rsv
        features[f'W1_DRAWDOWN_{window}'] = close / (rolling_high + EPS) - 1.0
        features[f'W1_RANGE_MEAN_{window}'] = range_.rolling(window).mean()
        features[f'W1_TRUE_RANGE_MEAN_{window}'] = safe_div(true_range.rolling(window).mean(), prev_close)
        features[f'W1_PARKINSON_MEAN_{window}'] = parkinson.rolling(window).mean()
        features[f'W1_GARMAN_KLASS_MEAN_{window}'] = garman_klass.clip(lower=0.0).rolling(window).mean()
        features[f'W1_DOWNSIDE_VOL_{window}'] = downside.rolling(window).std()
        features[f'W1_UPSIDE_VOL_{window}'] = upside.rolling(window).std()
        features[f'W1_VOL_ASYM_{window}'] = safe_div(upside.rolling(window).std(), downside.abs().rolling(window).std()) - 1.0
        features[f'W1_TURN_MEAN_{window}'] = turnover_mean
        features[f'W1_TURN_Z_{window}'] = (turnover - turnover_mean) / (turnover_std + EPS)
        features[f'W1_AMOUNT_MEAN_{window}'] = amount_mean
        features[f'W1_AMOUNT_Z_{window}'] = (log_amount - amount_mean) / (amount_std + EPS)
        features[f'W1_VOLUME_RATIO_{window}'] = volume / (volume_mean + EPS)
        features[f'W1_AMIHUD_MEAN_{window}'] = amihud_mean
        features[f'W1_AMIHUD_Z_{window}'] = (amihud - amihud_mean) / (amihud_std + EPS)
        features[f'W1_SIGNED_TURN_SUM_{window}'] = signed_sum(ret_1, turnover, window)
        features[f'W1_SIGNED_AMOUNT_SUM_{window}'] = signed_sum(ret_1, log_amount.abs(), window)
        features[f'W1_MONEY_FLOW_{window}'] = safe_div((clv * amount).rolling(window).sum(), amount.rolling(window).sum())
        features[f'W1_RET_TURN_CORR_{window}'] = rolling_corr(ret_1, turnover, window)
        features[f'W1_RET_AMOUNT_CORR_{window}'] = rolling_corr(ret_1, log_amount, window)
        features[f'W1_BODY_MEAN_{window}'] = body.rolling(window).mean()
        features[f'W1_BODY_ABS_MEAN_{window}'] = body_abs.rolling(window).mean()
        features[f'W1_CLV_MEAN_{window}'] = clv.rolling(window).mean()
        features[f'W1_GAP_MEAN_{window}'] = gap.rolling(window).mean()
        features[f'W1_GAP_ABS_MEAN_{window}'] = gap.abs().rolling(window).mean()
        features[f'W1_GAP_FOLLOW_MEAN_{window}'] = gap_follow.rolling(window).mean()
        features[f'W1_SHADOW_BALANCE_MEAN_{window}'] = shadow_balance.rolling(window).mean()
        features[f'W1_LAST_RET_RANK_{window}'] = ret_1.rolling(window).apply(rolling_rank_last, raw=True)
        features[f'W1_LAST_TURN_RANK_{window}'] = turnover.rolling(window).apply(rolling_rank_last, raw=True)

    feature_df = pd.DataFrame(features, index=df.index)
    return pd.concat([df, feature_df], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)


WINDOW_1_FEATURE_ENGINEERS = {
    WINDOW_1_FEATURE_NUM: engineer_window_1_features,
}
