"""基础和组合特征工程；训练/预测入口调用 preprocess_* 生成周频样本特征。"""

import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
import talib

from .windows import WINDOW_FEATURE_COLUMNS, WINDOW_FEATURE_ENGINEERS


EPS = 1e-12

FEATURE_COLUMNS = {
    '39': [
        'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
        'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
        'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread',
    ],
    '158+39': [
        'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
        'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0',
        'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5',
        'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5',
        'RSQR10', 'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5',
        'MAX10', 'MAX20', 'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10',
        'QTLU20', 'QTLU30', 'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10',
        'RANK20', 'RANK30', 'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10',
        'IMAX20', 'IMAX30', 'IMAX60', 'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10',
        'IMXD20', 'IMXD30', 'IMXD60', 'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10',
        'CORD20', 'CORD30', 'CORD60', 'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10',
        'CNTN20', 'CNTN30', 'CNTN60', 'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10',
        'SUMP20', 'SUMP30', 'SUMP60', 'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10',
        'SUMD20', 'SUMD30', 'SUMD60', 'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10',
        'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5', 'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10',
        'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5', 'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5',
        'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60', 'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd',
        'macd_signal', 'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k',
        'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 'volatility_10', 'volatility_20',
        'return_1', 'return_5', 'return_10', 'high_low_spread', 'open_close_spread', 'high_close_spread',
        'low_close_spread',
    ],
}
FEATURE_COLUMNS.update(WINDOW_FEATURE_COLUMNS)
for base_feature_num in ('39', '158+39'):
    for window_feature_num, window_columns in WINDOW_FEATURE_COLUMNS.items():
        FEATURE_COLUMNS[f'{base_feature_num}+{window_feature_num}'] = FEATURE_COLUMNS[base_feature_num] + [
            col for col in window_columns if col != 'instrument' and col not in FEATURE_COLUMNS[base_feature_num]
        ]


CROSS_SECTIONAL_SPECS = (
    ('PCT_CHG', '涨跌幅'),
    ('TURN', '换手率'),
    ('AMOUNT', '成交额'),
    ('RET_1', 'return_1'),
    ('RET_5', 'return_5'),
    ('RET_10', 'return_10'),
    ('VOL_10', 'volatility_10'),
    ('VOL_20', 'volatility_20'),
    ('WF_RET_5', 'WF_RET_5'),
    ('WF_RET_10', 'WF_RET_10'),
    ('WF_RET_20', 'WF_RET_20'),
    ('WF_RET_40', 'WF_RET_40'),
    ('WF_RET_60', 'WF_RET_60'),
    ('WF_IR_20', 'WF_RET_IR_20'),
    ('WF_IR_40', 'WF_RET_IR_40'),
    ('WF_IR_60', 'WF_RET_IR_60'),
    ('WF_VOL_20', 'WF_VOL_20'),
    ('WF_VOL_40', 'WF_VOL_40'),
    ('WF_VOL_60', 'WF_VOL_60'),
    ('WF_MDD_20', 'WF_MDD_20'),
    ('WF_MDD_40', 'WF_MDD_40'),
    ('WF_MDD_60', 'WF_MDD_60'),
    ('WF_RSV_20', 'WF_RSV_20'),
    ('WF_RSV_40', 'WF_RSV_40'),
    ('WF_RSV_60', 'WF_RSV_60'),
    ('WF_TURN_20', 'WF_TURN_MEAN_20'),
    ('WF_TURN_40', 'WF_TURN_MEAN_40'),
    ('WF_TURN_60', 'WF_TURN_MEAN_60'),
)

MARKET_STATE_SPECS = (
    ('PCT_CHG', '涨跌幅'),
    ('RET_1', 'return_1'),
    ('RET_5', 'return_5'),
    ('RET_10', 'return_10'),
    ('TURN', '换手率'),
    ('VOL_20', 'volatility_20'),
    ('WF_RET_20', 'WF_RET_20'),
    ('WF_VOL_20', 'WF_VOL_20'),
)


def cross_sectional_feature_columns(base_features: list[str]) -> list[str]:
    """根据基础特征列生成可用横截面特征列。"""
    base_set = set(base_features)
    columns = []
    for alias, source_col in CROSS_SECTIONAL_SPECS:
        if source_col in base_set:
            columns.extend([
                f'XS_RANK_{alias}',
                f'XS_REL_{alias}',
                f'XS_Z_{alias}',
            ])
    for alias, source_col in MARKET_STATE_SPECS:
        if source_col in base_set:
            columns.extend([
                f'MKT_MEAN_{alias}',
                f'MKT_STD_{alias}',
                f'MKT_POS_RATIO_{alias}',
                f'MKT_TOP20_{alias}',
                f'MKT_BOTTOM20_{alias}',
            ])
    return columns


for base_feature_num, base_columns in list(FEATURE_COLUMNS.items()):
    FEATURE_COLUMNS[f'{base_feature_num}+xsec'] = base_columns + [
        col for col in cross_sectional_feature_columns(base_columns) if col not in base_columns
    ]


def engineer_features_39(df: pd.DataFrame) -> pd.DataFrame:
    """计算单股 39 个基础技术指标。"""
    df = df.copy()
    open_ = df['开盘'].astype(float)
    high = df['最高'].astype(float)
    low = df['最低'].astype(float)
    close = df['收盘'].astype(float)
    volume = df['成交量'].astype(float)

    # 均线、动量和波动指标。
    df['sma_5'] = talib.SMA(close, timeperiod=5)
    df['sma_20'] = talib.SMA(close, timeperiod=20)
    df['ema_12'] = talib.EMA(close, timeperiod=12)
    df['ema_26'] = talib.EMA(close, timeperiod=26)
    df['ema_60'] = talib.EMA(close, timeperiod=60)
    df['macd'], df['macd_signal'], _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['rsi'] = talib.RSI(close, timeperiod=14)
    df['kdj_k'], df['kdj_d'] = talib.STOCH(high, low, close, fastk_period=9, slowk_period=3, slowd_period=3)
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']

    # 布林、ATR 和量价指标。
    df['boll_mid'], boll_upper, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['boll_std'] = (boll_upper - df['boll_mid']) / 2.0
    df['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
    df['obv'] = talib.OBV(close, volume)
    df['volume_change'] = volume.pct_change(fill_method=None)
    df['volume_ma_5'] = talib.SMA(volume, timeperiod=5)
    df['volume_ma_20'] = talib.SMA(volume, timeperiod=20)
    df['volume_ratio'] = df['volume_ma_5'] / (df['volume_ma_20'] + 1e-12)

    # 收益、波动和价差指标。
    df['return_1'] = close.pct_change(1, fill_method=None)
    df['return_5'] = close.pct_change(5, fill_method=None)
    df['return_10'] = close.pct_change(10, fill_method=None)
    df['volatility_10'] = df['return_1'].rolling(10).std()
    df['volatility_20'] = df['return_1'].rolling(20).std()
    df['high_low_spread'] = high - low
    df['open_close_spread'] = open_ - close
    df['high_close_spread'] = high - close
    df['low_close_spread'] = low - close
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def rolling_time_rsqr(values: np.ndarray) -> float:
    """滚动窗口内价格和时间序列线性相关的 R^2。"""
    y = np.asarray(values, dtype=np.float64)
    if len(y) < 2 or np.isnan(y).any():
        return np.nan
    x = np.arange(len(y), dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
    if denom <= 1e-12:
        return np.nan
    corr = float(np.sum(x * y) / denom)
    return corr * corr


def engineer_features_158(df: pd.DataFrame) -> pd.DataFrame:
    """计算单股 158 个 Alpha 风格滚动特征。"""
    df = df.copy()
    open_ = df['开盘'].astype(float)
    high = df['最高'].astype(float)
    low = df['最低'].astype(float)
    close = df['收盘'].astype(float)
    volume = df['成交量'].astype(float)
    vwap = df['成交额'].astype(float) / (volume + 1e-12)
    windows = [5, 10, 20, 30, 60]

    features = [
        (close - open_) / (open_ + 1e-12),
        (high - low) / (open_ + 1e-12),
        (close - open_) / (high - low + 1e-12),
        (high - pd.concat([open_, close], axis=1).max(axis=1)) / (open_ + 1e-12),
        (high - pd.concat([open_, close], axis=1).max(axis=1)) / (high - low + 1e-12),
        (pd.concat([open_, close], axis=1).min(axis=1) - low) / (open_ + 1e-12),
        (pd.concat([open_, close], axis=1).min(axis=1) - low) / (high - low + 1e-12),
        (2 * close - high - low) / (open_ + 1e-12),
        (2 * close - high - low) / (high - low + 1e-12),
        open_ / (close + 1e-12),
        high / (close + 1e-12),
        low / (close + 1e-12),
        vwap / (close + 1e-12),
    ]
    names = ['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2']
    names += ['OPEN0', 'HIGH0', 'LOW0', 'VWAP0']

    # 价格滚动窗口特征。
    for w in windows:
        features.append(close.shift(w) / (close + 1e-12))
        names.append(f'ROC{w}')
    for w in windows:
        features.append(talib.SMA(close, timeperiod=w) / (close + 1e-12))
        names.append(f'MA{w}')
    for w in windows:
        features.append(talib.STDDEV(close, timeperiod=w) / (close + 1e-12))
        names.append(f'STD{w}')
    for w in windows:
        slope = talib.LINEARREG_SLOPE(close, timeperiod=w)
        intercept = talib.LINEARREG_INTERCEPT(close, timeperiod=w)
        features.append(slope / (close + 1e-12))
        names.append(f'BETA{w}')
        features.append(close.rolling(w).apply(rolling_time_rsqr, raw=True))
        names.append(f'RSQR{w}')
        features.append((close - (slope * (w - 1) + intercept)) / (close + 1e-12))
        names.append(f'RESI{w}')

    # 极值、分位、排名和位置特征。
    for w in windows:
        features.append(talib.MAX(high, timeperiod=w) / (close + 1e-12))
        names.append(f'MAX{w}')
    for w in windows:
        features.append(talib.MIN(low, timeperiod=w) / (close + 1e-12))
        names.append(f'MIN{w}')
    for w in windows:
        features.append(close.rolling(w).quantile(0.8) / (close + 1e-12))
        names.append(f'QTLU{w}')
    for w in windows:
        features.append(close.rolling(w).quantile(0.2) / (close + 1e-12))
        names.append(f'QTLD{w}')
    for w in windows:
        features.append(close.rolling(w).rank(pct=True))
        names.append(f'RANK{w}')
    for w in windows:
        min_low = low.rolling(w).min()
        max_high = high.rolling(w).max()
        features.append((close - min_low) / (max_high - min_low + 1e-12))
        names.append(f'RSV{w}')
    for w in windows:
        features.append(high.rolling(w).apply(np.argmax, raw=True) / w)
        names.append(f'IMAX{w}')
    for w in windows:
        features.append(low.rolling(w).apply(np.argmin, raw=True) / w)
        names.append(f'IMIN{w}')
    for w in windows:
        imax = high.rolling(w).apply(np.argmax, raw=True)
        imin = low.rolling(w).apply(np.argmin, raw=True)
        features.append((imax - imin) / w)
        names.append(f'IMXD{w}')

    # 相关性和涨跌统计特征。
    log_volume = np.log(volume + 1.0)
    for w in windows:
        features.append(talib.CORREL(close, log_volume, timeperiod=w))
        names.append(f'CORR{w}')
    close_ret = close / (close.shift(1) + 1e-12)
    volume_ret = volume / (volume.shift(1) + 1e-12)
    corr_df = pd.concat([close_ret, np.log(volume_ret + 1.0)], axis=1).fillna(0.0)
    for w in windows:
        features.append(talib.CORREL(corr_df.iloc[:, 0], corr_df.iloc[:, 1], timeperiod=w))
        names.append(f'CORD{w}')

    close_diff = close - close.shift(1)
    close_up = close_diff > 0
    close_down = close_diff < 0
    for w in windows:
        features.append(close_up.rolling(w).mean())
        names.append(f'CNTP{w}')
    for w in windows:
        features.append(close_down.rolling(w).mean())
        names.append(f'CNTN{w}')
    for w in windows:
        features.append(close_up.rolling(w).mean() - close_down.rolling(w).mean())
        names.append(f'CNTD{w}')

    # 价格变化和成交量变化的方向强度特征。
    close_abs = close_diff.abs()
    close_pos = close_diff.clip(lower=0.0)
    close_neg = -close_diff.clip(upper=0.0)
    for w in windows:
        features.append(close_pos.rolling(w).sum() / (close_abs.rolling(w).sum() + 1e-12))
        names.append(f'SUMP{w}')
    for w in windows:
        features.append(close_neg.rolling(w).sum() / (close_abs.rolling(w).sum() + 1e-12))
        names.append(f'SUMN{w}')
    for w in windows:
        features.append((close_pos.rolling(w).sum() - close_neg.rolling(w).sum()) / (close_abs.rolling(w).sum() + 1e-12))
        names.append(f'SUMD{w}')
    for w in windows:
        features.append(talib.SMA(volume, timeperiod=w) / (volume + 1e-12))
        names.append(f'VMA{w}')
    for w in windows:
        features.append(talib.STDDEV(volume, timeperiod=w) / (volume + 1e-12))
        names.append(f'VSTD{w}')
    weighted_volume_return = (close / (close.shift(1) + 1e-12) - 1.0).abs() * volume
    for w in windows:
        features.append(weighted_volume_return.rolling(w).std() / (weighted_volume_return.rolling(w).mean() + 1e-12))
        names.append(f'WVMA{w}')

    volume_diff = volume - volume.shift(1)
    volume_abs = volume_diff.abs()
    volume_pos = volume_diff.clip(lower=0.0)
    volume_neg = -volume_diff.clip(upper=0.0)
    for w in windows:
        features.append(volume_pos.rolling(w).sum() / (volume_abs.rolling(w).sum() + 1e-12))
        names.append(f'VSUMP{w}')
    for w in windows:
        features.append(volume_neg.rolling(w).sum() / (volume_abs.rolling(w).sum() + 1e-12))
        names.append(f'VSUMN{w}')
    for w in windows:
        features.append((volume_pos.rolling(w).sum() - volume_neg.rolling(w).sum()) / (volume_abs.rolling(w).sum() + 1e-12))
        names.append(f'VSUMD{w}')

    feature_df = pd.concat(features, axis=1)
    feature_df.columns = names
    return pd.concat([df, feature_df], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def engineer_features_158plus39(df: pd.DataFrame) -> pd.DataFrame:
    """合并 Alpha 特征和基础技术指标。"""
    alpha = engineer_features_158(df)
    tech = engineer_features_39(df)
    tech_cols = [col for col in FEATURE_COLUMNS['39'] if col not in alpha.columns and col != 'instrument']
    merged = pd.concat([alpha, tech[tech_cols]], axis=1)
    return merged.loc[:, ~merged.columns.duplicated()].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def merge_window_features(base: pd.DataFrame, window: pd.DataFrame, window_feature_num: str) -> pd.DataFrame:
    """把窗口特征追加到已有特征，保留同名基础行情列一份。"""
    window_cols = [
        col
        for col in WINDOW_FEATURE_COLUMNS[window_feature_num]
        if col != 'instrument' and col in window.columns and col not in base.columns
    ]
    merged = pd.concat([base, window[window_cols]], axis=1)
    return merged.loc[:, ~merged.columns.duplicated()].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def engineer_features_with_window(df: pd.DataFrame, base_feature_num: str, window_feature_num: str) -> pd.DataFrame:
    """把基础因子和指定输入窗口因子合并。"""
    if base_feature_num == '39':
        base = engineer_features_39(df)
    elif base_feature_num == '158+39':
        base = engineer_features_158plus39(df)
    else:
        raise ValueError(f'unsupported base feature: {base_feature_num}')
    return merge_window_features(base, WINDOW_FEATURE_ENGINEERS[window_feature_num](df), window_feature_num)


FEATURE_ENGINEERS = {
    '39': engineer_features_39,
    '158+39': engineer_features_158plus39,
}
FEATURE_ENGINEERS.update(WINDOW_FEATURE_ENGINEERS)
for base_feature_num in ('39', '158+39'):
    for window_feature_num in WINDOW_FEATURE_ENGINEERS:
        FEATURE_ENGINEERS[f'{base_feature_num}+{window_feature_num}'] = partial(
            engineer_features_with_window,
            base_feature_num=base_feature_num,
            window_feature_num=window_feature_num,
        )


def resolve_feature_num(feature_num: str) -> tuple[str, bool]:
    """解析是否追加横截面特征。"""
    if feature_num.endswith('+xsec'):
        return feature_num.removesuffix('+xsec'), True
    return feature_num, False


def add_cross_sectional_features(processed: pd.DataFrame, base_features: list[str]) -> pd.DataFrame:
    """按日期追加横截面排名、相对市场和市场状态特征。"""
    work = processed.copy()
    group_key = '日期'
    features = {}
    grouped = work.groupby(group_key, sort=False)

    for alias, source_col in CROSS_SECTIONAL_SPECS:
        if source_col not in base_features or source_col not in work.columns:
            continue
        values = pd.to_numeric(work[source_col], errors='coerce').astype(float)
        mean = grouped[source_col].transform('mean').astype(float)
        std = grouped[source_col].transform('std').astype(float)
        features[f'XS_RANK_{alias}'] = grouped[source_col].rank(pct=True, method='average')
        features[f'XS_REL_{alias}'] = values - mean
        features[f'XS_Z_{alias}'] = (values - mean) / (std.fillna(0.0) + EPS)

    for alias, source_col in MARKET_STATE_SPECS:
        if source_col not in base_features or source_col not in work.columns:
            continue
        source = pd.to_numeric(work[source_col], errors='coerce').astype(float)
        by_date = source.groupby(work[group_key], sort=False)
        features[f'MKT_MEAN_{alias}'] = by_date.transform('mean')
        features[f'MKT_STD_{alias}'] = by_date.transform('std')
        features[f'MKT_POS_RATIO_{alias}'] = (source > 0.0).astype(float).groupby(work[group_key], sort=False).transform('mean')
        features[f'MKT_TOP20_{alias}'] = by_date.transform(lambda values: values.quantile(0.8))
        features[f'MKT_BOTTOM20_{alias}'] = by_date.transform(lambda values: values.quantile(0.2))

    if not features:
        return work
    feature_df = pd.DataFrame(features, index=work.index)
    return pd.concat([work, feature_df], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def preprocess_features(
    df: pd.DataFrame,
    feature_num: str,
    stockid2idx: dict[str, int],
) -> tuple[pd.DataFrame, list[str]]:
    """按股票并行计算特征，并映射截面内股票编号。"""
    base_feature_num, use_cross_sectional = resolve_feature_num(feature_num)
    if base_feature_num not in FEATURE_ENGINEERS:
        raise ValueError(f'unsupported feature_num: {feature_num}')

    work = df.copy()
    work['日期'] = pd.to_datetime(work['日期'], errors='coerce').dt.normalize()
    work = work.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [group for _, group in work.groupby('股票代码', sort=False)]
    if not groups:
        raise ValueError('empty dataframe, cannot engineer features')

    # 每只股票独立计算滚动特征，避免窗口跨股票泄漏。
    num_processes = min(10, mp.cpu_count(), len(groups))
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(pool.imap(FEATURE_ENGINEERS[base_feature_num], groups))

    processed = pd.concat(processed_list, ignore_index=True)
    processed['股票代码'] = processed['股票代码'].astype(str).str.zfill(6)
    processed['日期'] = pd.to_datetime(processed['日期'], errors='coerce').dt.normalize()
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.float32)

    base_features = FEATURE_COLUMNS[base_feature_num]
    if use_cross_sectional:
        processed = add_cross_sectional_features(processed, base_features)
    features = FEATURE_COLUMNS[feature_num]
    missing = [col for col in features if col not in processed.columns]
    if missing:
        raise ValueError(f'missing engineered features: {missing[:10]}')

    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan)
    return processed.sort_values(['日期', '股票代码']).reset_index(drop=True), features


def preprocess_stock_data_samples(samples: tuple, feature_num: str, stockid2idx: dict[str, int]) -> tuple[pd.DataFrame, list[str]]:
    """从 StockData 样本构造周频训练特征。"""
    if not samples:
        raise ValueError('empty stock data samples')

    source = stock_history_source_frame((sample.stock_id, sample.history_weeks) for sample in samples)
    processed, features = preprocess_features(source, feature_num, stockid2idx)
    feature_rows = processed.rename(columns={'日期': 'feature_date'})

    sample_rows = []
    for sample_id, sample in enumerate(samples):
        sample_rows.append({
            'sample_id': sample_id,
            '股票代码': sample.stock_id,
            'feature_date': sample.history_weeks[-1].end_date,
            'target_week': sample.future_week.start_date,
            'label': sample.future_return,
        })

    sample_df = pd.DataFrame(sample_rows)
    merged = sample_df.merge(feature_rows, on=['股票代码', 'feature_date'], how='inner').copy()
    if len(merged) != len(sample_df):
        raise ValueError(f'missing sample features: {len(sample_df) - len(merged)}')

    merged['日期'] = merged['target_week']
    return merged.sort_values(['日期', '股票代码']).reset_index(drop=True), features


def preprocess_stock_history_samples(samples: tuple, feature_num: str, stockid2idx: dict[str, int]) -> tuple[pd.DataFrame, list[str]]:
    """从历史周样本构造预测特征。"""
    if not samples:
        raise ValueError('empty stock history samples')

    source = stock_history_source_frame(samples)
    processed, features = preprocess_features(source, feature_num, stockid2idx)
    feature_rows = processed.rename(columns={'日期': 'feature_date'})

    sample_rows = []
    for sample_id, (stock_id, history_weeks) in enumerate(samples):
        sample_rows.append({
            'sample_id': sample_id,
            '股票代码': stock_id,
            'feature_date': history_weeks[-1].end_date,
            'target_week': history_weeks[-1].start_date + pd.Timedelta(days=7),
        })

    sample_df = pd.DataFrame(sample_rows)
    merged = sample_df.merge(feature_rows, on=['股票代码', 'feature_date'], how='inner').copy()
    if len(merged) != len(sample_df):
        raise ValueError(f'missing history features: {len(sample_df) - len(merged)}')

    merged['日期'] = merged['target_week']
    return merged.sort_values(['股票代码']).reset_index(drop=True), features


def stock_history_source_frame(samples) -> pd.DataFrame:
    """拼接历史周行情，重复周只保留一份。"""
    frames = []
    seen = set()
    for stock_id, history_weeks in samples:
        for week in history_weeks:
            key = (stock_id, week.start_date)
            if key in seen:
                continue
            seen.add(key)
            frames.append(week.to_frame())
    if not frames:
        raise ValueError('empty history weeks')
    return pd.concat(frames, ignore_index=True).sort_values(['股票代码', '日期']).reset_index(drop=True)


def date_group_sizes(df: pd.DataFrame, min_group_size: int) -> tuple[pd.DataFrame, list[int]]:
    """按交易日生成 XGBoost ranking group。"""
    grouped = df.sort_values(['日期', '股票代码']).groupby('日期', sort=True)
    frames = []
    groups = []
    for _, group in grouped:
        if len(group) < min_group_size:
            continue
        frames.append(group)
        groups.append(len(group))

    if not frames:
        raise ValueError('no valid date groups after min_group_size filter')
    return pd.concat(frames, ignore_index=True), groups
