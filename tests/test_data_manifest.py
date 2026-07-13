from pathlib import Path

import pandas as pd
import pytest

from code.utils.data_manifest import build_data_manifest, validate_data_manifest


def market_frame() -> pd.DataFrame:
    return pd.DataFrame({
        '股票代码': ['000001', '000001'],
        '日期': ['2026/6/1', '2026/6/2'],
        '开盘': [10.0, 11.0],
        '收盘': [11.0, 12.0],
        '最高': [11.5, 12.5],
        '最低': [9.5, 10.5],
        '成交量': [100, 110],
        '成交额': [1000, 1210],
        '振幅': [20.0, 18.18],
        '涨跌额': [1.0, 1.0],
        '换手率': [1.0, 1.1],
        '涨跌幅': [10.0, 9.09],
    })


def test_manifest_round_trip_and_tamper_detection(tmp_path: Path):
    data_path = tmp_path / 'stock_data.csv'
    constituents_path = tmp_path / 'hs300_stock_list.csv'
    manifest_path = tmp_path / 'manifest.json'
    market_frame().to_csv(data_path, index=False, encoding='utf-8')
    constituents_path.write_text('code,code_name\nsz.000001,A\n', encoding='utf-8')

    build_data_manifest(data_path, constituents_path, manifest_path, source='baostock')
    summary = validate_data_manifest(data_path, manifest_path)

    assert summary['rows'] == 2
    assert summary['stock_count'] == 1
    data_path.write_text(data_path.read_text(encoding='utf-8') + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='hash'):
        validate_data_manifest(data_path, manifest_path)


def test_manifest_rejects_duplicate_market_keys(tmp_path: Path):
    data_path = tmp_path / 'stock_data.csv'
    constituents_path = tmp_path / 'hs300_stock_list.csv'
    duplicate = pd.concat([market_frame(), market_frame().iloc[[0]]], ignore_index=True)
    duplicate.to_csv(data_path, index=False, encoding='utf-8')
    constituents_path.write_text('code,code_name\nsz.000001,A\n', encoding='utf-8')

    with pytest.raises(ValueError, match='duplicate'):
        build_data_manifest(data_path, constituents_path, tmp_path / 'manifest.json', source='baostock')
