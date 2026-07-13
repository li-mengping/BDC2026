"""冻结并校验比赛行情数据清单。"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    '股票代码', '日期', '开盘', '收盘', '最高', '最低',
    '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
}


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inspect_market_data(data_path: Path) -> dict:
    frame = pd.read_csv(data_path, dtype={'股票代码': str})
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f'missing columns: {sorted(missing)}')
    frame['股票代码'] = frame['股票代码'].astype(str).str.zfill(6)
    dates = pd.to_datetime(frame['日期'], errors='coerce')
    if dates.isna().any():
        raise ValueError('invalid market dates')
    if frame.duplicated(['股票代码', '日期']).any():
        raise ValueError('duplicate market keys')
    if frame.empty:
        raise ValueError('empty market data')
    return {
        'rows': int(len(frame)),
        'stock_count': int(frame['股票代码'].nunique()),
        'date_min': str(dates.min().date()),
        'date_max': str(dates.max().date()),
        'required_columns': sorted(REQUIRED_COLUMNS),
    }


def build_data_manifest(
    data_path: Path,
    constituents_path: Path,
    manifest_path: Path,
    source: str,
) -> dict:
    data_path = Path(data_path)
    constituents_path = Path(constituents_path)
    manifest_path = Path(manifest_path)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if not constituents_path.is_file():
        raise FileNotFoundError(constituents_path)
    summary = inspect_market_data(data_path)
    manifest = {
        'schema_version': 1,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': source,
        'data': {
            'file': data_path.name,
            'bytes': data_path.stat().st_size,
            'sha256': file_hash(data_path, 'sha256'),
            'md5': file_hash(data_path, 'md5'),
            **summary,
        },
        'constituents': {
            'file': constituents_path.name,
            'bytes': constituents_path.stat().st_size,
            'sha256': file_hash(constituents_path, 'sha256'),
            'md5': file_hash(constituents_path, 'md5'),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


def validate_data_manifest(data_path: Path, manifest_path: Path) -> dict:
    data_path = Path(data_path)
    manifest_path = Path(manifest_path)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1:
        raise ValueError('unsupported data manifest schema')
    expected = manifest.get('data', {})
    actual_hash = file_hash(data_path, 'sha256')
    if actual_hash != expected.get('sha256'):
        raise ValueError('data hash does not match manifest')
    summary = inspect_market_data(data_path)
    for key in ('rows', 'stock_count', 'date_min', 'date_max'):
        if summary[key] != expected.get(key):
            raise ValueError(f'data {key} does not match manifest')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('build', 'validate'))
    parser.add_argument('--data', type=Path, default=Path('data/stock_data.csv'))
    parser.add_argument('--constituents', type=Path, default=Path('data/hs300_stock_list.csv'))
    parser.add_argument('--manifest', type=Path, default=Path('data/manifest.json'))
    parser.add_argument('--source', default='baostock')
    args = parser.parse_args()
    if args.command == 'build':
        result = build_data_manifest(args.data, args.constituents, args.manifest, args.source)['data']
    else:
        result = validate_data_manifest(args.data, args.manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
