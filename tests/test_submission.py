from pathlib import Path

import pytest

from code.utils.submission import validate_result_file


def test_valid_result_returns_summary(tmp_path: Path):
    path = tmp_path / 'result.csv'
    path.write_text('stock_id,weight\n000001,0.2\n000002,0.3\n', encoding='utf-8')

    assert validate_result_file(path) == {'stock_count': 2, 'weight_sum': 0.5}


@pytest.mark.parametrize(
    'content',
    [
        'stock_id,weight\n000001,-0.1\n',
        'stock_id,weight\n000001,0.6\n000002,0.5\n',
        'stock_id,weight\n000001,0.2\n000001,0.2\n',
        'stock_id,weight\n000001,0.1\n000002,0.1\n000003,0.1\n000004,0.1\n000005,0.1\n000006,0.1\n',
        'code,weight\n000001,0.2\n',
    ],
)
def test_invalid_result_is_rejected(tmp_path: Path, content: str):
    path = tmp_path / 'result.csv'
    path.write_text(content, encoding='utf-8')

    with pytest.raises(ValueError):
        validate_result_file(path)
