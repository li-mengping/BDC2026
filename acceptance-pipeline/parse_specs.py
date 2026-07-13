"""把 ATDD 文本规格解析为稳定 JSON IR。"""

import json
import re
from pathlib import Path


def parse_specs(spec_dir: Path) -> list[dict]:
    cases = []
    source_root = spec_dir.resolve().parent
    for path in sorted(spec_dir.glob('*.txt')):
        source = path.resolve().relative_to(source_root).as_posix()
        title = ''
        case_id = None
        steps = []
        first_line = None
        for line_no, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
            line = raw.strip()
            if line.startswith('; @id '):
                case_id = line.removeprefix('; @id ').strip()
            elif line.startswith(';') and not set(line[1:]) <= {'='}:
                title = line[1:].strip()
            match = re.match(r'^(GIVEN|WHEN|THEN)\s+(.+)[.。]$', line)
            if match:
                first_line = first_line or line_no
                steps.append({'kind': match.group(1), 'text': match.group(2), 'line': line_no})
            elif line and not line.startswith(';'):
                raise ValueError(f'{path}:{line_no}: 无法解析规格行')
            if line.startswith('THEN') and steps and line_no + 1 <= len(path.read_text(encoding='utf-8').splitlines()):
                continue
            if not line and steps and any(step['kind'] == 'THEN' for step in steps):
                if not case_id:
                    raise ValueError(f'{path}:{first_line}: 规格缺少 ; @id')
                cases.append({'id': case_id, 'source': source, 'line': first_line, 'title': title, 'steps': steps})
                steps = []
                first_line = None
                case_id = None
        if steps:
            if not case_id:
                raise ValueError(f'{path}:{first_line}: 规格缺少 ; @id')
            cases.append({'id': case_id, 'source': source, 'line': first_line, 'title': title, 'steps': steps})
    ids = [case['id'] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError('规格 @id 必须唯一')
    return cases


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / 'acceptance-pipeline' / 'ir' / 'specs.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(parse_specs(root / 'specs'), ensure_ascii=False, indent=2), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
