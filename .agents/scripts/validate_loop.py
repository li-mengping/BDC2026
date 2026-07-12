"""校验 AutoResearchMatrix 状态文件可被新 Agent 恢复。"""

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    loops = sorted((root / '.agents' / 'loops').glob('loop-*'))
    if not loops:
        raise ValueError('no loop directories found')
    summaries = []
    for loop in loops:
        required = [
            loop / 'goal.md', loop / 'plan.md', loop / 'handoff.md',
            loop / 'state' / 'task_spec.md', loop / 'state' / 'progress.json',
            loop / 'state' / 'directions_tried.json', loop / 'state' / 'direction_proposals.json',
            loop / 'state' / 'findings.jsonl', loop / 'state' / 'iteration_log.jsonl',
            loop / 'logs' / 'work.jsonl', loop / 'logs' / 'audit.jsonl',
        ]
        missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
        if missing:
            raise ValueError(f'missing loop files: {missing}')
        for path in required:
            if path.suffix == '.json':
                json.loads(path.read_text(encoding='utf-8'))
            elif path.suffix == '.jsonl':
                for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
                    if line.strip():
                        try:
                            json.loads(line)
                        except json.JSONDecodeError as error:
                            raise ValueError(f'invalid JSONL: {path}:{line_no}') from error
        progress = json.loads((loop / 'state' / 'progress.json').read_text(encoding='utf-8'))
        summaries.append({'loop': loop.name, 'phase': progress['phase'], 'iteration': progress['iteration']})
    print(json.dumps(summaries, ensure_ascii=False))


if __name__ == '__main__':
    main()
