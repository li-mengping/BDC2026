"""记录指定 Python 环境和 GPU 的可审计快照。"""

import argparse
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def portable_freeze(lines: list[str]) -> list[str]:
    """移除 pip freeze 中不可分发的本机 file URL。"""
    return [re.sub(r'\s+@\s+file://\S+', ' @ <local-build>', line) for line in lines]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    snapshot = {
        'captured_at_utc': datetime.now(timezone.utc).isoformat(),
        'python_executable': '<active-python>',
        'python_version': sys.version,
        'platform': platform.platform(),
        'pip_freeze': portable_freeze(
            command_output([sys.executable, '-m', 'pip', 'freeze']).splitlines()
        ),
        'gpu': command_output([
            'nvidia-smi',
            '--query-gpu=name,memory.total,driver_version',
            '--format=csv,noheader',
        ]).splitlines(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
