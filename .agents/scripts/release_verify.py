"""在断网 Python 子进程中执行双跑并生成发布证据。"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from code.utils.data_manifest import validate_data_manifest
from code.utils.submission import validate_result_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], env: dict[str, str], log_path: Path, timeout: int) -> float:
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    log_path.write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
    if result.returncode != 0:
        raise RuntimeError(f'command failed ({result.returncode}): {command}; see {log_path}')
    return time.perf_counter() - started


def verify_offline_guard(env: dict[str, str], evidence: Path) -> None:
    probe = subprocess.run(
        [sys.executable, '-c', "import socket; socket.create_connection(('127.0.0.1', 9), timeout=0.1)"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    log = probe.stdout + '\n' + probe.stderr
    (evidence / 'offline-guard-probe.log').write_text(log, encoding='utf-8')
    if probe.returncode == 0 or 'network access is disabled by BDC2026 release verification' not in log:
        raise RuntimeError('offline guard probe did not fail through sitecustomize')


def model_hashes(metadata_path: Path) -> dict[str, str]:
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    output_dir = metadata_path.parent
    hashes = {}
    for model in metadata['models']:
        artifact = output_dir / Path(model['path'])
        if not artifact.is_file():
            raise FileNotFoundError(f'model artifact missing: {artifact}')
        hashes[str(artifact.relative_to(ROOT))] = sha256(artifact)
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--loop', required=True, help='当前 loop 标识')
    parser.add_argument(
        '--evidence-dir', type=Path,
        default=None,
    )
    args = parser.parse_args()
    if not args.loop.startswith('loop-'):
        raise ValueError('--loop must use loop-NNN')
    evidence_dir = args.evidence_dir or Path('.agents') / 'loops' / args.loop / 'evidence' / 'host-release'
    evidence = (ROOT / evidence_dir).resolve()
    if ROOT not in evidence.parents:
        raise ValueError(f'evidence directory must stay inside repository: {evidence}')
    evidence.mkdir(parents=True, exist_ok=True)
    validate_data_manifest(ROOT / 'data' / 'stock_data.csv', ROOT / 'data' / 'manifest.json')
    offline_site = ROOT / '.agents' / 'scripts' / 'offline_site'
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(offline_site), str(ROOT)])
    env['CUDA_VISIBLE_DEVICES'] = env.get('CUDA_VISIBLE_DEVICES', '0')
    verify_offline_guard(env, evidence)
    runs = []
    for run_number in (1, 2):
        train_seconds = run(
            [sys.executable, '-m', 'code.models.xgboost.train'], env,
            evidence / f'run-{run_number}-train.log', 8 * 60 * 60,
        )
        predict_seconds = run(
            [sys.executable, '-m', 'code.models.xgboost.predict'], env,
            evidence / f'run-{run_number}-predict.log', 5 * 60,
        )
        result_path = ROOT / 'output' / 'result.csv'
        summary = validate_result_file(result_path)
        metadata_path = ROOT / 'model' / 'xgboost' / 'metadata.json'
        runs.append({
            'run': run_number,
            'train_seconds': train_seconds,
            'predict_seconds': predict_seconds,
            'result_sha256': sha256(result_path),
            'metadata_sha256': sha256(metadata_path),
            'model_artifact_sha256': model_hashes(metadata_path),
            **summary,
        })
    deterministic_result = runs[0]['result_sha256'] == runs[1]['result_sha256']
    deterministic_metadata = runs[0]['metadata_sha256'] == runs[1]['metadata_sha256']
    deterministic_model = runs[0]['model_artifact_sha256'] == runs[1]['model_artifact_sha256']
    repository_bytes = sum(
        path.stat().st_size for path in ROOT.rglob('*')
        if path.is_file() and '.git' not in path.parts and '.venv' not in path.parts
    )
    gates = {
        'offline_guard_active': True,
        'deterministic_result': deterministic_result,
        'deterministic_metadata': deterministic_metadata,
        'deterministic_model': deterministic_model,
        'under_train_limit': max(item['train_seconds'] for item in runs) < 8 * 60 * 60,
        'under_predict_limit': max(item['predict_seconds'] for item in runs) < 5 * 60,
        'under_workspace_size_limit': repository_bytes < 10 * 1024 ** 3,
    }
    payload = {
        'status': 'validated' if all(gates.values()) else 'failed',
        'offline_guard': str(offline_site.relative_to(ROOT)),
        'gates': gates,
        'max_train_seconds': max(item['train_seconds'] for item in runs),
        'max_predict_seconds': max(item['predict_seconds'] for item in runs),
        'repository_bytes_excluding_git_and_venv': repository_bytes,
        'size_scope': 'workspace_only; Docker image size is a separate release gate',
        'runs': runs,
    }
    output = evidence / 'release-verification.json'
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise RuntimeError(f'release gates failed: {failed}')


if __name__ == '__main__':
    main()
