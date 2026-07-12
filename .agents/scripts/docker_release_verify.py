"""构建发布镜像并在无网络容器内执行代码审计、训练和预测。"""

import hashlib
import json
import argparse
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
IMAGE = 'bdc2026:latest'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def extract_bundle(bundle: Path, destination: Path) -> None:
    """只展开不会逃逸目录且不含链接的发布 tar。"""
    root = destination.resolve()
    with tarfile.open(bundle) as archive:
        for member in archive.getmembers():
            target = (root / member.name).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f'bundle member escapes destination: {member.name}')
            if member.issym() or member.islnk():
                raise ValueError(f'bundle links are not allowed: {member.name}')
        archive.extractall(root)


def verify_bundle_hashes(root: Path) -> dict[str, str]:
    manifest = root / 'SHA256SUMS'
    if not manifest.is_file():
        raise FileNotFoundError('bundle missing SHA256SUMS')
    verified = {}
    for line in manifest.read_text(encoding='utf-8').splitlines():
        expected, separator, name = line.partition('  ')
        if not separator or len(expected) != 64:
            raise ValueError(f'invalid SHA256SUMS line: {line!r}')
        relative = Path(name)
        path = (root / relative).resolve()
        if relative.is_absolute() or root.resolve() not in path.parents or not path.is_file():
            raise ValueError(f'invalid bundle member in SHA256SUMS: {name}')
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f'bundle SHA-256 mismatch: {name}')
        verified[relative.as_posix()] = actual
    required = {'bdc2026-image.tar', 'docker-compose.yml', 'readme.md', 'data/stock_data.csv', 'data/manifest.json'}
    if missing := sorted(required - set(verified)):
        raise ValueError(f'bundle hash manifest missing required files: {missing}')
    return verified


def container_hashes(container: str) -> dict[str, str]:
    command = [
        'docker', 'exec', container, 'python', '-c',
        "import hashlib,json,pathlib; root=pathlib.Path('/app'); m=root/'model/xgboost/metadata.json'; d=json.loads(m.read_text()); files=[root/'output/result.csv',m]+[root/'model/xgboost'/x['path'] for x in d['models']]; print(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},sort_keys=True))",
    ]
    result, _ = run(command, 'container-hashes.log', 60)
    return json.loads(result.stdout.strip())


def run(command: list[str], log_name: str, timeout: int, check: bool = True) -> tuple[subprocess.CompletedProcess, float]:
    started = time.perf_counter()
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=timeout,
    )
    seconds = time.perf_counter() - started
    (EVIDENCE / log_name).write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
    if check and result.returncode != 0:
        raise RuntimeError(f'command failed ({result.returncode}): {command}; see {log_name}')
    return result, seconds


def main() -> None:
    global EVIDENCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loop', required=True, help='证据写入的当前 loop 标识')
    args = parser.parse_args()
    if not args.loop.startswith('loop-'):
        raise ValueError('--loop must use loop-NNN')
    evidence = ROOT / '.agents' / 'loops' / args.loop / 'evidence' / 'docker-release'
    EVIDENCE = evidence
    from code.utils.submission import validate_result_file

    evidence.mkdir(parents=True, exist_ok=True)
    container = f'bdc2026-release-{int(time.time())}'
    payload = {
        'status': 'failed',
        'image': IMAGE,
        'runtime_network': 'none',
        'container': container,
    }
    error = None
    with tempfile.TemporaryDirectory(prefix='bdc2026-release-') as workspace:
      workspace_path = Path(workspace)
      bundle_path = ROOT / 'dist' / f'{args.loop}-bundle.tar'
      try:
        _, package_seconds = run(['bash', 'package.sh', bundle_path.name], 'package.log', 30 * 60)
        bundle_path = ROOT / 'dist' / bundle_path.name
        if not bundle_path.is_file():
            raise FileNotFoundError(f'package did not create {bundle_path}')
        extract_bundle(bundle_path, workspace_path)
        bundle_hashes = verify_bundle_hashes(workspace_path)
        bundle_bytes = bundle_path.stat().st_size
        image_tar = workspace_path / 'bdc2026-image.tar'
        if not image_tar.is_file():
            raise FileNotFoundError('bundle missing bdc2026-image.tar')
        run(['docker', 'load', '--input', str(image_tar)], 'docker-load.log', 10 * 60)
        output_dir = workspace_path / 'output'
        temp_dir = workspace_path / 'temp'
        output_dir.mkdir(exist_ok=True)
        temp_dir.mkdir(exist_ok=True)
        result_path = output_dir / 'result.csv'
        result_path.unlink(missing_ok=True)
        size_result, _ = run(
            ['docker', 'image', 'inspect', IMAGE, '--format', '{{.Size}}'],
            'docker-image-size.log', 60,
        )
        id_result, _ = run(
            ['docker', 'image', 'inspect', IMAGE, '--format', '{{.Id}}'],
            'docker-image-id.log', 60,
        )
        image_bytes = int(size_result.stdout.strip())
        run_command = [
            'docker', 'run', '--detach', '--name', container,
            '--network', 'none', '--gpus', 'all', '--memory', '16g', '--memory-swap', '16g',
            '--mount', f'type=bind,source={workspace_path / "data"},target=/app/data,readonly',
            '--mount', f'type=bind,source={output_dir},target=/app/output',
            '--mount', f'type=bind,source={temp_dir},target=/app/temp',
            IMAGE, 'sleep', 'infinity',
        ]
        run(run_command, 'docker-run.log', 120)
        audit_commands = [
            [
                'docker', 'exec', container, 'python', '-c',
                "from pathlib import Path; files=list(Path('code').rglob('*.py'))+list(Path('data').rglob('*.py')); [compile(p.read_bytes(),str(p),'exec') for p in files]; print(f'compiled {len(files)} python files in memory')",
            ],
            ['docker', 'exec', container, 'bash', '/app/init.sh'],
            [
                'docker', 'exec', container, 'python', '-c',
                "from code.utils.data_manifest import validate_data_manifest; validate_data_manifest('data/stock_data.csv','data/manifest.json')",
            ],
            [
                'docker', 'exec', container, 'python', '-c',
                "import json,pandas,xgboost,talib,yaml; print(json.dumps({'python':'3.10','pandas':pandas.__version__,'xgboost':xgboost.__version__,'talib':talib.__version__,'pyyaml':yaml.__version__}))",
            ],
        ]
        for index, command in enumerate(audit_commands, start=1):
            run(command, f'container-audit-{index}.log', 5 * 60)
        offline_probe, _ = run(
            [
                'docker', 'exec', container, 'python', '-c',
                "import socket; socket.create_connection(('1.1.1.1',53),timeout=2)",
            ],
            'container-offline-probe.log', 30, check=False,
        )
        if offline_probe.returncode == 0:
            raise RuntimeError('network probe unexpectedly succeeded inside --network none container')
        _, train_seconds_first = run(
            ['docker', 'exec', container, 'bash', '/app/train.sh'],
            'container-train-1.log', 8 * 60 * 60,
        )
        _, predict_seconds_first = run(
            ['docker', 'exec', container, 'bash', '/app/test.sh'],
            'container-predict-1.log', 5 * 60,
        )
        hashes_first = container_hashes(container)
        _, train_seconds_second = run(
            ['docker', 'exec', container, 'bash', '/app/train.sh'],
            'container-train-2.log', 8 * 60 * 60,
        )
        _, predict_seconds_second = run(
            ['docker', 'exec', container, 'bash', '/app/test.sh'],
            'container-predict-2.log', 5 * 60,
        )
        hashes_second = container_hashes(container)
        submission = validate_result_file(result_path)
        deterministic_container = hashes_first == hashes_second
        gates = {
            'container_code_audit': True,
            'runtime_offline': True,
            'deterministic_container_artifacts': deterministic_container,
            'under_train_limit': max(train_seconds_first, train_seconds_second) < 8 * 60 * 60,
            'under_predict_limit': max(predict_seconds_first, predict_seconds_second) < 5 * 60,
            'under_image_size_limit': image_bytes < 10 * 1024 ** 3,
            'under_bundle_size_limit': bundle_bytes < 10 * 1024 ** 3,
        }
        payload.update({
            'status': 'validated' if all(gates.values()) else 'failed',
            'gates': gates,
            'image_id': id_result.stdout.strip(),
            'image_bytes': image_bytes,
            'bundle_bytes': bundle_bytes,
            'bundle_sha256': sha256(bundle_path),
            'bundle_member_sha256': bundle_hashes,
            'package_seconds': package_seconds,
            'train_seconds': max(train_seconds_first, train_seconds_second),
            'predict_seconds': max(predict_seconds_first, predict_seconds_second),
            'container_run_1_hashes': hashes_first,
            'container_run_2_hashes': hashes_second,
            'result_sha256': sha256(result_path),
            **submission,
        })
        if not all(gates.values()):
            raise RuntimeError(f'Docker release gates failed: {gates}')
      except Exception as exception:
        error = exception
        payload['error'] = repr(exception)
      finally:
        subprocess.run(
            ['docker', 'rm', '--force', container], cwd=ROOT,
            capture_output=True, text=True, timeout=120,
        )
        (evidence / 'docker-release-verification.json').write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8',
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if error is not None:
        raise error


if __name__ == '__main__':
    main()
