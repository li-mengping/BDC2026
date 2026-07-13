from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIGEST = "sha256:ff7161e2b8e2a56fc6a62a6099ff8feb72f1a6dbae9860cdcb9a6c65cf4c6be9"


def test_submission_readme_and_project_metadata_are_rooted() -> None:
    assert (ROOT / "readme.md").is_file()
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'readme = "readme.md"' in pyproject


def test_package_is_pinned_and_has_no_fallback() -> None:
    package = (ROOT / "package.sh").read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert PYTHON_DIGEST in package
    assert PYTHON_DIGEST in dockerfile
    assert "docker pull" not in package
    assert "PYTHON_IMAGE:-python:3.12" not in package
    for required in ("bdc2026-image.tar", "manifest.json", "hs300_stock_list.csv", "SHA256SUMS"):
        assert required in package


def test_bundle_contract_has_required_inputs_and_mounts() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "network_mode: none" in compose
    for directory in ("code", "data", "model", "output", "temp"):
        assert (ROOT / directory).is_dir()
    for script in ("init.sh", "train.sh", "test.sh"):
        assert (ROOT / script).is_file()


def test_release_verifiers_route_evidence_to_requested_loop_and_reload_bundle() -> None:
    docker_verify = (ROOT / ".agents/scripts/docker_release_verify.py").read_text(encoding="utf-8")
    host_verify = (ROOT / ".agents/scripts/release_verify.py").read_text(encoding="utf-8")
    assert "--loop" in docker_verify
    assert "--loop" in host_verify
    assert "docker', 'load'" in docker_verify
    assert "bdc2026-image.tar" in docker_verify
    assert "verify_bundle_hashes" in docker_verify
    assert "under_bundle_size_limit" in docker_verify
    assert "'--memory', '16g'" in docker_verify
    assert "loop-007" not in docker_verify
    assert "loop-007" not in host_verify
