"""Loop evidence 内容完整性回归测试。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bdc_agentctl_evidence", REPO / ".agents" / "agentctl.py")
assert SPEC and SPEC.loader
agentctl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(agentctl)


def test_indexed_json_must_parse_unless_explicitly_marked_corrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agentctl, "ROOT", tmp_path)
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"broken":', encoding="utf-8")
    entry = {"path": "evidence.json", "kind": "metric", "sha256": "0" * 64}

    with pytest.raises(SystemExit, match="无法读取 JSON"):
        agentctl.validate_indexed_evidence_content([entry])

    entry["kind"] = agentctl.KNOWN_CORRUPTED_EVIDENCE_KIND
    assert agentctl.validate_indexed_evidence_content([entry]) == ["evidence.json"]


def test_loop_008_has_parseable_recovered_reports_and_explicit_legacy_damage() -> None:
    index_path = REPO / ".agents/loops/loop-008/evidence/index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    by_name = {Path(entry["path"]).name: entry for entry in index["entries"]}

    for name in (
        "e0-e3-structure-nested-invalid-preprocessing.json",
        "e0-e3-structure-nested.json",
    ):
        assert by_name[name]["kind"] == agentctl.KNOWN_CORRUPTED_EVIDENCE_KIND

    for name in (
        "e0-e3-structure-nested-invalid-preprocessing-recovered-v1.json",
        "e0-e3-structure-nested-recovered-v1.json",
    ):
        path = index_path.parent / name
        assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 2
