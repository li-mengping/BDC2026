"""仓库专属 Agent Harness V2 的状态机与分发测试。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bdc_agentctl", REPO / ".agents" / "agentctl.py")
assert SPEC and SPEC.loader
agentctl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(agentctl)


@pytest.fixture
def harness_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    agents = tmp_path / ".agents"
    (agents / "loops").mkdir(parents=True)
    monkeypatch.setattr(agentctl, "ROOT", tmp_path)
    monkeypatch.setattr(agentctl, "AGENTS", agents)
    monkeypatch.setattr(agentctl, "GOALS", agents / "goals")
    return tmp_path


def ns(**values: object) -> argparse.Namespace:
    return argparse.Namespace(**values)


def goal_args(goal_id: str) -> argparse.Namespace:
    return ns(contract=None, id=goal_id, title="纠偏", objective="形成可验证改进", success_criterion=["测试通过"], verification_command=["python -m pytest"], verification_pass="退出码为 0", constraint=[], non_goal=[], blocker=[], iteration_priority=["先纠正口径"])


def test_goal_requires_evidence_and_achieved_is_immutable(harness_root: Path) -> None:
    agentctl.goal_create(goal_args("goal-008"))
    agentctl.goal_close(ns(id="goal-008", status="active", evidence=None, final_finding=None))
    with pytest.raises(SystemExit, match="证据为空"):
        agentctl.goal_close(ns(id="goal-008", status="evidence_ready", evidence=None, final_finding=None))
    agentctl.goal_close(ns(id="goal-008", status="evidence_ready", evidence=["evidence/a.json"], final_finding=None))
    agentctl.goal_close(ns(id="goal-008", status="validated", evidence=None, final_finding=None))
    agentctl.goal_close(ns(id="goal-008", status="achieved", evidence=None, final_finding=["全部门禁通过"]))
    with pytest.raises(SystemExit, match="不允许状态迁移"):
        agentctl.goal_close(ns(id="goal-008", status="blocked", evidence=None, final_finding=None))


def test_achieved_goal_can_only_be_superseded(harness_root: Path) -> None:
    agentctl.goal_create(goal_args("old"))
    old = json.loads((harness_root / ".agents/goals/old.json").read_text(encoding="utf-8"))
    old.update(status="achieved", evidence=["evidence/final.json"])
    agentctl.write_json(harness_root / ".agents/goals/old.json", old)
    agentctl.goal_supersede(ns(id="old", new_id="new", title="新目标", contract=None))
    assert json.loads((harness_root / ".agents/goals/old.json").read_text(encoding="utf-8"))["status"] == "achieved"
    assert json.loads((harness_root / ".agents/goals/new.json").read_text(encoding="utf-8"))["supersedes"] == "old"


def test_supersede_can_replace_objective_with_new_contract(harness_root: Path) -> None:
    agentctl.goal_create(goal_args("old"))
    contract = {
        "id": "ignored", "title": "纠偏合同", "objective": "重建模型结构审计",
        "success_criteria": ["结构实验可恢复"], "verification": [{"command": "python -m pytest", "pass_criteria": "全部通过"}],
        "constraints": ["离线"], "non_goals": [], "blockers": [], "iteration_priority": ["先审计结构"],
        "final_audit": {"status": "pending", "findings": [], "evidence": []},
    }
    (harness_root / "new-contract.json").write_text(json.dumps(contract, ensure_ascii=False), encoding="utf-8")
    agentctl.goal_supersede(ns(id="old", new_id="new", title=None, contract="new-contract.json"))
    new = json.loads((harness_root / ".agents/goals/new.json").read_text(encoding="utf-8"))
    assert new["id"] == "new"
    assert new["objective"] == "重建模型结构审计"
    assert new["status"] == "planned" and new["evidence"] == [] and new["supersedes"] == "old"


def test_loop_claim_conflict_handoff_and_validation(harness_root: Path) -> None:
    agentctl.goal_create(goal_args("goal-008"))
    agentctl.loop_init(ns(id="loop-008", goal_id="goal-008"))
    agentctl.loop_claim(ns(id="loop-008", agent="implementer"))
    with pytest.raises(SystemExit, match="占用"):
        agentctl.loop_claim(ns(id="loop-008", agent="reviewer"))
    agentctl.loop_handoff(ns(id="loop-008", from_agent="implementer", to_agent="reviewer", summary="实现完成", input=["配置"], output=["代码"], verification=["pytest"], limitation=["顺序自审"], next_action="独立复核", evidence=["evidence/test.json"]))
    state = json.loads((harness_root / ".agents/loops/loop-008/state/loop.json").read_text(encoding="utf-8"))
    assert state["claimed_by"] == "reviewer"
    agentctl.loop_validate(ns(id="loop-008"))


def test_reflection_requires_evidence(harness_root: Path) -> None:
    with pytest.raises(SystemExit, match="必须关联"):
        agentctl.reflect(ns(loop="loop-008", expectation="a", observation="b", difference="c", failure_category="data", learning="d", action="e", evidence=[]))


def test_evidence_rejects_absolute_or_escaping_paths() -> None:
    with pytest.raises(SystemExit, match="仓库相对路径"):
        agentctl.validate_evidence(["X:/private/evidence.json"])
    with pytest.raises(SystemExit, match="仓库相对路径"):
        agentctl.validate_evidence(["../outside.json"])


def test_goal_create_accepts_complete_repo_relative_contract(harness_root: Path) -> None:
    contract = {
        "id": "contract-goal", "title": "合同目标", "objective": "完成闭环",
        "success_criteria": ["可恢复"], "verification": [{"command": "python test.py", "pass_criteria": "退出码为 0"}],
        "constraints": [], "non_goals": [], "blockers": [], "iteration_priority": ["审计"],
        "final_audit": {"status": "pending", "findings": [], "evidence": []},
    }
    (harness_root / "contract.json").write_text(json.dumps(contract, ensure_ascii=False), encoding="utf-8")
    agentctl.goal_create(ns(contract="contract.json", id=None))
    agentctl.goal_audit(ns(id="contract-goal"))


def test_goal_schema_rejects_incomplete_contract(harness_root: Path) -> None:
    (harness_root / "bad.json").write_text(json.dumps({"id": "bad", "title": "不完整"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="objective"):
        agentctl.goal_create(ns(contract="bad.json", id=None))


def test_current_goal_and_loop_are_discovered_from_state(harness_root: Path) -> None:
    agentctl.goal_create(goal_args("goal-008"))
    agentctl.goal_close(ns(id="goal-008", status="active", evidence=None, final_finding=None))
    agentctl.loop_init(ns(id="loop-008", goal_id="goal-008"))
    goal_id, loop_id, errors = agentctl.discover_active_state()
    assert (goal_id, loop_id, errors) == ("goal-008", "loop-008", [])


def test_loop_validate_rejects_missing_indexed_evidence(harness_root: Path) -> None:
    agentctl.goal_create(goal_args("goal-008"))
    agentctl.loop_init(ns(id="loop-008", goal_id="goal-008"))
    index = {"schema_version": "2", "loop_id": "loop-008", "entries": [{"path": ".agents/loops/loop-008/evidence/missing.json", "kind": "metric", "sha256": "a" * 64}]}
    agentctl.write_json(harness_root / ".agents/loops/loop-008/evidence/index.json", index)
    with pytest.raises(SystemExit, match="不存在文件"):
        agentctl.loop_validate(ns(id="loop-008"))


def test_loop_validate_checks_jsonl_schema_and_evidence_hash(harness_root: Path) -> None:
    agentctl.goal_create(goal_args("goal-008"))
    agentctl.loop_init(ns(id="loop-008", goal_id="goal-008"))
    evidence = harness_root / ".agents/loops/loop-008/evidence/metric.json"
    evidence.write_text("{}", encoding="utf-8")
    index = {"schema_version": "2", "loop_id": "loop-008", "entries": [{"path": ".agents/loops/loop-008/evidence/metric.json", "kind": "metric", "sha256": "0" * 64}]}
    agentctl.write_json(harness_root / ".agents/loops/loop-008/evidence/index.json", index)
    with pytest.raises(SystemExit, match="SHA-256"):
        agentctl.loop_validate(ns(id="loop-008"))
    index["entries"][0]["sha256"] = agentctl.hashlib.sha256(evidence.read_bytes()).hexdigest()
    agentctl.write_json(harness_root / ".agents/loops/loop-008/evidence/index.json", index)
    agentctl.loop_handoff(ns(id="loop-008", from_agent=None, to_agent="reviewer", summary="进入审查", input=["metric"], output=["review"], verification=["pytest"], limitation=[], next_action="校验实验", evidence=[]))
    (harness_root / ".agents/loops/loop-008/state/experiments.jsonl").write_text('{"schema_version":"2","status":"unknown"}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="hypothesis"):
        agentctl.loop_validate(ns(id="loop-008"))


def test_evidence_registration_and_orphan_guard(harness_root: Path) -> None:
    agentctl.goal_create(goal_args("goal-008"))
    agentctl.loop_init(ns(id="loop-008", goal_id="goal-008"))
    evidence = harness_root / ".agents/loops/loop-008/evidence/metric.json"
    evidence.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="未登记"):
        agentctl.loop_validate(ns(id="loop-008"))
    agentctl.evidence_register(ns(loop="loop-008", path=".agents/loops/loop-008/evidence/metric.json", kind="metric"))
    with pytest.raises(SystemExit, match="尚未写入有效 handoff"):
        agentctl.loop_validate(ns(id="loop-008"))


def test_commit_template_rejects_missing_fields() -> None:
    with pytest.raises(SystemExit, match="缺少"):
        agentctl.commit_check(ns(message="feat(agent): add harness\n\n说明"))


def test_public_cli_contains_all_contract_commands() -> None:
    help_text = agentctl.parser().format_help()
    for command in ("doctor", "goal", "loop", "experiment", "reflect", "commit", "release"):
        assert command in help_text


def test_schemas_and_skill_metadata_are_valid() -> None:
    schemas = sorted((REPO / ".agents/schemas").glob("*.schema.json"))
    assert len(schemas) >= 4
    for path in schemas:
        assert json.loads(path.read_text(encoding="utf-8"))["$schema"].endswith("2020-12/schema")
    skills = sorted((REPO / ".agents/skills").glob("*/SKILL.md"))
    assert {path.parent.name for path in skills} >= {"bdc2026-goal", "bdc2026-develop", "bdc2026-review", "bdc2026-audit", "bdc2026-experiment", "bdc2026-release"}
    for path in skills:
        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\nname:")
        assert (path.parent / "agents/openai.yaml").is_file()


def test_harness_contains_no_machine_specific_paths_or_names() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", ".agents"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    paths = [REPO / value for value in tracked]
    paths += [REPO / "AGENTS.md", REPO / "CLAUDE.md", REPO / ".agents/agentctl.py"]
    for directory in (".claude", ".agents/agents", ".agents/schemas", ".agents/skills"):
        paths += list((REPO / directory).rglob("*"))
    forbidden_words = ("yinpin" + "_test", "sa" + "s4", "train" + "plat")
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        assert not re.search(r"(?<![a-z])[a-z]:[\\/]", text), path
        assert not re.search(r"/(?:users|home)/[^/\s]+", text), path
        assert all(word not in text for word in forbidden_words), path
