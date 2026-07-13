#!/usr/bin/env python3
"""BDC2026 仓库专属 Agent 控制器。

控制器只依赖 Python 标准库，所有状态都写入 .agents/，便于 Codex 与
Claude Code 通过同一事实源恢复工作。旧版 loop 只读，不会被此脚本改写。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / ".agents"
GOALS = AGENTS / "goals"
EXPERIMENTS = AGENTS / "experiments"
SCHEMAS = AGENTS / "schemas"
V2_STATES = {"planned", "active", "evidence_ready", "validated", "achieved", "rejected", "blocked", "superseded"}
TRANSITIONS = {
    "planned": {"active", "rejected", "superseded"},
    "active": {"evidence_ready", "blocked", "rejected", "superseded"},
    "evidence_ready": {"validated", "blocked", "rejected", "superseded"},
    "validated": {"achieved", "blocked", "superseded"},
    "blocked": {"active", "superseded"},
    "rejected": {"superseded"},
    "achieved": set(),
    "superseded": set(),
}
COMMIT_FIELDS = ("Agent-Task:", "Agent-Decision:", "Agent-Limitation:", "Agent-Harness:", "Agent-Skills:", "Agent-Verification:", "Agent-Feedback:")
OPEN_GOAL_STATES = {"planned", "active", "evidence_ready", "validated", "blocked"}
KNOWN_CORRUPTED_EVIDENCE_KIND = "corrupted-legacy-evidence"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"无法读取 JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"JSON 顶层必须是对象: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip(): continue
        try: value = json.loads(line)
        except json.JSONDecodeError as exc: raise SystemExit(f"JSONL 非法: {path}:{line_number}") from exc
        if not isinstance(value, dict): raise SystemExit(f"JSONL 每行必须是对象: {path}:{line_number}")
        values.append(value)
    return values


def goal_path(goal_id: str) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", goal_id):
        raise SystemExit("goal id 只能包含小写字母、数字、点、下划线和连字符")
    return GOALS / f"{goal_id}.json"


def loop_path(loop_id: str) -> Path:
    if not re.fullmatch(r"loop-[0-9]{3,}", loop_id):
        raise SystemExit("loop id 必须形如 loop-008")
    return AGENTS / "loops" / loop_id


def validate_evidence(values: list[str] | None) -> list[str]:
    """只接受仓库相对、不可逃逸的证据路径。"""
    result = values or []
    for value in result:
        normalized = value.replace("\\", "/")
        if re.match(r"^[A-Za-z]:/", normalized) or normalized.startswith("/") or ".." in Path(normalized).parts:
            raise SystemExit(f"evidence 必须是仓库相对路径: {value}")
    return result


def evidence_register(args: argparse.Namespace) -> None:
    """登记当前 loop 的 evidence 并计算其 SHA-256。"""
    loop_dir = loop_path(args.loop)
    relative = Path(args.path).as_posix()
    validate_evidence([relative])
    evidence_root = (loop_dir / "evidence").resolve()
    path = repo_file(relative).resolve()
    if evidence_root not in path.parents or not path.is_file():
        raise SystemExit("evidence 必须是当前 loop evidence 下的现有文件")
    index_path = loop_dir / "evidence" / "index.json"
    index = read_json(index_path)
    entries = [entry for entry in index.get("entries", []) if entry.get("path") != relative]
    entry = {"path": relative, "kind": args.kind, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    entries.append(entry)
    index.update({"schema_version": "2", "loop_id": args.loop, "entries": entries})
    write_json(index_path, index)
    print(json.dumps(entry, ensure_ascii=False))


def repo_file(value: str) -> Path:
    validate_evidence([value])
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise SystemExit(f"路径逃逸仓库: {value}") from exc
    return path


def validate_indexed_evidence_content(entries: list[dict[str, Any]]) -> list[str]:
    """解析机器可读证据；已知损坏的历史文件必须显式标记并对外报告。"""
    known_corrupted: list[str] = []
    for entry in entries:
        relative = entry["path"]
        if entry.get("kind") == KNOWN_CORRUPTED_EVIDENCE_KIND:
            known_corrupted.append(relative)
            continue
        path = repo_file(relative)
        if path.suffix == ".json":
            read_json(path)
        elif path.suffix == ".jsonl":
            read_jsonl(path)
    return known_corrupted


def schema_errors(value: Any, schema: dict[str, Any], location: str = "$") -> list[str]:
    """校验本仓库 schema 使用的 JSON Schema 2020-12 子集。"""
    errors: list[str] = []
    expected = schema.get("type")
    types = expected if isinstance(expected, list) else [expected] if expected else []
    type_map = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float), "boolean": bool, "null": type(None)}
    if types and not any(isinstance(value, type_map[item]) for item in types):
        return [f"{location}: 类型应为 {expected}"]
    if "const" in schema and value != schema["const"]: errors.append(f"{location}: 必须等于 {schema['const']}")
    if "enum" in schema and value not in schema["enum"]: errors.append(f"{location}: 不在允许枚举中")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0): errors.append(f"{location}: 字符串过短")
        if schema.get("pattern") and not re.search(schema["pattern"], value): errors.append(f"{location}: 不匹配 pattern")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0): errors.append(f"{location}: 列表项不足")
        if "items" in schema:
            for index, item in enumerate(value): errors.extend(schema_errors(item, schema["items"], f"{location}[{index}]"))
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value: errors.append(f"{location}: 缺少 {key}")
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties: errors.extend(schema_errors(item, properties[key], f"{location}.{key}"))
            elif schema.get("additionalProperties") is False: errors.append(f"{location}: 不允许字段 {key}")
    return errors


def validate_with_schema(value: dict[str, Any], name: str) -> None:
    schema = read_json(SCHEMAS / name)
    errors = schema_errors(value, schema)
    if errors: raise SystemExit("；".join(errors))


def goal_create(args: argparse.Namespace) -> None:
    if args.contract:
        value = read_json(repo_file(args.contract))
        goal_id = args.id or value.get("id")
    else:
        goal_id = args.id
        value = {
            "id": goal_id, "title": args.title, "objective": args.objective,
            "success_criteria": args.success_criterion or [],
            "verification": [{"command": command, "pass_criteria": args.verification_pass or "命令退出码为 0"} for command in (args.verification_command or [])],
            "constraints": args.constraint or [], "non_goals": args.non_goal or [],
            "blockers": args.blocker or [], "iteration_priority": args.iteration_priority or [],
            "final_audit": {"status": "pending", "findings": [], "evidence": []},
        }
    if not goal_id: raise SystemExit("goal contract 或 --id 必须提供 id")
    path = goal_path(goal_id)
    if path.exists():
        raise SystemExit(f"goal 已存在: {goal_id}")
    value.update({"schema_version": "2", "id": goal_id, "status": "planned", "created_at": now(), "updated_at": now(), "evidence": [], "supersedes": None, "superseded_by": None})
    validate_with_schema(value, "goal-v2.schema.json")
    write_json(path, value)
    print(json.dumps(value, ensure_ascii=False))


def goal_supersede(args: argparse.Namespace) -> None:
    old_path = goal_path(args.id)
    old = read_json(old_path)
    if old.get("status") == "superseded":
        raise SystemExit("goal 已经 superseded")
    new_path = goal_path(args.new_id)
    if new_path.exists():
        raise SystemExit(f"新 goal 已存在: {args.new_id}")
    new = read_json(repo_file(args.contract)) if getattr(args, "contract", None) else dict(old)
    new.update({"schema_version": "2", "id": args.new_id, "title": args.title or new.get("title") or f"替代 {args.id}", "status": "planned", "created_at": now(), "updated_at": now(), "evidence": [], "supersedes": args.id, "superseded_by": None, "final_audit": {"status": "pending", "findings": [], "evidence": []}})
    validate_with_schema(new, "goal-v2.schema.json")
    if old.get("status") != "achieved":
        old["status"], old["superseded_by"], old["updated_at"] = "superseded", args.new_id, now()
        write_json(old_path, old)
    write_json(new_path, new)
    print(json.dumps(new, ensure_ascii=False))


def goal_audit(args: argparse.Namespace) -> None:
    value = read_json(goal_path(args.id))
    validate_with_schema(value, "goal-v2.schema.json")
    validate_evidence(value.get("evidence"))
    print(json.dumps({"id": args.id, "valid": True, "status": value["status"]}, ensure_ascii=False))


def goal_close(args: argparse.Namespace) -> None:
    path = goal_path(args.id)
    value = read_json(path)
    target = args.status
    current = value.get("status")
    if target not in V2_STATES or target == "superseded": raise SystemExit("close 不接受该状态")
    if target not in TRANSITIONS.get(current, set()): raise SystemExit(f"不允许状态迁移: {current} -> {target}")
    evidence = validate_evidence(args.evidence)
    if evidence: value.setdefault("evidence", []).extend(evidence)
    if target in {"evidence_ready", "validated", "achieved"} and not value.get("evidence"):
        raise SystemExit("证据为空，不能进入证据/完成状态")
    if target == "achieved":
        if not args.final_finding:
            raise SystemExit("achieved 必须记录 final audit finding")
        value["final_audit"] = {"status": "passed", "findings": args.final_finding, "evidence": list(value["evidence"])}
    elif target == "rejected" and args.final_finding:
        value["final_audit"] = {"status": "failed", "findings": args.final_finding, "evidence": list(value.get("evidence", []))}
    value["status"], value["updated_at"] = target, now()
    validate_with_schema(value, "goal-v2.schema.json")
    write_json(path, value)
    print(json.dumps(value, ensure_ascii=False))


def loop_init(args: argparse.Namespace) -> None:
    path = loop_path(args.id)
    if path.exists(): raise SystemExit(f"loop 已存在: {args.id}")
    if not goal_path(args.goal_id).is_file(): raise SystemExit(f"goal 不存在: {args.goal_id}")
    for rel in ("state", "evidence", "logs"):
        (path / rel).mkdir(parents=True)
    write_json(path / "evidence" / "index.json", {"schema_version": "2", "loop_id": args.id, "entries": []})
    (path / "goal.md").write_text(f"# {args.id}\n\nGoal: `{args.goal_id}`\n\n状态由 `.agents/agentctl.py` 管理。\n", encoding="utf-8")
    (path / "plan.md").write_text("# 计划\n\n- 每一步写入 evidence。\n", encoding="utf-8")
    (path / "handoff.md").write_text("# 交接\n\n尚未交接。\n", encoding="utf-8")
    write_json(path / "state" / "loop.json", {"schema_version": "2", "id": args.id, "goal_id": args.goal_id, "status": "planned", "claimed_by": None, "created_at": now(), "updated_at": now()})
    for name, value in (("task_spec.md", "# 任务规格\n"), ("findings.jsonl", ""), ("iteration_log.jsonl", ""), ("work.jsonl", ""), ("audit.jsonl", "")):
        target = path / ("logs" if name in {"work.jsonl", "audit.jsonl"} else "state") / name
        target.write_text(value, encoding="utf-8")
    for name, value in (("progress.json", {"schema_version": "2", "phase": "planned", "iteration": 0}), ("directions_tried.json", []), ("direction_proposals.json", [])):
        write_json(path / "state" / name, value)
    print(args.id)


def loop_state(path: Path) -> tuple[Path, dict[str, Any]]:
    state_path = path / "state" / "loop.json"
    if not state_path.is_file(): raise SystemExit(f"不是 v2 loop: {path}")
    return state_path, read_json(state_path)


def loop_claim(args: argparse.Namespace) -> None:
    path = loop_path(args.id); state_path, state = loop_state(path)
    if state.get("claimed_by") and state["claimed_by"] != args.agent: raise SystemExit(f"loop 已被 {state['claimed_by']} 占用")
    state.update({"claimed_by": args.agent, "status": "active", "updated_at": now()}); write_json(state_path, state); print(args.agent)


def loop_handoff(args: argparse.Namespace) -> None:
    path = loop_path(args.id); state_path, state = loop_state(path)
    if args.from_agent and state.get("claimed_by") not in {None, args.from_agent}: raise SystemExit("交接来源不是当前持有者")
    value = {"schema_version": "2", "loop_id": args.id, "from": args.from_agent, "to": args.to_agent, "summary": args.summary, "inputs": args.input or [], "outputs": args.output or [], "verification": args.verification or [], "limitations": args.limitation or [], "next_action": args.next_action, "evidence": validate_evidence(args.evidence), "created_at": now()}
    validate_with_schema(value, "handoff-v2.schema.json")
    write_json(path / "state" / "handoff.json", value); (path / "handoff.md").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    state.update({"claimed_by": args.to_agent, "updated_at": now()}); write_json(state_path, state)
    print(json.dumps(value, ensure_ascii=False))


def loop_validate(args: argparse.Namespace) -> None:
    path = loop_path(args.id); _, state = loop_state(path)
    required = [path / "goal.md", path / "plan.md", path / "handoff.md", path / "evidence" / "index.json", path / "state" / "progress.json", path / "state" / "directions_tried.json", path / "state" / "direction_proposals.json"]
    missing = [str(x.relative_to(ROOT)) for x in required if not x.is_file()]
    if missing: raise SystemExit("缺少: " + ", ".join(missing))
    validate_with_schema(state, "loop-v2.schema.json")
    index = read_json(path / "evidence" / "index.json"); validate_with_schema(index, "evidence-index-v2.schema.json")
    absent = [entry["path"] for entry in index["entries"] if not repo_file(entry["path"]).is_file()]
    if absent: raise SystemExit("证据索引引用不存在文件: " + ", ".join(absent))
    mismatched = []
    for entry in index["entries"]:
        evidence_path = repo_file(entry["path"])
        if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != entry["sha256"]: mismatched.append(entry["path"])
    if mismatched: raise SystemExit("证据索引 SHA-256 不匹配: " + ", ".join(mismatched))
    known_corrupted = validate_indexed_evidence_content(index["entries"])
    indexed = {entry["path"] for entry in index["entries"]}
    orphaned = []
    for evidence_path in (path / "evidence").rglob("*"):
        if not evidence_path.is_file() or evidence_path.name == "index.json":
            continue
        relative = evidence_path.relative_to(ROOT).as_posix()
        if relative not in indexed:
            orphaned.append(relative)
    if orphaned: raise SystemExit("存在未登记 evidence: " + ", ".join(sorted(orphaned)))
    handoff = path / "state" / "handoff.json"
    if not handoff.is_file(): raise SystemExit("loop 尚未写入有效 handoff")
    validate_with_schema(read_json(handoff), "handoff-v2.schema.json")
    for filename, schema_name in (("experiments.jsonl", "experiment-v2.schema.json"), ("experiment-decisions.jsonl", "experiment-decision-v2.schema.json"), ("reflections.jsonl", "reflection-v2.schema.json")):
        state_file = path / "state" / filename
        if state_file.is_file():
            for value in read_jsonl(state_file): validate_with_schema(value, schema_name)
    print(json.dumps({
        "loop": args.id,
        "valid": True,
        "status": state.get("status"),
        "known_corrupted_evidence": known_corrupted,
    }, ensure_ascii=False))


def experiment_register(args: argparse.Namespace) -> None:
    loop = loop_path(args.loop); loop_state(loop)
    path = loop / "state" / "experiments.jsonl"; path.parent.mkdir(parents=True, exist_ok=True)
    value = {"schema_version": "2", "id": args.id, "loop_id": args.loop, "hypothesis": args.hypothesis, "direction": args.direction, "status": "proposed", "created_at": now()}
    validate_with_schema(value, "experiment-v2.schema.json")
    with path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    print(json.dumps(value, ensure_ascii=False))


def experiment_decide(args: argparse.Namespace) -> None:
    loop = loop_path(args.loop); loop_state(loop)
    path = loop / "state" / "experiment-decisions.jsonl"; path.parent.mkdir(parents=True, exist_ok=True)
    if args.decision not in {"accept", "reject", "hold"}: raise SystemExit("decision 必须是 accept/reject/hold")
    value = {"schema_version": "2", "id": args.id, "loop_id": args.loop, "decision": args.decision, "reason": args.reason, "evidence": validate_evidence(args.evidence), "created_at": now()}
    if not value["evidence"] and args.decision == "accept": raise SystemExit("accept 必须提供 evidence")
    validate_with_schema(value, "experiment-decision-v2.schema.json")
    with path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    print(json.dumps(value, ensure_ascii=False))


def reflect(args: argparse.Namespace) -> None:
    evidence = validate_evidence(args.evidence)
    if not evidence: raise SystemExit("反思必须关联 evidence")
    loop = loop_path(args.loop); loop_state(loop)
    path = loop / "state" / "reflections.jsonl"; path.parent.mkdir(parents=True, exist_ok=True)
    value = {"schema_version": "2", "loop_id": args.loop, "expectation": args.expectation, "observation": args.observation, "difference": args.difference, "failure_category": args.failure_category, "learning": args.learning, "action": args.action, "evidence": evidence, "created_at": now()}
    validate_with_schema(value, "reflection-v2.schema.json")
    with path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    print(json.dumps(value, ensure_ascii=False))


def commit_check(args: argparse.Namespace) -> None:
    message = args.message or ""
    lines = message.splitlines()
    if not lines or not re.fullmatch(r"[a-z]+\([a-z0-9._-]+\): .+", lines[0]):
        raise SystemExit("commit 首行必须是 <type>(<scope>): <summary>")
    missing = [field for field in COMMIT_FIELDS if field not in message]
    if missing: raise SystemExit("commit message 缺少: " + ", ".join(missing))
    fields = {line.split(":", 1)[0]: line.split(":", 1)[1].strip() for line in lines if ":" in line}
    empty = [field[:-1] for field in COMMIT_FIELDS if not fields.get(field[:-1])]
    if empty: raise SystemExit("commit 字段为空: " + ", ".join(empty))
    if not re.search(r"\.agents/loops/loop-[0-9]{3,}", fields.get("Agent-Harness", "").replace("\\", "/")):
        raise SystemExit("Agent-Harness 必须引用当前 loop 的仓库相对路径")
    if fields.get("Agent-Verification", "").lower() == "none":
        raise SystemExit("Agent-Verification 必须记录实际检查")
    result = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0: raise SystemExit("无法读取 git staged scope")
    staged = result.stdout.splitlines()
    if not staged: raise SystemExit("没有 staged 文件可校验")
    print(json.dumps({"valid": True, "staged": staged}, ensure_ascii=False))


def release_check(args: argparse.Namespace) -> None:
    required = [ROOT / "readme.md", ROOT / "docker-compose.yml"]
    missing = [str(x.relative_to(ROOT)) for x in required if not x.exists()]
    if missing: raise SystemExit("发布文件缺少: " + ", ".join(missing))
    print(json.dumps({"valid": True, "network": "must-be-disabled-at-runtime", "missing": []}, ensure_ascii=False))


def validate_skill(path: Path) -> list[str]:
    """以标准库校验仓库内 skill 的最小可发现契约。"""
    errors: list[str] = []
    skill_md = path / "SKILL.md"
    metadata = path / "agents" / "openai.yaml"
    if not skill_md.is_file():
        return [f"{path.name}:缺少 SKILL.md"]
    text = skill_md.read_text(encoding="utf-8", errors="ignore")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        errors.append(f"{path.name}:frontmatter 非法")
    else:
        frontmatter = match.group(1)
        name = re.search(r"^name:\s*([^\s]+)\s*$", frontmatter, re.MULTILINE)
        description = re.search(r"^description:\s*(.+)$", frontmatter, re.MULTILINE)
        if not name or name.group(1) != path.name:
            errors.append(f"{path.name}:name 与目录不一致")
        if not description or not description.group(1).strip():
            errors.append(f"{path.name}:description 为空")
    if not metadata.is_file():
        errors.append(f"{path.name}:缺少 agents/openai.yaml")
    elif f"${path.name}" not in metadata.read_text(encoding="utf-8", errors="ignore"):
        errors.append(f"{path.name}:default_prompt 未显式引用 skill")
    return errors


def discover_active_state() -> tuple[str | None, str | None, list[str]]:
    """从状态文件发现唯一当前 Goal 与其最新 loop。"""
    open_goals = []
    for path in sorted(GOALS.glob("*.json")):
        goal = read_json(path)
        if goal.get("status") in OPEN_GOAL_STATES:
            open_goals.append(str(goal.get("id") or path.stem))
    errors = []
    if len(open_goals) > 1:
        errors.append("存在多个未终结 Goal: " + ", ".join(open_goals))
        return None, None, errors
    active_goal = open_goals[0] if open_goals else None
    current_loop = None
    if active_goal:
        matching = []
        for state_path in sorted((AGENTS / "loops").glob("loop-*/state/loop.json")):
            state = read_json(state_path)
            if state.get("goal_id") == active_goal:
                matching.append(state_path.parents[1].name)
        if matching:
            current_loop = sorted(matching)[-1]
    return active_goal, current_loop, errors


def doctor(args: argparse.Namespace) -> None:
    required = [AGENTS / "AGENTS.md", AGENTS / "skills", AGENTS / "schemas", AGENTS / "loops"]
    missing = [str(x.relative_to(ROOT)) for x in required if not x.exists()]
    forbidden = []
    candidates = [ROOT / "AGENTS.md", ROOT / "CLAUDE.md"] + list((ROOT / ".claude").rglob("*")) + list(AGENTS.rglob("*"))
    forbidden_words = ("yinpin" + "_test", "sa" + "s4", "train" + "plat")
    for path in candidates:
        if path.is_file():
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".png", ".tar", ".zip"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            if re.search(r"(?<![a-z])[a-z]:[\\/]", text) or re.search(r"/(?:users|home)/[^/\s]+", text):
                forbidden.append(str(path.relative_to(ROOT)) + ":absolute-path")
            for word in forbidden_words:
                if word in text: forbidden.append(str(path.relative_to(ROOT)) + ":machine-specific")
    skill_errors = []
    for path in sorted((AGENTS / "skills").glob("*")):
        if path.is_dir(): skill_errors.extend(validate_skill(path))
    active_goal, current_loop, state_errors = discover_active_state()
    if missing or forbidden or skill_errors or state_errors:
        raise SystemExit(json.dumps({"missing": missing, "forbidden": forbidden, "skills": skill_errors, "state": state_errors}, ensure_ascii=False))
    print(json.dumps({"valid": True, "skills": len(list((AGENTS / "skills").glob("*/SKILL.md"))), "v2": True, "active_goal": active_goal, "current_loop": current_loop}, ensure_ascii=False))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BDC2026 Agent 状态与证据控制器")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor").set_defaults(func=doctor)
    goal = sub.add_parser("goal"); gs = goal.add_subparsers(dest="action", required=True)
    c = gs.add_parser("create"); c.add_argument("--contract"); c.add_argument("--id"); c.add_argument("--title"); c.add_argument("--objective"); c.add_argument("--success-criterion", action="append"); c.add_argument("--verification-command", action="append"); c.add_argument("--verification-pass"); c.add_argument("--constraint", action="append"); c.add_argument("--non-goal", action="append"); c.add_argument("--blocker", action="append"); c.add_argument("--iteration-priority", action="append"); c.set_defaults(func=goal_create)
    c = gs.add_parser("supersede"); c.add_argument("--id", required=True); c.add_argument("--new-id", required=True); c.add_argument("--title"); c.add_argument("--contract"); c.set_defaults(func=goal_supersede)
    c = gs.add_parser("audit"); c.add_argument("--id", required=True); c.set_defaults(func=goal_audit)
    c = gs.add_parser("close"); c.add_argument("--id", required=True); c.add_argument("--status", required=True, choices=sorted(V2_STATES - {"superseded"})); c.add_argument("--evidence", action="append"); c.add_argument("--final-finding", action="append"); c.set_defaults(func=goal_close)
    loop = sub.add_parser("loop"); ls = loop.add_subparsers(dest="action", required=True)
    c = ls.add_parser("init"); c.add_argument("--id", required=True); c.add_argument("--goal-id", required=True); c.set_defaults(func=loop_init)
    c = ls.add_parser("claim"); c.add_argument("--id", required=True); c.add_argument("--agent", required=True); c.set_defaults(func=loop_claim)
    c = ls.add_parser("handoff"); c.add_argument("--id", required=True); c.add_argument("--from", dest="from_agent"); c.add_argument("--to", dest="to_agent", required=True); c.add_argument("--summary", required=True); c.add_argument("--input", action="append"); c.add_argument("--output", action="append"); c.add_argument("--verification", action="append"); c.add_argument("--limitation", action="append"); c.add_argument("--next-action", required=True); c.add_argument("--evidence", action="append"); c.set_defaults(func=loop_handoff)
    c = ls.add_parser("validate"); c.add_argument("--id", required=True); c.set_defaults(func=loop_validate)
    evidence = sub.add_parser("evidence"); es = evidence.add_subparsers(dest="action", required=True)
    c = es.add_parser("register"); c.add_argument("--loop", required=True); c.add_argument("--path", required=True); c.add_argument("--kind", required=True); c.set_defaults(func=evidence_register)
    exp = sub.add_parser("experiment"); es = exp.add_subparsers(dest="action", required=True)
    c = es.add_parser("register"); c.add_argument("--loop", required=True); c.add_argument("--id", required=True); c.add_argument("--hypothesis", required=True); c.add_argument("--direction", required=True); c.set_defaults(func=experiment_register)
    c = es.add_parser("decide"); c.add_argument("--loop", required=True); c.add_argument("--id", required=True); c.add_argument("--decision", required=True); c.add_argument("--reason", required=True); c.add_argument("--evidence", action="append"); c.set_defaults(func=experiment_decide)
    c = sub.add_parser("reflect"); c.add_argument("--loop", required=True); c.add_argument("--expectation", required=True); c.add_argument("--observation", required=True); c.add_argument("--difference", required=True); c.add_argument("--failure-category", required=True); c.add_argument("--learning", required=True); c.add_argument("--action", required=True); c.add_argument("--evidence", action="append", required=True); c.set_defaults(func=reflect)
    c = sub.add_parser("commit"); cs = c.add_subparsers(dest="action", required=True); x = cs.add_parser("check"); x.add_argument("--message", required=True); x.set_defaults(func=commit_check)
    c = sub.add_parser("release"); rs = c.add_subparsers(dest="action", required=True); x = rs.add_parser("check"); x.set_defaults(func=release_check)
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    if not hasattr(arguments, "func"):
        raise SystemExit("缺少命令处理器")
    arguments.func(arguments)
