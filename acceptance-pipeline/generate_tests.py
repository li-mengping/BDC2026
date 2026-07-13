"""根据 BDC2026 ATDD IR 生成 pytest 验收测试。"""

from __future__ import annotations

import json
from pathlib import Path


HEADER = '''"""由 acceptance-pipeline 生成；禁止手工编辑。"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
'''


def _official_window() -> str:
    return '''
def test_official_five_session_return_contract():
    from code.models.spine import EvaluationWindow

    dates = pd.date_range('2026-06-29', periods=5, freq='D')
    first = EvaluationWindow.from_frame(pd.DataFrame({
        '日期': dates, '开盘': [100, 101, 102, 103, 110], '收盘': [1, 2, 3, 4, 5],
    }))
    second = EvaluationWindow.from_frame(pd.DataFrame({
        '日期': dates, '开盘': [100, 101, 102, 103, 110], '收盘': [9, 9, 9, 9, 9999],
    }))
    assert first.return_value == pytest.approx(0.10)
    assert second.return_value == first.return_value
'''


def _full_week_policy() -> str:
    return '''
def test_full_week_policy_and_missing_week_guard():
    from code.utils.runtime_split import build_stock_weeks
    from code.utils.stock import has_contiguous_history

    rows = []
    for date in pd.date_range('2026-01-05', periods=5, freq='D'):
        rows.append({'股票代码': '000001', '日期': date, '开盘': 1, '收盘': 1})
    for date in pd.date_range('2026-01-12', periods=4, freq='D'):
        rows.append({'股票代码': '000001', '日期': date, '开盘': 1, '收盘': 1})
    weeks = build_stock_weeks(pd.DataFrame(rows))['000001']
    assert [week.start_date for week in weeks] == [pd.Timestamp('2026-01-05')]
    assert not has_contiguous_history(
        [pd.Timestamp('2026-01-05'), pd.Timestamp('2026-01-19')],
        pd.Timestamp('2026-01-26'),
        pd.to_datetime(['2026-01-05', '2026-01-12', '2026-01-19', '2026-01-26']),
    )
'''


def _cutoff_guard() -> str:
    return '''
def test_competition_cutoff_guard():
    from code.utils.runtime_split import validate_data_cutoff

    frame = pd.DataFrame({'日期': pd.to_datetime(['2026-06-26', '2026-06-29'])})
    with pytest.raises(ValueError, match='data cutoff'):
        validate_data_cutoff(frame, '2026-06-29')
'''


def _offline_workflow() -> str:
    return '''
def test_official_entrypoints_and_offline_guard_exist():
    from code.models.xgboost.predict import main as predict_main
    from code.models.xgboost.train import main as train_main

    assert callable(train_main)
    assert callable(predict_main)
    assert (ROOT / '.agents/scripts/offline_site/sitecustomize.py').is_file()
'''


def _result_contract() -> str:
    return '''
def test_result_contract(tmp_path: Path):
    from code.utils.submission import validate_result_file

    path = tmp_path / 'result.csv'
    path.write_text('stock_id,weight\\n000001,0.2\\n000002,0.3\\n', encoding='utf-8')
    summary = validate_result_file(path)
    assert summary == {'stock_count': 2, 'weight_sum': 0.5}
'''


def _reproducible_run() -> str:
    return '''
def test_reproducibility_gate_is_part_of_release_verifier():
    source = (ROOT / '.agents/scripts/release_verify.py').read_text(encoding='utf-8')
    assert 'for run_number in (1, 2)' in source
    assert 'deterministic_result' in source
    assert 'deterministic_model' in source
'''


def _load_agentctl_helper() -> str:
    return '''
def _load_agentctl():
    spec = importlib.util.spec_from_file_location('acceptance_agentctl', ROOT / '.agents/agentctl.py')
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
'''


def _immutable_goal() -> str:
    return '''
def test_achieved_goal_requires_superseding_goal(tmp_path: Path, monkeypatch):
    ctl = _load_agentctl()
    agents = tmp_path / '.agents'
    (agents / 'loops').mkdir(parents=True)
    monkeypatch.setattr(ctl, 'ROOT', tmp_path)
    monkeypatch.setattr(ctl, 'AGENTS', agents)
    monkeypatch.setattr(ctl, 'GOALS', agents / 'goals')
    old = {
        'schema_version': '2', 'id': 'old', 'title': 'old', 'objective': 'old objective',
        'success_criteria': ['criterion'],
        'verification': [{'command': 'python -m pytest', 'pass_criteria': 'exit 0'}],
        'constraints': [], 'non_goals': [], 'blockers': [], 'iteration_priority': ['verify'],
        'final_audit': {'status': 'passed', 'findings': ['done'], 'evidence': ['evidence/final.json']},
        'status': 'achieved', 'created_at': ctl.now(), 'updated_at': ctl.now(),
        'evidence': ['evidence/final.json'], 'supersedes': None, 'superseded_by': None,
    }
    ctl.write_json(agents / 'goals/old.json', old)
    with pytest.raises(SystemExit, match='不允许状态迁移'):
        ctl.goal_close(argparse.Namespace(id='old', status='blocked', evidence=None))
    ctl.goal_supersede(argparse.Namespace(id='old', new_id='new', title='纠偏', contract=None))
    assert ctl.read_json(agents / 'goals/new.json')['supersedes'] == 'old'
'''


def _handoff() -> str:
    return '''
def test_handoff_and_reflection_contract_fields():
    handoff_schema = json.loads((ROOT / '.agents/schemas/handoff-v2.schema.json').read_text(encoding='utf-8'))
    reflection_schema = json.loads((ROOT / '.agents/schemas/reflection-v2.schema.json').read_text(encoding='utf-8'))
    handoff_text = json.dumps(handoff_schema, ensure_ascii=False)
    reflection_text = json.dumps(reflection_schema, ensure_ascii=False)
    for field in ('inputs', 'outputs', 'verification', 'limitations', 'next_action'):
        assert field in handoff_text
    for field in ('expectation', 'observation', 'difference', 'failure_category', 'learning', 'action'):
        assert field in reflection_text
'''


def _commit_contract() -> str:
    return '''
def test_commit_contract_declares_all_agent_fields():
    ctl = _load_agentctl()
    assert ctl.COMMIT_FIELDS == (
        'Agent-Task:', 'Agent-Decision:', 'Agent-Limitation:', 'Agent-Harness:',
        'Agent-Skills:', 'Agent-Verification:', 'Agent-Feedback:',
    )
'''


def _direct_top5() -> str:
    return '''
def test_direct_top5_needs_no_covariance(monkeypatch):
    from code.portfolio import postprocess

    scored = pd.DataFrame({
        '股票代码': [f'{i:06d}' for i in range(1, 7)],
        'xgb_rank_pairwise_score': [6, 5, 4, 3, 2, 1],
    })
    monkeypatch.setattr(postprocess, 'estimate_weekly_covariance', lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    _, _, selected = postprocess.select_portfolio(
        scored, pd.DataFrame(), '2026-06-29', np.asarray([]), ['xgb_rank_pairwise'],
    )
    assert selected['stock_id'].tolist() == [f'{i:06d}' for i in range(1, 6)]
    assert selected['final_weight'].tolist() == [0.2] * 5
'''


def _release_bundle() -> str:
    return '''
def test_release_bundle_contract_is_declared():
    package = (ROOT / 'package.sh').read_text(encoding='utf-8')
    dockerfile = (ROOT / 'Dockerfile').read_text(encoding='utf-8')
    assert 'python:3.10-slim-bookworm@sha256:' in dockerfile
    assert 'PYTHON_IMAGE' in package
    for item in ('docker-compose.yml', 'stock_data.csv', 'manifest.json', 'hs300_stock_list.csv', 'readme.md', 'SHA256SUMS'):
        assert item in package
'''


SCENARIO_RENDERERS = {
    'official-five-session-return': _official_window,
    'full-week-training-policy': _full_week_policy,
    'competition-cutoff-guard': _cutoff_guard,
    'offline-workflow': _offline_workflow,
    'result-contract': _result_contract,
    'reproducible-run': _reproducible_run,
    'immutable-achieved-goal': _immutable_goal,
    'resumable-agent-handoff': _handoff,
    'commit-contract': _commit_contract,
    'direct-top5-contract': _direct_top5,
    'release-bundle-contract': _release_bundle,
}


def render_tests(cases: list[dict]) -> str:
    ids = {case['id'] for case in cases}
    missing = sorted(ids - set(SCENARIO_RENDERERS))
    stale = sorted(set(SCENARIO_RENDERERS) - ids)
    if missing or stale:
        raise ValueError(f'acceptance renderer mismatch: missing={missing}, stale={stale}')

    parts = [HEADER, 'import json\n', _load_agentctl_helper()]
    for case in cases:
        source = Path(case['source']).as_posix()
        parts.append(f"\n# source: {source}:{case['line']}\n")
        parts.append(SCENARIO_RENDERERS[case['id']]())
    return ''.join(parts)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ir_path = root / 'acceptance-pipeline' / 'ir' / 'specs.json'
    cases = json.loads(ir_path.read_text(encoding='utf-8'))
    if not cases:
        raise ValueError('没有可生成的验收规格')
    output = root / 'generated-acceptance-tests' / 'test_competition_workflow.py'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_tests(cases), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
