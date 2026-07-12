{
  "schema_version": "2",
  "loop_id": "loop-008",
  "from": "model-architect",
  "to": "experiment-runner",
  "summary": "E0/E2/E3 与 XGBoost 已统一为折内 checkpoint、完整 outer-train 重训和同折 outer-only 评估",
  "inputs": [
    "code/experiments/real_structure_benchmark.py",
    ".agents/loops/loop-008/evidence/e1-representation-nested.json"
  ],
  "outputs": [
    ".agents/loops/loop-008/evidence/e0-e3-structure-nested.json"
  ],
  "verification": [
    "python -m pytest tests/test_real_structure_benchmark.py tests/test_neural_experiments.py -q"
  ],
  "limitations": [
    "clean-room E0 仅复现公开结构契约；full 运行预算为30 epoch上限，rolling 不是独立 holdout"
  ],
  "next_action": "运行四折 nested 结构比较并登记实际显存、时间和 Pareto",
  "evidence": [
    "code/experiments/real_structure_benchmark.py",
    "tests/test_real_structure_benchmark.py"
  ],
  "created_at": "2026-07-12T09:06:56+00:00"
}
