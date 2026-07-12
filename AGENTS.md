# BDC2026 Project Agent Rules

Version: v1.0.1  
Scope: repository root and all descendants. A more specific nested `AGENTS.md` adds local rules. Ask the user before resolving any conflict.

## 1. User authorization and file versions

- Do not modify, overwrite, delete, or move an existing file without first describing the target, reason, and expected change, then obtaining explicit user confirmation.
- A file operation explicitly named by the user in the current request is authorized only for that named operation. Do not expand it to other existing files.
- Give every new artifact a clear version. Fixed-name files such as `AGENTS.md` record a semantic version in their body. Iterated artifacts use `name-vN`, `name-vN.M`, or the monotonic Goal/Loop identifiers created by `.agents/agentctl.py`.
- Historical Goals, Loops, experiment conclusions, and evidence are read-only. Correct them with a new version or `goal supersede`; never rewrite history.
- Do not commit, push, publish, write to remotes, or switch production configuration unless the user explicitly requests it.

## 2. Control plane and startup order

1. Run `python .agents/agentctl.py doctor`.
2. Read `active_goal` and `current_loop` only from `doctor`; never infer them from filenames or prose.
3. Read the active Goal and current Loop's `state/loop.json`, `plan.md`, `handoff.md`, evidence index, and latest reflections.
4. Select the smallest applicable skill under `.agents/skills/` and follow its complete `SKILL.md` plus required references.
5. Change state only through `.agents/agentctl.py`; run `goal audit` or the applicable verifier after every transition.

## 3. Working method

- Use this closed loop: factual audit -> one falsifiable hypothesis -> minimal change -> registered experiment -> evidence review -> handoff/release audit.
- Before implementation, obtain user confirmation for the optimization direction and every affected existing file.
- Test one primary hypothesis per iteration. No unbounded HPO, silent retry, or treating unexecuted plans as evidence.
- Every handoff includes inputs, outputs, verification, limitations, next_action, repository-relative evidence paths, and the real executor.
- If one Agent performs multiple roles sequentially, label the review `non-independent`; never claim independent review.
- Keep facts, hypotheses, historical evidence, current validation results, and evidence gaps separate.

## 4. Competition and data invariants

- The primary metric is absolute Top5 portfolio return: buy at the first evaluation day's open and sell at the fifth evaluation day's open.
- Training, validation, features, and selection must not read market data on or after `2026-06-29`.
- The final holdout is report-only and cannot select directions, tune parameters, or choose checkpoints.
- Keep the complete-week training policy separate from the scoring formula. Validate negative paths for suspensions, missing weeks, disorder, duplicate keys, and invalid opens.
- Output UTF-8 `stock_id,weight`: 1-5 unique stocks, finite non-negative weights, total at most 1.
- Training and prediction are offline. Use repository-relative paths; never persist usernames or machine-specific absolute paths.
- Do not commit raw CSV, models, output, temp, dist, or release bundles.

## 5. Model and release gates

- Compare the clean-room official Transformer, XGBoost, and naive baselines on identical frozen data, rolling/nested folds, and an identical absolute-return MetricReport.
- Smoke tests prove runnability only. A single fold is not completion evidence.
- Propose a production-config change only after complete Pareto evidence and skeptic review pass; the actual edit still requires user confirmation.
- Before release, verify determinism, two offline runs, CSV, Docker, train/predict limits, size, and data/model hashes.
- Before a commit, run `python .agents/agentctl.py commit check` and use the repository's full Agent commit template.

## 6. Takeover notes at v1.0.1

- `.agents/` is currently untracked by Git. It can be inspected as the workspace control plane, but is not automatically part of version history.
- Some implementations and acceptance files referenced by V2 state may be absent from the current worktree. Reconcile code, tests, data manifest, and evidence reproducibility before continuing experiments.
- When machine state, prose claims, and actual files disagree, prefer runnable code and reproducible evidence. Record the gap without rewriting historical state.
