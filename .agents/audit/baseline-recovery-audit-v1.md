# BDC2026 Baseline Recovery Audit v1.0.1

Status: evidence-gathering

Executor: Codex primary agent
Review independence: not reviewed

## Scope

Determine whether `BDC2026-test/` is a credible source for restoring the V2 baseline, locate the frozen market data, and separate verified facts from remaining evidence gaps. No training or production configuration changes were performed.

## Verified facts

1. The formal repository is on `dev` at `c73acb9`; `origin/dev` points to the same commit.
2. The formal worktree is dirty before recovery work: `AGENTS.md` is modified, while `.agents/.agents/` and `BDC2026-test/` are untracked user/workspace content.
3. `.agents/.agents/` is an exact 187-file duplicate of the outer `.agents/` tree.
4. `BDC2026-test/` is a 291-file snapshot without `.git`. It was copied into the workspace on 2026-07-13, while internal mtimes point to 2026-07-12 around 20:33 Asia/Shanghai.
5. The snapshot contains the missing XGBoost train/predict entrypoints, shared model spine, V2 experiment code, unit tests, acceptance pipeline, manifest, constituents file, and root readme.
6. All 78 Python files in the snapshot parse successfully.
7. Old XGBoost train/predict entrypoints are independently traceable to Git commits `bbb72cf` and `dbf01df`. V2 experiment code, tests, and manifest are not present in any current Git ref.
8. Five Loop-008 evidence hash mismatches are fully explained by LF/CRLF conversion: converting the snapshot LF bytes back to CRLF reproduces every indexed SHA-256.
9. The constituent snapshot mismatch is also fully explained by LF/CRLF conversion: CRLF normalization reproduces the manifest byte count (10451) and SHA-256 (`900f28ad...`).
10. The declared frozen market file is `data/stock_data.csv`, 28,555,122 bytes, 250,233 rows, 300 stocks, 2023-01-03 through 2026-06-26, SHA-256 `04d7e4da...`. The file is absent from the formal repository and the recovery snapshot.
11. No `stock_data.csv` was found in the repository parent tree, common user document/download/cloud locations, or the primary secondary-drive development/research/download directories checked. Broad whole-drive searches timed out and are not completion evidence.
12. Docker Desktop is not running, so no container/image data source could be inspected.

## Evidence gaps

- The original filesystem or archive source of `BDC2026-test/` is unknown.
- The frozen `stock_data.csv` has not been recovered or regenerated.
- The 11 shared files with semantic differences between the formal repository and snapshot require file-by-file review.
- The snapshot has no Git provenance and cannot be merged as a unit without reconstructing version history.
- Loop-008 must not be called validated until evidence is restored with a documented line-ending policy and rerunnable commands.

## Recovery decision

`BDC2026-test/` is a credible code recovery candidate, not a complete reproducible release. Its evidence content is internally consistent after line-ending normalization, but training and independent reproduction remain blocked by the missing frozen market CSV and missing provenance.

## Next action

Create a versioned recovery Goal, preserve Loop-001 through Loop-008 as read-only history, initialize a new Loop, then request explicit approval for the exact formal-repository files to restore or modify.
