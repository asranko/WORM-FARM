# G7-by-Arm Diagnostic — Execution Status

## Historical source check
- `.git` directory: **ABSENT**
- Therefore the original Claim-Diverse Simple wrapper cannot be recovered from repository history in this snapshot.
- No claim is made that the current diagnostic reproduces the historical wrapper byte-for-byte.

## Instrumented diagnostic
The experimental runner `diagnostics/g7_arm_stratified_v1/run_g7_arm_stratified.py` was created outside `worm_farm/*.py`.

Target:
- 12 tasks
- 15 seeds/task
- 2 policies
- G7 endpoint
- chosen mode per decision
- arm cell n and SD

## Harness failures rejected
1. Initial worker import path failure: `ModuleNotFoundError` — no scientific result.
2. Asset path failure in diagnostic runner — no scientific result.
3. Parallel execution attempts exceeded runtime limits; completed runs were stored, but incomplete batches were not interpreted as a final experiment.

## Replay-off equivalence test
A two-seed smoke comparison was performed with routing counterfactual replay disabled.
The resulting G7 values differed from the archived cost-parity values for the same seed/task/policy.
Therefore replay is **not proven to be observationally inert** under this environment.

This optimization is rejected and will not be used for the final diagnostic.

## Current valid diagnostic rows
Only the final replay-on smoke rows are retained in the directory (4 rows: T01 / seeds 42,77 / both policies).
They are **not sufficient** for arm-stratified final inference.

## Current verdict
- Input-token provenance: **previously verified on current source**.
- Reward arithmetic: **previously verified**.
- Historical Simple wrapper provenance: **UNVERIFIED** (no git/history artifact).
- G7-by-arm: **NOT COMPLETE**.
- Full pytest in the latest verification attempt: **TIMEOUT**, so not reported as PASS.
- Mainline `worm_farm/*.py` was not modified by the diagnostic work.
