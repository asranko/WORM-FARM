# G7 Arm-Stratification Reconciliation Audit v1

## Finding 1 — same keys, different trajectories
The archived cost-parity `RUN_G7.csv` and the instrumented `G7_ARM_STRATIFIED_RUNS_V2.csv` have exactly 360 identical `(fold, task_id, seed, policy)` keys. However, **0/360 G7 values match exactly**. The instrumented pass is therefore a fresh execution, not a recomputation.

Mean instrumented-minus-archived change:
- Thompson: -0.000342584643
- Simple: -0.001775130505

This difference cannot be used to compare the two experiments as though they shared trajectories.

## Finding 2 — replay setting
The instrumented runner does not override `routing_replay_enabled`; the v3.5.0 `FarmConfig` default is `True`. It explicitly sets `routing_replay_max_per_generation=12`. The normal v3.5.0 runner also sets the same replay cap. The historical cost-parity runner source is not preserved, so replay equivalence between the archived cost-parity run and the instrumented rerun is **not independently provable**.

The project source states that routing counterfactual replay is an active path when `routing_replay_enabled` is true.

## Finding 3 — arm-conditioned metric definition
The instrumented runner computes full G7 from `generation_history[-1].routing_mean_fitness`. Separately, it filters routing records to generation 7, resolved, non-control decisions, obtains the corresponding worm, and averages that worm's `routing_fitness` by chosen mode.

Therefore the arm-conditioned quantity is **not the full-population G7 endpoint**. It is a decision-linked subset statistic.

The observed subset coverage is only about 13–16 decision-linked worms per run. Consequently a weighted mean of the arm-conditioned values (~0.40) is not expected to equal the full-population G7 (~0.425–0.428).

## Finding 4 — independence unit
The arm-conditioned `n` values (e.g. 1274) are counts of decision-linked rows nested inside 180 runs and 12 tasks. They are not independent samples. The current arm-conditioned report is therefore descriptive only. Any inferential comparison must use run- or task-clustered inference.

## Status
- Full endpoint equivalence between the two 360-run artifacts: **NO**.
- Historical Simple wrapper provenance: **UNVERIFIED**.
- Replay equivalence: **UNVERIFIED historically; current instrumented run = ON**.
- Arm-conditioned metric definition: **RESOLVED**.
- Raw arm counts may be reported descriptively; no naive significance claims.
- Reward-aligned Thompson remains blocked by the unresolved historical-run comparability.
