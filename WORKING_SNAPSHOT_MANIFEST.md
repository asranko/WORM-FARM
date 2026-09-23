# WORM FARM v3.5.0 — Working Complete Snapshot

Snapshot date: 2026-09-23

## Source of truth

The canonical mainline is the extracted contents of:

`WORM-FARM-v3.5.0-FINAL.zip`

The `worm_farm/` source tree in this snapshot is preserved from that canonical package.

## Experimental overlay included

This snapshot additionally contains the latest research artifacts from the current working cycle:

- `diagnostics/g7_arm_stratified_v1/`
- `results/claim_diverse_v1_execution/`
- `results/claim_diverse_cost_parity_v2/`
- `research_protocols/claim_diverse_v1/`
- `research_protocols/independence_v1/`

These overlays are experimental/evidentiary artifacts. They are not silently promoted into the mainline source.

## Mainline integrity

No file under `worm_farm/` was modified while assembling this snapshot.

The current research state includes unresolved provenance for the historical Claim-Diverse Simple wrapper and replay configuration used by the original cost-parity run. This is explicitly recorded rather than inferred away.

## Important status boundaries

- Thompson mainline status is subject to the scope and endpoint qualifications in the included reports.
- G7-by-arm results are descriptive/stratified observations, not causal arm effects.
- Reward-aligned Thompson remains experimental and not included as a mainline change.
- Adaptive Bias remains frozen.
- RL and Meta-Routing remain closed.

## Verification performed for this snapshot

- File inventory generated.
- Mainline source tree preserved from canonical package.
- `pyproject.toml` and wheel preserved.
- Experimental overlays copied without modifying `worm_farm/`.
