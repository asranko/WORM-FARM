# WORM FARM v3.5.0 — Execution Bundle

This bundle contains the executed v3.5.0 patch set:

1. Tournament + softmax parent selection with fitness-magnitude sensitivity.
2. Empirical Bayesian routing signal with binary outcome semantics.
3. Deterministic self-development positive-control test.

Acceptance status:
- Fix 1: PASS
- Fix 2: IMPLEMENTED, ACCEPTANCE NOT MET
- Fix 3: PASS

The Adaptive Bias branch remains frozen and is not part of the mainline routing path.

The final integrated 10-seed × 8-generation run is included under `results/final_10x8_v350/`.
The lambda comparison is included under `results/selector_lambda_full_v350/`.
The empirical OOF evaluation is included under `results/empirical_routing_v350/`.
