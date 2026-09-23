# FINAL INTEGRATED RUN — WORM FARM v3.5.0

## Configuration
- Tournament parent selector enabled
- Empirical routing signal enabled
- `routing_fitness_lambda=0`
- Adaptive Bias branch excluded/frozen
- Self-development runtime disabled for isolation; its deterministic positive-control is tested separately
- WormDualLM checkpoint: `wormdual_192x4.pt`
- CPU
- Seeds: 42, 77, 101, 113, 211, 313, 401, 509, 607, 709
- 8 generations each

## Aggregate
- Decisions: **1,280**
- Resolved: **1,099**
- Control decisions: **127 (9.921875%)**
- AR decisions: **667**
- DIFFUSION decisions: **613**
- Positive binary outcomes: **167**
- Every run completed

## Diversity note
The generation-history field remained numerically constant in this run because the population reached 30 genomes early and the same stored population was used by the metric at later boundaries. This run therefore does **not** establish a G0→G7 diversity effect.

## Status
This integrated run verifies that the three patches coexist and execute together. It does not override the failed statistical acceptance of Fix 2.
