# FIX 1 ACCEPTANCE — WORM FARM v3.5.0

## Patch
Replaced deterministic greedy parent selection with tournament selection using softmax over fitness differences normalized by tournament spread.

## Exact run
- Same 10 seeds: 42, 77, 101, 113, 211, 313, 401, 509, 607, 709
- 8 generations
- Same WormDualLM checkpoint
- Same empirical-routing configuration
- Only `routing_fitness_lambda` changed: 0.0 vs 5.0
- Self-development disabled in this causal isolation run

## Acceptance criterion
Parent-pair identity must be different for at least one seed.

## Result
- Identical seeds: **2/10**
- Different seeds: **8/10**
- Acceptance: **PASS**

The selector therefore responds to fitness magnitude under the patched mechanism. This does not by itself establish a long-run diversity benefit.
