# WORM FARM — Claim-Diverse Benchmark v1 Preflight

## Power analysis

Prior seed-level paired effect:
`d_z = 1.92069`

For a paired task-level test with 12 tasks, alpha=0.05, the noncentral-t calculation gives:

- estimated power if the prior effect transferred unchanged: **1.0000**
- minimum n for 80% power at that prior effect: **5 tasks**

This does **not** establish that the task-level effect will be as large as 1.92. The prior effect was observed across stochastic seeds for one task. Therefore 12 tasks is explicitly classified as an **exploratory benchmark size**, not a guaranteed powered sample for smaller task-level effects.

## Task construction

12 tasks were authored with materially different reasoning structures:

- causal mediation
- counterfactual control
- boundary-condition failure
- hidden assumption detection
- anomaly detection
- evidence sufficiency
- mechanism discrimination
- competing explanations
- uncertainty calibration
- adversarial logical analysis
- hypothesis generation
- multi-step interaction inference

Every task has a unique SHA-256 text hash.

## Null similarity check

Exact SCMA similarity implementation:

- HashingVectorizer
- 2048 dimensions
- alternate_sign=False
- L2 normalized
- cosine similarity

All 66 unique task pairs were evaluated.

Results:

- mean cosine: **0.137991**
- median: **0.127604**
- min: **0.000000**
- max: **0.549442**
- std: **0.089741**
- fraction >= 0.50: **0.0152**
- fraction >= 0.70: **0.0000**

### Predefined gate

PASS if:

`mean random-pair cosine <= 0.50`

Observed:

`mean = 0.137991`

**Result: PASS**

## Important boundary

The benchmark may proceed only if the task text passes this gate.

No task is to be removed because it has an inconvenient outcome later. The task set is frozen before execution.

No Thompson, Simple Rule, Random Rule, or RAG run has been executed under this protocol.
