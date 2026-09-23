# G7-by-Arm Stratified Diagnostic — Final Status

## Execution
- 360/360 unique `(fold, task, seed, policy)` runs completed.
- v2 schema only; legacy rows were removed and rerun.
- Arm conditioning uses resolved, non-control generation-7 routing decisions.

## Historical wrapper provenance
The restored project contains no `.git` history and no historical Simple wrapper source. The current source proves the meaning of `input_tokens`, but the historical wrapper's exact call path cannot be reconstructed byte-for-byte.

## Full test suite
`63 passed, 2 warnings, 38.13s` in the final full-suite run.

Warnings are SWIG deprecation warnings.

## Arm-conditioned results
The arm-conditioned statistic is the mean `routing_fitness` of worms that had a resolved non-control generation-7 routing decision with the indicated chosen mode. It is not identical to the full-population `G7 routing_mean_fitness` endpoint because other worms can be in cooldown / lack a resolved G7 decision.

Thompson:
- AR decision-linked n = 1274
- AR weighted mean = 0.3981506800
- AR run SD = 0.0090293216
- DIFFUSION decision-linked n = 1341
- DIFFUSION weighted mean = 0.4020393192
- DIFFUSION run SD = 0.0103584870

Simple:
- AR decision-linked n = 502
- AR weighted mean = 0.3938440671
- AR run SD = 0.0167999545
- DIFFUSION decision-linked n = 2125
- DIFFUSION weighted mean = 0.3964450367
- DIFFUSION run SD = 0.0066616122

Arm mix:
- Thompson: AR 48.72%, DIFFUSION 51.28%
- Simple: AR 19.11%, DIFFUSION 80.89%

Descriptively, Thompson's conditional mean is higher within both AR and DIFFUSION. The Simple Rule's larger AR share is therefore not the source of a higher conditional arm fitness in this rerun. Because arm assignment is policy-dependent, these conditional comparisons are descriptive rather than causal arm effects.

## Final interpretation
The requested G7-by-arm diagnostic does not support the hypothesis that the observed policy difference can be explained merely by Simple obtaining a rare, unusually high AR conditional fitness. In this rerun, Simple's AR conditional fitness is lower than Thompson's, and its AR cell also has substantially higher run-to-run variance with fewer observations.

No Reward-aligned Thompson experiment was started from this result.
