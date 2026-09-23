# FIX 2 ACCEPTANCE — WORM FARM v3.5.0

## Patch
Replaced the NLL-derived routing score as the primary learned signal with an empirical Bayesian signal keyed by `(specialty, mode)` and updated only from the binary outcome:

`audit_pass AND survived_to_next_gen`

Cold start remains explicit until both modes reach the configured minimum observation count.

## Evaluation set
- 771 resolved historical routing decisions
- 10 original seeds
- 5 seed-grouped out-of-fold folds
- Beta(2,2) prior

## Proper Brier
- Empirical signal: **0.1661379786**
- Constant-rate baseline: **0.1712203299**
- Point difference (empirical - baseline): **-0.0050823513**
- Decision-bootstrap 95% CI: **[-0.0135443492, 0.0030860116]**
- Seed-block bootstrap 95% CI: **[-0.0126709188, 0.0014971076]**

## Acceptance criterion
Proper Brier must beat constant baseline with a bootstrap confidence interval excluding zero.

## Result
**FAIL / NOT ACCEPTED**

The point estimate is lower than baseline, but both bootstrap intervals cross zero. Therefore the evidence does not establish a statistically significant predictive improvement.

This is a finding about insufficient demonstrated predictive lift, not evidence that the empirical signal is useless in all future designs.
