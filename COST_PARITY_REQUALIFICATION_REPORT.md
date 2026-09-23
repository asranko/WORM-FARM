# WORM FARM — Claim-Diverse Cost-Parity Requalification v1

## Critical finding
The previous Claim-Diverse v1 Simple Rule wrapper used `cost_units=1.0`, while the main Thompson router used `max(0.1, input_tokens/64.0)`. Because normalized cost enters the routing reward with a negative weight of 0.05, the two policies were not evaluated under symmetric cost accounting.

This is an experimental bookkeeping defect. The WORM FARM mainline was not modified.

## Corrective experiment
Exactly one variable was changed: the Simple Rule wrapper now uses the same cost calculation as Thompson. Tasks, seeds, construction artifacts, policy exceptions, Thompson empirical priors, reward weights, environment, checkpoint, and 8-generation protocol were held constant.

## Corrected result
- Thompson mean G7: **0.428795742**
- Simple mean G7: **0.426972967**
- Mean task-level difference (Thompson - Simple): **0.001822775**
- Task-level SD of differences: **0.000748875**
- Cohen d_z: **2.4340**
- 95% paired CI: **[0.001346963, 0.002298588]**
- Task wins: **12/12**
- Task losses: **0/12**

## Reward-component decomposition
| Component | Thompson | Simple | Δ (T−S) |
|---|---:|---:|---:|
| Audit pass | 0.101157 | 0.091346 | +0.009812 |
| Survival | 0.101205 | 0.091346 | +0.009859 |
| Signature delta | 0.020749 | 0.020821 | -0.000072 |
| Robustness | 0.840304 | 0.839876 | +0.000427 |
| Normalized cost | 0.999556 | 0.999482 | +0.000075 |
| Composite reward | 0.157317 | 0.151360 | +0.005957 |

## What caused the old reversal?
Before cost parity, the Simple Rule had mean normalized cost near 0.25 while Thompson was near 1.0, creating an artificial reward advantage of approximately **0.0375** for Simple from the cost term alone. That penalty gap was much larger than the old G7 policy gap (~0.0059), so the bookkeeping asymmetry was sufficient to reverse the policy ordering.

After correction, normalized cost is approximately identical between policies (~1.0). The remaining reward difference is driven primarily by audit/survival; signature difference is slightly negative for Thompson and robustness slightly positive.

## Objective-alignment finding
A genuine architectural mismatch remains: when the empirical signal is ready, Thompson selection uses the `(specialty, mode)` empirical Beta posterior built from binary `audit_pass AND survival`, while `routing_fitness` is generated from the five-component composite reward.

However, after fixing cost parity, Thompson is higher on both the direct binary outcome and the composite reward/G7 endpoint in this benchmark. Therefore the objective mismatch is a **design risk to test later**, not the explanation for the previous Simple>Thompson reversal.

## Arm composition
Thompson and Simple use different AR/DIFFUSION mixtures. This is expected because the policies select different modes. Conditional component differences are therefore descriptive, not a causal arm-effect estimate.

## Survival field check
The instrumentation pass found **1 survival/audit mismatch among 42235 resolved records**. Thus the earlier exact equality was a property of the prior aggregate artifacts, not a guaranteed invariant of the implementation.

## Status
- Previous Claim-Diverse v1 negative conclusion on G7: **INVALID / REQUALIFIED** due cost-accounting asymmetry.
- Cost-parity corrected task-level comparison: **Thompson > Simple in all 12 tasks in this rerun**.
- This does not establish universal optimality or broad bandit-regime generalization.
- Horizon expansion remains **blocked** until separately justified.

## Integrity
- 360/360 evaluation runs complete.
- 42,235 resolved component records.
- Stored reward == recomputed reward: exact in the component instrumentation pass.
- Mainline tests: **63 passed, 2 warnings**.
- `compileall`: **PASS**.
- `worm_farm/*.py`: unchanged.
