# WORM FARM — Claim-Diverse v1 Result Review v2

## Scope
This review audits the task-diverse Thompson-vs-Simple experiment before any final scientific interpretation.

## 1. Raw G7 variance
Across all 180 task×seed runs per policy:

- Thompson mean G7 = 0.4287957422
- Thompson SD = 0.0042459911
- Thompson range = [0.4190079135, 0.4369031599]
- Simple mean G7 = 0.4346698419
- Simple SD = 0.0034948758
- Simple range = [0.4271257784, 0.4417317597]

Pooled within-task SD:
- Thompson = 0.0013363947
- Simple = 0.0011541561

Thus the raw metric is not variance-free.

## 2. Why task-level d_z is huge
The primary task-level effect is based on 12 paired task means.

- mean task difference (Thompson − Simple) = -0.0058740997
- SD of 12 task differences = 0.0008621319
- d_z = -6.8134582

The effect is large because the policy gap is unusually stable across the 12 task means, not because the underlying G7 metric has zero stochastic variance.

The 180 matched seed-task differences have:
- mean = -0.0058740997
- SD = 0.0018987807
- d_z at seed-task unit = -3.0936

These are descriptive diagnostics only; the task-level analysis remains the appropriate unit for task-generalization claims.

## 3. Wilcoxon interpretation
The one-sided exact Wilcoxon p-value with 12 pairs and identical sign is:

`1/2^12 = 0.000244140625`.

This is the minimum possible one-sided exact Wilcoxon p-value for n=12 with no ties. It encodes the 12/12 directional consistency and should not be presented as independent evidence of effect magnitude.

## 4. Fold dependence
Fold A and Fold B reuse the same 12 tasks and only swap construction/evaluation roles.

Therefore they are complementary cross-validation folds, not independent samples from an external task universe.

The valid claim is:

> The observed policy ordering was stable under both directions of task-level holdout within this predefined 12-task benchmark.

## 5. Task-regime structure
All 12 tasks were run under the same WORM FARM mechanics:

- 2 arms: AR and DIFFUSION
- 8 generations
- 16 maximum findings per generation
- 128 routing decisions per run
- same evaluator/environment configuration
- same routing control fraction
- same reward weights
- same model checkpoint

The tasks differ in claim content/reasoning structure, but the benchmark does not vary bandit-regime parameters such as arm count, horizon, or environment stationarity.

Therefore this is:

**content-diverse / fixed-regime task evaluation**, not a broad bandit-regime generalization study.

## 6. Horizon
Exact routing opportunities are:

- 8 generations × 16 findings/generation = 128 routing decisions per task×seed×policy run.
- With a 10% control cohort, the nominal non-control routing opportunity is approximately 115 decisions/run, but the exact control count varies by stable hash assignment.

This is not an extremely short total horizon at the run level. However, the routing state is contextual and partitioned by specialty/depth/generation/genome/strategy in the underlying router, so effective exposure per contextual state is much smaller.

Importantly, when the empirical signal is ready, selection uses the `(specialty, mode)` empirical posterior rather than the contextual `arms` posterior.

## 7. Prior configuration
Two prior layers exist:

### RoutingBandit internal posterior
`Beta(1,1)` by default in `FarmConfig`.

### Empirical routing signal
`Beta(2,2)` smoothing prior.

For the task-held-out experiment, the evaluation Thompson router is initialized with the construction-half empirical state:

`alpha = 2 + successes`
`beta  = 2 + failures`

and can continue updating online during the held-out task.

Therefore the task-diverse test is not a pure uninformed cold-start test. It is a **cross-task empirical-prior transfer + online adaptation** test.

## 8. Outcome collapse warning
Every one of the 360 evaluation run summaries satisfies:

`positive_outcomes == round(audit_pass_rate × resolved)`

with no nonzero discrepancy.

Thus, in the executed workload, every resolved decision that passed audit also survived to the next-generation boundary. Consequently:

`audit_pass AND downstream_survival`

collapsed empirically to audit-pass for this run.

This does not prove the survival term is redundant in general, but it means survival did not discriminate outcomes in this benchmark.

## 9. Endpoint alignment warning
The primary endpoint `G7 routing_mean_fitness` is a smoothed worm-state quantity. It is not the same variable as the binary routing outcome.

The current data show:

- Thompson has a higher positive-outcome rate than Simple on **all 12 tasks**.
- Simple nevertheless has a higher G7 routing_mean_fitness on **all 12 tasks**.

This is a critical endpoint-alignment signal.

Therefore the statement:

> "Thompson performs worse"

is not justified without specifying the endpoint. The supported statement is:

> "Under the current G7 routing_mean_fitness endpoint, Simple exceeds Thompson in all 12 tested task instances, while Thompson simultaneously records higher binary positive-outcome rates."

## 10. Scientific status
The original task-level result should not be generalized to:

- universal Thompson failure;
- all bandit horizons;
- all priors;
- arbitrary task distributions;
- different reward/evaluator definitions.

A horizon extension is a valid mechanistic follow-up, but the more immediate unresolved issue is endpoint alignment: the binary routing outcome and the evolutionary G7 fitness endpoint currently disagree in direction.

## 11. Recommended next diagnostic
Before treating the current result as a clean bandit-regime conclusion, run a separately preregistered horizon ablation with fixed tasks and policies (e.g. current 8-generation horizon versus a longer horizon) and report both:

- G7 / final routing_mean_fitness;
- cumulative positive outcome rate.

The purpose is diagnostic, not to rescue the original result. If the two endpoints converge in direction with horizon, the current gap is a horizon/regime effect. If they remain opposed, the primary endpoint is misaligned with routing success and should not be used alone to judge routing policy quality.
