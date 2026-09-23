# WORM FARM — Claim-Diverse Protocol v1
## Pre-Registration Addendum: Fixed-Rule Construction

This addendum is frozen before benchmark execution.

### Construction data

For each 6-task construction half:

- 15 fixed seeds per task.
- A balanced data-collection router selects AR/DIFFUSION by a deterministic seed/worm/generation hash.
- No Thompson outcome, no task-evaluation result, and no evaluation-task result is used to construct the rule.

### Static rule construction algorithm

For every specialist:

1. Count binary successes separately for AR and DIFFUSION.
2. Require at least 10 resolved observations in each mode.
3. Compute Beta(2,2) posterior means.
4. Compute `gap = posterior_AR - posterior_DIFFUSION`.
5. Keep only positive gaps.
6. Rank descending by gap.
7. Select at most the top **5** specialists as AR exceptions.
8. Every other specialist defaults to DIFFUSION.

The resulting exception set is frozen before evaluation of the held-out tasks.

### Evaluation

Each held-out task is evaluated with:

- Thompson Sampling using the empirical specialty×mode signal and normal v3.5.0 online adaptation.
- The frozen static rule above.

Both policies use the same seed list.

### Primary analysis

The independent inference unit is **task**, not decision.

For each task, the mean G7 `routing_mean_fitness` across the 15 seeds is computed.

The final comparison reports:

- 12 task-level paired differences;
- mean difference;
- median difference;
- paired Cohen's d_z;
- 95% paired bootstrap CI;
- task-level wins/losses/ties.

Acceptance requires:

- both bidirectional task folds complete;
- task hashes unique;
- no task appears in its own construction set;
- pooled task-level CI excludes zero in the predefined direction.

No policy is accepted on the basis of raw decision count.

### Important boundary

This benchmark tests task-level generalization only across the 12 predefined synthetic tasks. It does not establish generalization to arbitrary real-world tasks.
