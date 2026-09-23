# WORM FARM — Experiment Independence Standard v1

## Mandatory reporting contract

Before a run is accepted, the report MUST declare the independent experimental units.

### Required fields

```yaml
stochastic_units:
  type: seed
  count: <integer>

task_units:
  type: claim | task | scenario
  count: <integer>

seed_task_mapping:
  description: <how seeds map to tasks>

primary_unit_of_inference:
  one_of:
    - seed
    - task
    - seed_within_task

holdout_dimension:
  one_of:
    - none
    - seed
    - task
    - both

task_reuse:
  allowed: true|false

task_text_hashes:
  required: true
```

## Mandatory pre-run gate

The following must be answered BEFORE execution:

```text
Q1: What is the claim/task being generalized over?
Q2: How many unique claims/tasks exist?
Q3: How many stochastic seeds exist per task?
Q4: Which unit is treated as independent for the planned inference?
Q5: Is the evaluation set unseen in task space, seed space, or both?
```

If any answer is missing, the experiment is **NOT READY**.

## Independence matrix

Every benchmark must be representable as:

```text
             Task 01  Task 02  Task 03 ... Task K
Seed 01          X
Seed 02          X
Seed 03                  X
...
Seed N
```

The report must state whether rows, columns, or both constitute independent replication.

## Claim-count rule

`decision_count` MUST NOT be used as a proxy for `unique_task_count`.

A report containing:

```text
3,284 decisions
30 seeds
```

must separately state:

```text
unique tasks = ?
```

## Generalization gate

A claim such as:

```text
"generalizes across tasks"
```

is prohibited unless:

```text
unique_task_count > 1
AND
evaluation tasks were not used to select the tested policy
```

For stronger claims, task-level held-out evaluation is required.

## Seed replication rule

Multiple seeds on one task are valid evidence for:

```text
stochastic robustness
```

They are NOT, by themselves, evidence for:

```text
task generalization
```

## Mandatory artifact

Each accepted experiment must produce:

```text
INDEPENDENCE_MANIFEST.json
```

containing:
- seed list;
- unique task IDs;
- task text hashes;
- seed × task assignments;
- train/build/evaluation partition;
- primary inference unit.
