# WORM FARM — Root-Cause Audit: Claim Independence Gap v1

## Scope

This audit answers one question:

> Did previous WORM FARM evidence reports explicitly establish claim/task diversity, or was independence at the claim level left implicit?

Reviewed:
- FIX 2 / EMPIRICAL SIGNAL acceptance
- SPECIALIST CONFOUNDING v37
- SIMPLE RULE / bidirectional held-out comparison
- WORM FARM v2 research protocol
- later Unique Claim Independence Audit

## Finding

**Claim/task diversity was not reported as an experimental-strength dimension in the prior acceptance reports.**

The reports explicitly recorded:
- number of seeds;
- number of generations;
- number of decisions / resolved decisions;
- seed-grouped evaluation where applicable;
- outcome definitions;
- bootstrap or paired statistics.

They did **not** require or report:
- number of unique claim texts;
- number of unique task instances;
- claim-to-seed mapping;
- whether a seed represented an independent task or repeated stochastic execution of the same task;
- a claim-level independence matrix.

## Evidence by report

### FIX 2

The acceptance report states:
- 771 resolved historical routing decisions;
- 10 original seeds;
- 5 seed-grouped OOF folds.

It does not state the number of unique claims or tasks.

The later audit established:
- 10 seed-specific claim IDs;
- **1 unique claim text**;
- the single claim text occurred across all 771 resolved records.

Therefore the prior report supported seed-level replication, but not claim-level generalization.

### SPECIALIST CONFOUNDING

The v37 audit states:
- 960 historical decisions;
- 771 resolved;
- specialist × resolution association;
- generation-stratified permutation testing.

It does not establish claim/task diversity.

Because the same historical book belongs to the fixed-task benchmark, the association is properly interpreted as within-task evidence unless a task-diversity artifact is independently established.

### SIMPLE RULE / Thompson comparison

The report explicitly defines:
- seed-level split;
- 8 generations;
- held-out evaluation across seeds.

It does not define a claim-level split.

The two halves therefore separate stochastic seeds, not unseen tasks.

### WORM FARM v2 protocol

The protocol requires reporting:
- generation;
- population;
- active population;
- findings;
- depth;
- births;
- arena matches;
- learning events;
- resilience;
- adaptation;
- genome diversity;
- strategy diversity;
- deterministic digest.

It does not require:
- unique claim/task count;
- independence unit;
- task diversity;
- seed × task matrix.

## Root cause

This was primarily a **protocol/schema omission**, not a statistical-computation error.

The review process had explicit controls for:
- ID uniqueness;
- leakage;
- calibration semantics;
- endpoint consistency;
- seed-level replication;
- bootstrap dependence.

It did not have an explicit gate asking:

> What is the independent experimental unit for the claim being made?

As a result, changing the seed count was implicitly treated as increasing evidence even when the underlying task/claim remained fixed.

## Why this survived multiple audits

The prior protocol was designed around a single synthetic colony/task and emphasized reproducibility across stochastic seeds.

That is valid for a narrow question such as:

> Does policy A behave differently from policy B across stochastic executions of one fixed task?

It is insufficient for:

> Does policy A generalize across different tasks/claims?

Because the protocol did not force this distinction into the report schema, the missing dimension was not surfaced.

## Classification

**Not an intentional single-task controlled design in the documentation.**

There is no reviewed report that explicitly declares:

> "This experiment intentionally fixes one task to isolate routing."

Instead, the safer classification is:

**unreported single-task structure / benchmark limitation / protocol omission.**

## Consequence

The following claims must be scope-limited:

- empirical signal: seed-replication evidence within a fixed task;
- Thompson vs Simple Rule: seed-held-out robustness for the same task;
- Thompson vs Random Exceptions: same fixed-task scope;
- specialist-resolution association: within-task association, not population-wide task generalization.

These results are **not automatically retracted**. They require scope narrowing and a claim-diverse follow-up before any claim of task-level generalization.

## Corrective rule

Every future experiment report must state, before any inferential statistic:

1. Independent stochastic units (for example seeds).
2. Independent task/claim units.
3. Number of unique task/claim instances.
4. Mapping between seeds and tasks.
5. Which dimension is held out.
6. Whether the primary claim is seed-level robustness, task-level generalization, or both.

No report may use “N decisions” as a proxy for independent task count.
