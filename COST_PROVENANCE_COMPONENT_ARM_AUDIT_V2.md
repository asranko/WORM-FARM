# Cost Provenance / Reward / Arm Audit v2

## 1) `input_tokens` provenance — PASS

Direct inspection of the restored v3.5.0 source establishes this execution order inside `WormDualBrain.propose()`:

```text
text = _input_text(...)
ids = tokenizer.encode(text)
signal = score_sequence(model, ids, ...)
router.select(..., input_tokens=len(ids), ...)
```

`score_sequence()` calculates both `ar_signal` and `diff_signal` before the router is called. Therefore `input_tokens` is derived from the complete input sequence **before the policy selects the arm**. It is not a Thompson-only posterior/decision overhead.

Router accounting is explicitly:

```text
cost_units = max(0.1, input_tokens / 64.0)
```

The restored v3.5.0 code therefore supports the semantic basis of cost parity **when both policy wrappers use the same `WormDualBrain` invocation path**.

Caveat: the historical claim-diverse Simple wrapper source is not present in the restored v3.5.0 package, so that wrapper's historical call path cannot be independently proven from the package alone.

## 2) Component recomputation — PASS

Full-precision weighted recomputation reproduces the stored rewards:

- Thompson recomputed = `0.157317348247149`
- Thompson stored = `0.157317348247149`
- Simple recomputed = `0.151360061913767`
- Simple stored = `0.151360061913767`

Match within `2e-12`:

```text
Thompson = True
Simple   = True
```

No extra Jensen/clamping explanation is needed for the aggregate table discrepancy: the full-precision component means already reconstruct the stored reward.

## 3) Arm-level decomposition available from the stored artifact

The artifact contains descriptive reward aggregates conditional on selected arm:

```text
  policy      mode     n    audit  survival  signature  robustness     cost   reward
  simple        AR  6748 0.139004  0.139004   0.022411    0.859807 0.999241 0.183642
  simple DIFFUSION 14490 0.069151  0.069151   0.020080    0.830595 0.999594 0.136327
thompson        AR 10023 0.106056  0.106156   0.020767    0.846982 0.999386 0.161419
thompson DIFFUSION 10974 0.096683  0.096683   0.020732    0.834205 0.999712 0.153571
```

The arm-weighted rewards reconstruct the policy-level reward means:

- Thompson = `0.157317348247149`
- Simple   = `0.151360061913767`

However, `RUN_G7.csv` has only:

```text
fold, task_id, seed, policy, g7_routing_mean_fitness
```

and contains no selected-mode field. Therefore the exact requested **G7-by-arm** quantities cannot be recovered from this artifact.

## 4) Final status

```text
Input-token provenance     PASS
Reward recomputation       PASS
Arm-level reward data      AVAILABLE (descriptive)
G7-by-arm                  NOT AVAILABLE
```

No Reward-aligned Thompson experiment is authorized based on this audit alone.
