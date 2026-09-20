# Choosing Takase weight initialization

The Takase scale strategy supports two matrix distributions:

```yaml
# Model architecture YAML
init_strategy: takase
init_orthogonal: false
```

Or set the override in a **new run config**, keeping its other training settings the same:

```yaml
run_name: v2_big_1-4B_truncated_normal
resume: false
model_overwrite:
  init_orthogonal: false
```

`init_orthogonal: true` is the default. `false` selects
independent normal entries truncated at +/-3 standard deviations. It applies to Q/K/V, both fused MLP input
halves, attention/MLP output projections, adapters, embeddings and an untied head. A tied head shares the
initialized embedding parameter. Norm scales are initialized to one and biases (including Q/K bias) to zero.
Only actual booleans are accepted; the selector survives JSON/Hugging Face config round trips.

## Scales and bounds

Let `h` be hidden width and `L = prelude_layers + sum(core_layers[j] * mean_recurrence[j]) + coda_layers`.

| Parameter group | Normal std before truncation | Bounds |
|---|---|---|
| Embedding, head, adapter, Q/K/V, both MLP input halves | `sqrt(2/(5*h))` | `[-3*std, +3*std]` |
| Attention and MLP output projections | `sqrt(1/(5*h*L))` | `[-3*std, +3*std]` |

Embedding outputs are multiplied by `sqrt(h)`; logit scale remains 1. The depth in this formula is the expected
unrolled depth, including **every** recurrent core, not the unique physical layer count or differentiable tail.
The fused MLP gate and up halves both receive the base std, matching the upstream implementation's dispatch.

For example, with `h=2048` and `L=126`: base std is `0.01397542486`, bounds approximately
`+/-0.04192627458`; output std is `0.00088036902`, bounds approximately `+/-0.00264110705`.

Truncation reduces realized variance to approximately `0.973337 * std²` (std approximately `0.986578 * std`).
There is no variance-restoring rescale: this matches upstream. The bounds are multiples of the layer's std;
PyTorch's default absolute bounds `[-2,2]` would be a different initializer at these small scales.

## Reference and scope

The equivalent [upstream selection](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/recpre/init.py)
is `init_strategy="takase", orthogonal=False, truncate_normals=True, mup_model_scaling_factor=1`.
This specifies the weight initializer, not the full training recipe or identical whole-model initialization:
module counts, shapes and construction order differ. The effective-depth formula sums over all recurrent cores.

The recurrent initial-state distribution is separate: `state_init: normal` uses variance 1, not the paper's
reported state variance 2/5. The weight-init selector does not change optimizer, LR, precision, recurrence or normalization.

Use a fresh run name and `resume: false` for a new-initialization experiment. Loading a checkpoint restores its
saved weights; it is not a way to apply a new initialization to an already-trained model. No existing checkpoint
is reinitialized by this feature.
