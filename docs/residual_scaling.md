# Fixed residual branch scaling

Enable in the **model architecture YAML**:

```yaml
residual_scaling: inverse_sqrt_depth
```

Or use a model override in a new **top-level run config**:

```yaml
run_name: v2_big_1-4B_residual_scaling
resume: false
model_overwrite:
  init_orthogonal: false  # retain this if comparing against the truncated-normal run
  residual_scaling: inverse_sqrt_depth
```

The default is `residual_scaling: none`. Only `none` and `inverse_sqrt_depth` are accepted. All shipped model
templates keep the default off. No LR, optimizer, normalization, initialization or recurrence-sampling setting
is changed by enabling this feature. The new example retains truncated-normal initialization as an explicit
choice; keep all other settings/data fixed for a one-factor comparison.

## Formula and placement

```text
L = prelude_layers + sum(core_layers[j] * mean_recurrence[j]) + coda_layers
alpha = 1 / sqrt(L)

u = norm2(x + alpha * Attention(norm1(x)))
y = norm4(u + alpha * MLP(norm3(u)))
```

`L` counts complete transformer layers, not individual residual additions: the audited 1.4B architecture has
`L = 3 + 3*5*8 + 3 = 126`, so `alpha = 0.08908708064`. The same coefficient applies to both sublayers of
**every prelude, core and coda sandwich block**. It is a fixed Python float, not a trainable gate or new weight.
No coefficient is applied to the skip input, the adapter, or the outer whole-core addition `x_next = x + s`.
The multiplication is out of place, after the attention/MLP projection and before the residual sum and outer norm.

The coefficient is computed once at construction using the configured expected depth. Changing sampled training
depth or requested evaluation/generation depth does not change it. This makes recurrence-depth comparisons use
the same recurrent map. Effective depth already includes all core blocks; do not double it for attention/MLP.

Scaling is additional to the existing small Takase output-projection initialization. It may also suppress useful
learning, so judge it by loss, token dispersion and recurrence gain, not correlation alone. The outer norms remain
in place: reducing these branches does not make the complete network an identity or directly attenuate the final
normalized core state added to the outer stream. Existing BF16 rounding behavior is retained; this option does
not introduce FP32 accumulation.

## Execution, logging and compatibility

All paths use `SandwichBlock.forward`: training/checkpoint recomputation, ordinary evaluation, cache-based
generation, Hugging Face export, and the separate representation probe. When disabled, the forward omits both
multiplications entirely; initialization, tensor layouts and random-number consumption remain unchanged.
Custom attention/MLP kernels still produce their ordinary branch outputs; scaling occurs afterwards in the
shared block code. Actual CUDA/custom-kernel integration and performance were not executed for this change.

`residual_scaling` is saved with the model config and survives HF/JSON round trips. The resolved coefficient is
logged as `recurrence_probe/residual_scale` whenever a document probe is available. Detailed attention/MLP
correlations still observe raw module outputs, before the new scalar multiplication. Adapter merged/core-output
metrics and representation summaries observe the actual scaled network trajectory.

Older model configs/checkpoints missing the field mean `none`; no checkpoint rewrite is needed. Enabling scaling
on a resume changes the model's forward function even though the tensor shapes are unchanged. The existing resume
policy therefore reports it as a model-config difference. For this experiment, use a fresh run name and
`resume: false`; no trained weights are automatically reinitialized or altered.

## Verification scope

Bounded CPU tests cover independent explicit sandwich outputs/gradients in FP32/BF16, AOT-eager compiled block
forward/backward, unchanged initialization/RNG/outer skips, full/selective checkpoint recomputation, cached
generation versus fixed-latent scoring, configuration/export metadata and legacy checkpoint checks, diagnostic
read-only guarantees, and the original default-mode golden optimizer-step reference. A separate probe test
compares the scaled diagnostic with the normal model forward using identical initial states.

These are tiny-model correctness checks, not evidence that scaling resolves the 1.4B collapse. No large model,
checkpoint load, GPU computation, remote deployment or training job was launched.
