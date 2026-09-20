# Model CUDA kernels

`use_custom_kernels: true` enables these kernels with strict support checks:

| File | Operation | Implementation |
|---|---|---|
| `mlp.py` | Biasless SwiGLU projections and backward | Vendor GEMMs and recomputed activation in reused backward scratch |
| `lm_head.py` | Loss-only tied LM-head projection and cross entropy | 2,048-token chunks and FP32 weight-gradient accumulation |
| `rope.py` | Q/K bias, RoPE, independent V storage and packed QKV gradients | Deterministic bias-gradient reduction |

The flag is independent of compilation and checkpoint mode. Top-level training settings override the architecture's
execution default and are written into `model_config.json`/HF config. Set it to `false` before model construction
for native operations. CPU inputs, missing Triton/CUDA support and unsupported layouts/dtypes raise
CustomKernelError with guidance to disable the flag. Loading and execution errors preserve their original cause;
no partially executed operation or microbatch is retried with native operations. MLP currently requires CUDA BF16 autocast; the LM head accepts supported BF16 inputs or
BF16 autocast. Requests returning logits and chunked validation loss retain their existing paths.

The LM-head forward computes cross-entropy entirely from maximum-shifted FP32 scores, including the target
subtraction. This preserves small losses under large finite common offsets (for example, equal logits of
either sign still yield `log(vocab_size)`). The loss uses BF16 projections, normalizes over the padded vocabulary,
and excludes ignored labels from the mean. Differences already lost in BF16 are not recovered.

`runtime.py` loads implementations during construction, before compilation, without initializing CUDA. Operator
namespaces are stable per imported module and distinct for independent HF exports. All production dependencies
stay under `model/`; none import `tools/`. Enabling the kernels does not change model parameter/state-dict keys.
CPU or FP32 reference runs must explicitly disable the flag.
