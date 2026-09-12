# Model CUDA kernels

`use_custom_kernels: true` requests the validated trio, with strict support checks:

| File | Operation | Implementation |
|---|---|---|
| `mlp.py` | Biasless SwiGLU projections and backward | Promoted `mlp_round2`; vendor GEMMs and recomputed activation in reused backward scratch |
| `lm_head.py` | Loss-only tied LM-head projection and cross entropy | Promoted `lm_head_round2`; 2,048-token chunks and FP32 weight-gradient accumulation |
| `rope.py` | Q/K bias, RoPE, independent V storage and packed QKV gradients | Promoted `rope_qkv_round2`; deterministic bias-gradient reduction |

The flag is independent of compilation and checkpoint mode. Top-level training settings override the architecture's
execution default and are written into `model_config.json`/HF config. Set it to `false` before model construction
for native operations. CPU inputs, missing Triton/CUDA support and unsupported layouts/dtypes raise
CustomKernelError with guidance to disable the flag. Loading and execution errors preserve their original cause;
no partially executed operation or microbatch is retried with native operations. MLP currently requires CUDA BF16 autocast; the LM head accepts supported BF16 inputs or
BF16 autocast. Requests returning logits and chunked validation loss retain their existing paths.

The LM-head forward computes cross-entropy entirely from maximum-shifted FP32 scores, including the target
subtraction. This preserves small losses under large finite common offsets (for example, equal logits of
either sign still yield `log(vocab_size)`). The BF16 projection rounding, padded-vocabulary denominator,
ignored-label mean behavior and backward are unchanged; differences already lost in BF16 are not recovered.

`runtime.py` loads implementations during construction, before compilation, without initializing CUDA. Operator
namespaces are stable per imported module and distinct for independent HF exports. All production dependencies
stay under `model/`; none import `tools/`. Model parameter/state-dict keys are unchanged. CPU or FP32 reference runs must explicitly disable the flag.

The benchmark harness starts with `use_custom_kernels=False` and applies only requested variants. Its round-two
registrations point to these promoted implementations, so baseline and single-variant comparisons stay isolated.
The revised `mlp_visible_forward` remains an experiment under `tools/` and is not enabled by this flag.
