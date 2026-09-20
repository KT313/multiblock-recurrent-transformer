# Execution precision

Training and validation use `model.execution.ExecutionPolicy` through the backend's `autocast()`
interface. `bf16-mixed` enters BF16 autocast; parameters remain FP32, while optimizer-state precision depends on
the optimizer. `32` uses a no-op context, preserving any ambient autocast supplied by a caller. Backward runs outside
the training forward context. Kernel enablement is independent and unsupported requests raise.
The backend precision property accepts supported reassignment and immediately updates its read-only execution
policy view. Missing precision is valid for legacy inference only; a training backend rejects it.

Sampling and benchmarks accept the optional keyword `execution_policy=ExecutionPolicy("bf16-mixed")`.
Repository training callers supply the backend policy. The complete inference operation runs in one
`evaluation.session.inference_session`: eval mode, inference mode, recurrence override and RNG state
are restored on exit, including errors. HFLM receives the corresponding `mixed_precision_dtype` because it opens an
inner autocast context for scoring and generation. For `32` or missing precision, HFLM disables its inner autocast context.
The session restores Python, NumPy global, Torch CPU and model-device CUDA RNG state; unrelated CUDA devices are not
initialized or reseeded. These process-global contexts must not overlap across threads.

Checkpoint evaluation reads `settings.precision`, with `--precision bf16-mixed` or `--precision 32` as an explicit
override. Missing precision metadata means caller-controlled execution (`ExecutionPolicy(None)`); it does not
infer precision from weights or custom-kernel enablement. Invalid stored metadata fails at policy resolution.
Sample decoding records and benchmark records include `execution_precision` when a policy is explicitly supplied.

HF exports optionally store `execution_precision` separately from architectural configuration. Direct HF and native
forwards are caller-controlled. For an exported model, explicitly enter its metadata-backed context:

```python
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
model.eval()
with torch.inference_mode(), model.execution_policy().autocast(model.device):
    output = model.generate(input_ids, max_new_tokens=32)
```

An export with missing metadata uses the same caller-controlled rule. Supply an explicit policy context to run
such an export with BF16. Do not cast the entire model to BF16 as a substitute: strict MLP kernels require active
CUDA BF16 autocast. The policy and session do not change the process's TF32 settings or initialize a backend.

Benchmark JSON also contains `execution_metadata` with `schema_version: 1`. Settings are recorded per
recurrence after the harness finishes: requested and configured batch size, retained automatic batch-size
selections, session and HFLM inner autocast dtypes, parameter dtypes, custom-kernel selection, context cap,
logits/request/response cache settings, and generation-cache/latent policy. Python callers can pass the harness's
`"auto"` or `"auto:N"` batch-size setting; the default is 8.

Observed harness settings carry a `source`. `source: "unavailable"` means the wrapper/harness did not
expose the value; it differs from an observed `null`, such as a disabled inner autocast dtype. Automatic
batch-size schedules may contain only the last scoring pass, so they are not a complete batch-length trace.
Generation records distinguish the wrapper's generation-config default from unobserved per-request
harness/task overrides and explain both cached persistent-latent and legacy uncached policies. An unset
wrapper config is not assumed to disable caching. Response-cache paths are reduced to an enabled flag.
Torch, Transformers, lm-eval and applicable loaded Triton distribution versions are recorded without
importing optional packages solely for metadata.

Both loss validation and inference sessions restore every incoming module's training flag, including a
child deliberately kept in eval mode inside a training model, on successful and exceptional exits. Loss
validation uses no-grad, backend autocast and forked Torch RNG.
