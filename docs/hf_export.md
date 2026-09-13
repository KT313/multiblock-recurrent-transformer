# Hugging Face export

`export_to_hf` saves weights, source modules, model configuration and generation configuration for offline
`AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)` loading. Training exports already supply the
training dataset's tokenizer directory.

```python
from model.hf import export_to_hf

export_to_hf(model, model.config, "export", tokenizer_dir="dataset/tokenizer")
# Or reuse an already loaded transformers tokenizer:
export_to_hf(model, model.config, "export", tokenizer=tokenizer)
# Explicit metadata without a tokenizer; BOS and PAD are optional:
export_to_hf(model, model.config, "export", eos_token_id=[2, 3], bos_token_id=1, pad_token_id=0)
```

EOS metadata is required by default so reloaded generation can stop without repeating token IDs. Supply either
`tokenizer_dir` or a `PreTrainedTokenizerBase` instance. The original positional tokenizer-directory argument remains
supported; all new options are keyword-only. `None` means unspecified and does not clear tokenizer metadata.
Explicit IDs must agree with existing tokenizer IDs. Missing tokenizer special-token roles can be supplied explicitly:
export fills them on a copy using existing vocabulary tokens, leaving the source tokenizer untouched.

IDs must be integers in `[0, model.config.vocab_size)`; booleans, fractional numbers, negative values and padded-only
embedding rows are rejected. Tokenizer exports also reject unknown tokenizer IDs. EOS accepts a scalar or nonempty
list. A singleton list agrees with the corresponding tokenizer scalar; multiple EOS IDs are supported for explicit-only
exports because a tokenizer has one EOS role. BOS/PAD may remain absent, and export never substitutes PAD=EOS or resizes
the vocabulary. Invalid metadata fails before creating or overwriting output artifacts.

When exporting without a tokenizer, use a destination without existing tokenizer artifacts. Reusing a directory from
a tokenizer-backed export would leave its old tokenizer next to the new model metadata, so export rejects that reuse
before accessing weights or writing files. Use a fresh directory or supply the tokenizer explicitly. Existing tokenizer
files are never silently deleted; this guard also applies to intentional model-only exports.

For an intentional model-only export without EOS stopping, opt in explicitly:

```python
export_to_hf(model, model.config, "weights_export", allow_missing_generation_metadata=True)
```

This opt-in preserves absent IDs as `None`; generation callers must supply stopping metadata if they need EOS stopping.
It does not bypass validation of supplied IDs. This is a deliberate compatibility change for callers that previously
exported without any tokenizer or token metadata. Optional [execution precision](execution_precision.md), native model
weights and strict custom-kernel behavior are unchanged.
