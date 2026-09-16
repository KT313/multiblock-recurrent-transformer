# Llama 32K tokenizer with explicit chat roles

Opt into `llama32k_chat_v1` for a **new run**. It retains all 32,000 existing token IDs and adds `<user>` (32000) and `<assistant>` (32001). BOS/EOS/UNK remain 1/2/0. Models must declare `vocab_size: 32002`; physical embedding/head padding remains separate (for example 32768 rows). Old checkpoints cannot resume with this new profile, even when their padded tensors have the same shape.

```yaml
tokenizer:
  name: llama-32k-chat-v1
  kind: hf
  hf_id: hf-internal-testing/llama-tokenizer
  revision: d02ad6cb9dd2c2296a6332199fa2fdca5938fef0
  profile: llama32k_chat_v1
```

Preparation verifies and reuses an existing `dataset/tokenizers/llama-32k` base offline, or obtains the pinned tokenizer through HF. It leaves that base folder unchanged. The extended folder stores its adapter, template and fingerprint. The profile fixes the Transformers 5 Llama conversion of the pinned payload; an incompatible change in tokenizer conversion fails explicitly rather than silently changing tokenization.

The dataset example is `config/datasets/instruction_sources_smoke.yaml`; its matching tiny-model run is `config/instruction_sources_smoke.yaml`. Prepare the dataset explicitly before launching that run. Existing research configs are unchanged: update both the dataset tokenizer and model real vocabulary deliberately when starting a new run.

## Literal content and chat structure

Readable chat format:

```text
<s><user>question</s><assistant>answer</s><user>next question</s><assistant>next answer</s>
```

Every message body is tokenized as **literal text**, including occurrences of `<s>`, `</s>`, `<user>`, `<assistant>`, and `<unk>`. No sample is filtered for containing these strings. Only the formatter inserts structural IDs. Chat-body encoding disables the automatic leading-space prefix, so role headers do not invent whitespace; ordinary pretraining tokenization retains its existing prefix behavior. Message roles and boundaries determine labels, so a literal `</s>` in an answer cannot stop that answer or split the conversation.

Assistant bodies and their structural EOS are supervised. User bodies, user EOS, role markers and BOS are masked. Whole exchanges are retained within the token budget; trailing users and exchanges that do not fit are removed with the existing counters. A conversation is one packed document, including its internal EOS tokens. Pretraining remains BOS + literal document text + EOS.

Every instruction source using this profile must explicitly select `instruction_format: messages`. Use an existing message converter, or use `fields: {instruction: question_column, output: answer_column}` without a converter. An optional input field is appended to the instruction with a blank line. Inversions remain unsupported for message sources. System/tool roles remain unsupported; the existing Nemotron source policy handles empty/nonempty system prompts before formatting.

## Hugging Face use

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(export_directory, trust_remote_code=True)
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain the literal string </s>."}],
    tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
)
```

The saved **tokenizer adapter is required** to preserve literal text. `apply_chat_template(tokenize=False)` is a readable rendering only: the rendered string cannot distinguish structural markers from identical message content. Do not re-encode it as a chat prompt. Plain `encode`/`__call__` intentionally encode every spelling literally (with an automatic BOS when requested). Use the structured `apply_chat_template(..., tokenize=True)` API for chat. Arbitrary templates, tools, assistant prefill and automatic chat truncation are rejected; shorten complete exchanges explicitly. The adapter and its dependencies are saved with exports, with no repository imports required. This is compatible with Transformers custom-tokenizer loading, not template-only consumers that ignore the adapter.

For training-style masks, pass `return_dict=True, return_assistant_tokens_mask=True`. Padding is not supervised; HF batching requires an explicit padding token (EOS may be used). The project's generation code already uses EOS as its padding fallback and supplies an attention mask.

## Validation, checkpoints and benchmarks

Training checks configuration before automatic preparation, then compares the HF adapter and actual training formatter on small CPU probes before constructing the model. It checks real/physical vocabulary, actual embedding/head dimensions and tying, and verifies tokenizer agreement across ranks. These checks do not run per microbatch. Training's existing padded-column loss normalization is preserved.

Each new-profile run retains `outputs/<run>/tokenizer/` and embeds its contract in checkpoints. Evaluation resolves that artifact independently of subsequent edits to the dataset YAML. Move the complete run directory, or copy the matching tokenizer and pass `--tokenizer_dir` when evaluating an isolated checkpoint. Changing either token IDs or formatting is not an allowed settings override. Profile data has a distinct raw/processed identity; old counts cannot simply be relabeled. Any required data rebuilding uses the existing explicit preparation/confirmation flow.

Plain benchmark prompting remains the default. Opt into chat with `benchmark_apply_chat_template: true` in training settings or `--benchmark_apply_chat_template` on the evaluation CLI. The project's lm-eval adapter transports structured messages through its string request interface and encodes candidate answers literally. It does not feed that transport representation to the model. Standard third-party HFLM used directly with a flattened chat string does not provide this guarantee. Keep plain/chat scores separate; the selected protocol and tokenizer contract are recorded in benchmark metadata.

Sample generation uses `training.data.formats.encode_generation_prompt`, which shares the training chat encoder. With this profile, built-in instruction samples and `# instruction` prompts automatically use the chat template; `# continuation` stays BOS + literal text, without a closing EOS so generation continues the document. `# chat` is also supported, and `Prompt(..., kind="chat", messages=[...])` accepts multi-turn history. Legacy tokenizer runs retain their historical instruction prefix. The human-readable prompt is logged separately from its actual templated input.

Focused offline checks:

```bash
MBRT_TEST_BASE_TOKENIZER=dataset/tokenizers/llama-32k uv run pytest tokenization -n 0
```

Tests using the actual pinned tokenizer skip if it is unavailable locally; they never download datasets. GPU custom-kernel and multi-GPU qualification are separate from these CPU checks.
