# Literal-aware Llama chat tokenizer validation

Implementation worktree: `/tmp/mbrt-llama32k-chat`, branch `feature/llama32k-chat`. Validated 2026-09-17. No GPU, multi-GPU training or full-corpus downloads were run.

## Evidence

- Combined serial CPU regressions: **806 passed, 1 skipped, 1 deselected** in 208.10 s. Coverage included tokenizer preparation/building, configuration and snapshot identities, training/checkpoints/resume, the golden training run, sample and benchmark evaluation, and standalone HF model export. The skipped test requires benchmark downloads; CUDA coverage was excluded.
- After the final chat-body whitespace correction: **82 passed** in 48.72 s across the dedicated tokenizer package, sample generation, legacy conversation formatting and CPU conversation training. These overlap the combined run and must not be added as unique tests.
- Final scoped Ruff passed; strict mypy passed for 55 files; basedpyright reported zero errors/warnings. `git diff --check` passed.
- Actual pinned-tokenizer tests used the existing local base artifact. Only the pinned upstream tokenizer JSON was fetched independently to establish the base reference; no dataset samples were fetched for this implementation.

The dedicated tests verify literal marker strings are retained as ordinary content, exact assistant masks, complete-exchange fitting, packing boundaries, usable/padded vocab checks, untied input-role embedding gradients, field-mapped question/answer preparation, mixed text/chat CPU training and exact resumed next-step loss. They also verify fresh acquisition with a mocked Hub, cached idempotence, failed-publication recovery, cross-rank contract disagreement with a fake backend, run-relative checkpoint resolution, actual installed HFLM candidate tokenization, and standalone tokenizer/model loading without repository imports.

Generation tests verify continuation prompts remain plain BOS/text and built-in, file-based and multi-turn instruction prompts are exact prefixes of their training conversations. Structural role/EOS IDs appear only at formatter boundaries. Decoded chats retain the specified spaces/newlines, including literal special-token spellings. The shared chat body encoder disables the automatic prefix space; ordinary pretraining encoding keeps the base tokenizer's behavior.

## Small startup measurement

A fresh CPU process measured one selected tokenizer instance, without model construction or dataset reads:

| Operation | Seconds |
| --- | ---: |
| Lightweight module imports | 0.016 |
| Tokenizer load and contract validation | 0.289 |
| First parity check, including lazy framework/HF imports and processor construction | 3.949 |
| Parity check with the processor already loaded | 0.004 |

This is one local observation, not a cross-machine benchmark. Training already imports PyTorch before preflight, so its incremental cold cost is not identical to this isolated process. Production performs the probes once during setup, not per microbatch.

## Deliberate compatibility boundaries

The saved custom tokenizer adapter is required for chat: retokenizing a flattened template cannot distinguish literal marker spellings from structure. Use `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` and `apply_chat_template(tokenize=True)`. Template-only consumers and unmodified external HFLM chat paths are not qualified; this repository provides a literal-aware HFLM adapter. Plain benchmark protocol remains the default.

Existing tokenizer profiles, research configs and training-loss padding behavior are preserved. The new smoke dataset/run opt in explicitly. Changing a trained checkpoint from the legacy tokenizer to this profile is rejected; no vocabulary resizing or checkpoint migration is implemented. GPU custom-kernel and real distributed startup/communication qualification remain separate work.
