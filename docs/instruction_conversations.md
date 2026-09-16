# Structured instruction conversations

`instruction_format: messages` opts an instruction source into structured multi-turn training. Existing sources default to `single_turn` and retain their existing preparation, hashes, formatting, and label masks.

The small integration config is `config/datasets/instruction_sources_smoke.yaml`. It selects a budgeted prefix of FineWeb-Edu's pinned 350BT pool and three instruction sources. It is not a recommended research mixture. Prepare it in a separate `--dataset_dir`; its token budgets and `check_limit` do not impose a network-byte limit.

## Source policies

| Converter | Input | Policy |
| --- | --- | --- |
| `opencode_messages` | `input`, `output` | Requires `filter: opencode_passed_tests`; parse the string/numeric score and retain exactly 1. |
| `webinstruct_messages` | `question`, `answer` | Nonempty strings; use `data/train-*.parquet` to exclude test and legacy splits. |
| `nemotron_messages` | `messages`, `reasoning` | Require reasoning off; remove an empty leading system message, exclude the entire record for a nonempty one, and validate alternating roles. |

Schema mistakes have malformed-row diagnostics and the existing consecutive-failure guard. Quality exclusions are counted separately in per-download-pass conversation diagnostics. Those diagnostic counters are not cumulative resumable totals; durable row offsets and malformed/oversize counters retain their existing contract.

Content is preserved, including code indentation, case, and mathematical notation. Evaluator judgments, unit tests, and category metadata do not become training text. Supplied code is never executed. WebInstruct's short reference answers are not generated worked solutions.

The new format requires `token_count: tokenizer` and rejects input inversions. Optional benchmark Bloom seeding currently supports the old flat formats only; combining it with message sources raises at configuration load. Chat deduplication does not establish benchmark decontamination.

## Serialization and loss

Policy `messages-complete-exchanges-v1` is defined in `data_preparation/lib/conversation_format.py`. It uses existing vocabulary, with no new special tokens:

```text
BOS + encode("User:\n") + encode(user_1) + encode("\n\nAssistant:\n") + encode(answer_1) + EOS
    + encode("\n\nUser:\n") + encode(user_2) + encode("\n\nAssistant:\n") + encode(answer_2) + EOS
```

Each displayed segment is encoded separately without automatic special tokens; concatenating and retokenizing the displayed text is **not** equivalent in general. Literal message content does not interpret special-token spellings as control IDs. The normal non-chat tokenizer remains unchanged.

All assistant bodies and their EOS tokens are supervised. User content, role headers, separators, BOS, and padding receive `IGNORE_INDEX` labels. Masking is structural, so a user writing `Assistant:` does not create a supervised span. User content still provides context for predicted assistant tokens.

One whole conversation is one packed document. Causal attention and positions continue between its turns; other conversations are isolated. Existing shifting, padding, and global supervised-token weighting are unchanged.

## Length fitting and planning

Keep the longest prefix ending at a complete assistant response. Drop an original trailing user message. If a subsequent exchange cannot fit, discard that exchange and everything after it. If even the first exchange cannot fit, discard the sample and count it. Never label an unfinished answer as completed by inserting EOS after a token cut.

The storage cap bounds serialized IDs. Training bounds shifted positions, so a limit of `L` allows up to `L + 1` serialized IDs. Training and validation use the same fitter before generic collation.

For message sources, the tokenizer definition is part of raw identity because stored exchange boundaries and trimmed prefixes depend on it. Changing that definition therefore requires the existing raw-repair workflow; legacy source identities are unchanged.

Raw and processed rows contain `messages`, `tokens`, and cumulative `exchange_ends`; processed rows also contain dedup keys. `exchange_ends` records exact token boundaries including specials, enabling the planner to count complete exchanges at the requested target without retokenizing a corpus. A 6K-token row may supply only a 2K exchange at a 4K target. If no raw exchange fits the target, planning raises rather than dividing by zero or claiming usable data.

Keys include ordered roles and exact content, preserving code case/whitespace. Chat keys are distinct from legacy flat keys. The same complete conversation stays on one side of the train/validation split. This implementation does not create overlapping per-answer examples or recover later turns by dropping their preceding context.

## Generation

Use the same formatter for prompts. History must end with a user query:

```python
from data_preparation.lib.conversation_format import encode_chat_prompt

messages = [
    {"role": "user", "content": "What is 2 + 2?"},
    {"role": "assistant", "content": "4."},
    {"role": "user", "content": "And twice that?"},
]
ids = encode_chat_prompt(messages, tokenizer, max_tokens=context_length - max_new_tokens)
```

Pass these IDs directly as `input_ids` to the existing model/HF generation path, inside the usual inference session. Do not decode and retokenize them. The IDs already include BOS and completed-history EOS tokens and end with the pending assistant header. Stop the generated reply at EOS. Overflow raises; it does not silently discard history. Plain-text sample generation and standard benchmark prompts retain their original behavior.

See the scoped integration validation report for the actual checks and download bounds used for this change.

For the explicit `<user>`/`<assistant>` tokenizer profile, literal special-token strings in message content,
startup checks and portable exports, see [Llama 32K chat tokenizer](llama32k_chat_tokenizer.md).
The new instruction smoke dataset opts into this profile; legacy message datasets keep their original text headers.
