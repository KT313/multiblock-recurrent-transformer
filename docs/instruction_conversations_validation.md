# Structured instruction dataset validation

Validated on CPU in the `feature/instruction-messages` worktree, 2026-09-16. No GPU benchmarks or production datasets were used. The source policies and exact token format are documented in [instruction_conversations.md](instruction_conversations.md).

## Offline checks

- 377 focused preparation/config/planner/build/download/global-admission and new conversation tests passed.
- 382 terminal, training-data, multiprocessing-loader, and CPU Gloo tests passed with `TERM=xterm-256color` and the required process/socket access.
- 48 targeted conversation/pipeline/description tests passed after final identity and startup-validation changes.
- Both tiny-model tests passed: manual masked-loss and packed/padded gradient parity, plus an actual two-step mixed text/chat training run resumed from step one. The resumed final model tensors and next loss matched the uninterrupted run exactly; consumed-row counts matched too.
- Scoped Ruff, strict mypy, basedpyright, and `git diff --check` passed.

The initial broad sandbox run was interrupted in a multiprocessing loader test after 1,237 passes. Its terminal-setting and Gloo restrictions were addressed by rerunning the relevant checks outside the sandbox. The new model test's initial test-setup mistakes were fixed (loss-statistics mode excludes logits; the model's output vocabulary need not equal the fixture tokenizer vocabulary). Production model logic was not changed to satisfy those tests. Counts above overlap and must not be added together as unique tests.

Coverage includes score-string parsing and malformed values, split glob selection, nonempty-system exclusions, trailing-user trimming, arbitrary turn counts, exact and one-token-over fitting boundaries, 4K/8K limits, assistant labels through next-token shifting, literal special-token spellings, ordered case/whitespace-sensitive dedup keys, nested Parquet rows, SQLite shuffle, global admission, preparation reuse, interrupted download/resume, and fitted planner counts.

## Bounded upstream fixture check

Selected 64 initial records from each pinned source; acquired only the needed Parquet columns/row groups or a 1 MB JSONL prefix. A hard combined 256 MB HTTP-read budget prevented unbounded downloads. FineWeb's initial fixture was reused after a metadata request was rate-limited; cached pinned metadata avoided repeating that request.

| Source | Revision | Dataset body bytes read |
| --- | --- | ---: |
| FineWeb-Edu 350BT | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | 5,625,159 |
| OpenCodeInstruct | `8f3ba5bafe4d6e8db46082cf7ae6741bc370604d` | 76,317,810 |
| WebInstruct-verified train | `3e8a350b3a935d68fe70bdf379692500abc9ff51` | 41,743,299 |
| Nemotron reasoning_off | `1a9454ed054b8544503ab8d8c0a519d141a44c5b` | 1,000,000 |
| Total | | 124,686,268 |

Parquet column chunks explain why the byte totals are much larger than the extracted 64 rows. An additional conservative 1 MB metadata allowance gave a budgeted total of 125,686,268 bytes. The original 100 MB Nemotron research audit preceded implementation and was not repeated.

Prepared the acquired records using a **local-loader variant** of the smoke YAML, the pinned Llama tokenizer, 8,192 storage tokens, and both 4,096 and 8,192 training positions. The fixture variant used 8,192 tokens per stage rather than the shipped config's 524,288 total, ensuring small samples could provide train and validation rows. All four sources produced both train and validation data; repeat preparation reused their shards. Local preparation peaked at approximately 351 MiB RSS and the fixture/preparation directory occupied about 9 MiB. No upstream file was downloaded in full except where the selected column chunks nearly cover its contents.

The exact first-64 instruction-fixture accounting, before deduplication and validation splitting:

| Source | Source-filter rejected | Usable at 4K | Usable at 8K | Multi-turn at 4K | Multi-turn at 8K |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenCodeInstruct | 34 | 30 | 30 | 0 | 0 |
| WebInstruct-verified | 0 | 64 | 64 | 0 | 0 |
| Nemotron reasoning_off | 0 | 62 | 64 | 24 | 27 |

At 4K, Nemotron lost five exchanges across these fixtures, including two conversations with no fitting first exchange. At 8K, it lost none. These prefix fixtures are not representative population estimates. System exclusions and malformed records absent from this small prefix are covered by offline fixtures and the earlier research audit.

| Source | Sequence positions at 4K | Supervised tokens at 4K | Sequence positions at 8K | Supervised tokens at 8K |
| --- | ---: | ---: | ---: | ---: |
| OpenCodeInstruct | 14,947 | 7,861 | 14,947 | 7,861 |
| WebInstruct-verified | 7,895 | 1,145 | 7,895 | 1,145 |
| Nemotron reasoning_off | 69,861 | 53,105 | 86,918 | 66,666 |

## Limits

This verifies real upstream records through local preparation and shared token formatting, plus offline mocked Hub selection. It is not a complete live `hf_files` preparation of the shipped smoke config, a full-corpus pass-rate/capacity estimate, benchmark decontamination, a throughput measurement, or an eight-GPU training qualification.

Temporary audit inputs/results are under `/tmp/mbrt-chat-online-smoke`; this document preserves the findings if those artifacts are removed. Production configs and existing dataset directories were not changed.
