# Prepared dataset build identity

Preparation publishes `dataset/snapshots/<dataset-config-hash>.json` after all required sources and the tokenizer are coherent. Its versioned `build_id` identifies the ordered source generations and tokenizer generation. Each processed/tokenizer manifest records its `generation_id` and whether publication is complete. Incremental updates mark the generation incomplete before changing shards and retain that ID when interrupted work resumes. Whole-folder builds finish the private generation before swapping it into place.

A managed rebuild, replacement, extension or repair of published output changes its generation. Snapshots referencing a shared source's previous generation become stale, including snapshots created by another dataset configuration. Unrelated sources and statistics-only saves do not change a valid snapshot. Repeated preparation and read-only status retain its ID. Status reports missing, stale and incomplete snapshot evidence without writing or adopting anything; run preparation to finish publication.

Training validates the descriptor against small manifests under the dataset lease, records `dataset_build_id` in checkpoints, and compares it on resume in addition to existing configuration, row-count and validation-split checks. All ranks agree on the resolved identity. Identity comparison reads no shard contents; ordinary preparation and training readiness checks still inspect Parquet metadata as before.

Legacy prepared output can be adopted by preparation without rebuilding its bytes. This establishes identity **from adoption onward**. Legacy checkpoints retain a missing ID as unknown provenance. Protected resume refuses missing or different identities; `allow_dataset_change: true` explicitly acknowledges that exact dataset replay cannot be established. `allow_settings_change` does not authorize a dataset identity mismatch.

Build IDs are managed-publication evidence, not content hashes. Manual file edits that retain the identity metadata are outside this guarantee. No sample-order, split, tokenizer behavior, stream-buffer restoration or training numerical settings are changed by identity bookkeeping.
