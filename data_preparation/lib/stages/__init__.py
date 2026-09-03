# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
The pipeline steps: download / download_github_code_group (stages/download.py, with the tokenizer
step, the raw-manifest state helpers and the shared manifest helpers), build_source (stages/build.py, both
source kinds); row-level helpers in row_pipeline.py, the exact-dedup filter in exact_dedup.py, near-duplicate
removal in fuzzy_dedup.py, benchmark n-grams in benchmarks.py, text truncation at the token cap for the
download in truncation.py.
"""
