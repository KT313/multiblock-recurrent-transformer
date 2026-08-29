# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Source registry: loaders (`sources/loaders.py`), converters + filters (`sources/converters.py`) and the
synthetic generator (`sources/synthetic.py`). Everything public is re-exported here."""

from data_preparation.lib.sources.converters import (
    CONVERTERS,
    FILTERS,
    Converter,
    Filter,
    fields_converter,
    first_two_turns,
    get_converter,
    get_filter,
    gsm8k_question_answer,
    instruction_input_output,
    sharegpt_conversations,
    sharegpt_quality,
)
from data_preparation.lib.sources.loaders import (
    GITHUB_CODE_DATA_FILES,
    LOADERS,
    MAX_CACHED_FILE_KEY,
    Loader,
    Row,
    get_loader,
    hub_fetcher,
    hub_file_index,
    iter_language,
    list_local_files,
    load_github_code,
    load_hf_files,
    load_hf_split,
    load_hf_stream,
    load_local,
    load_synthetic,
)
from data_preparation.lib.sources.hub_files import DEFAULT_MAX_CACHED_FILE_MB, FetchStats, HubFetcher
from data_preparation.lib.sources.synthetic import (
    N_WORD_TOKENS,
    SPECIALS,
    SYNTHETIC_DOC_WORDS,
    SYNTHETIC_INSTRUCTION_WORDS,
    SYNTHETIC_OUTPUT_WORDS,
    VOCAB_SIZE,
    synthetic_row,
    write_synthetic_tokenizer,
)

__all__ = [
    "CONVERTERS", "FILTERS", "Converter", "Filter", "fields_converter", "first_two_turns", "get_converter",
    "get_filter", "gsm8k_question_answer", "instruction_input_output", "sharegpt_conversations", "sharegpt_quality",
    "GITHUB_CODE_DATA_FILES", "LOADERS", "MAX_CACHED_FILE_KEY", "Loader", "Row", "get_loader", "hub_fetcher", "hub_file_index",
    "iter_language", "list_local_files", "load_github_code",
    "DEFAULT_MAX_CACHED_FILE_MB", "FetchStats", "HubFetcher",
    "load_hf_files", "load_hf_split", "load_hf_stream", "load_local", "load_synthetic",
    "N_WORD_TOKENS", "SPECIALS", "SYNTHETIC_DOC_WORDS", "SYNTHETIC_INSTRUCTION_WORDS", "SYNTHETIC_OUTPUT_WORDS",
    "VOCAB_SIZE", "synthetic_row", "write_synthetic_tokenizer",
]
