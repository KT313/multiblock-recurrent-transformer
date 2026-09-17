// (c) 2025-2026 Tobias Kerner. Apache-2.0.
// Experimental adapter only: all encoding, normalization, offsets and parallelism use tokenizers.
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tokenizers::Tokenizer;

#[pyclass]
struct NativeCounter {
    tokenizer: Tokenizer,
}

fn error(value: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(value.to_string())
}

fn prefix_bytes(text: &str, characters: usize) -> usize {
    text.char_indices()
        .nth(characters)
        .map_or(text.len(), |(byte, _)| byte)
}

#[pymethods]
impl NativeCounter {
    #[new]
    fn new(serialized: &str, encode_special_tokens: bool) -> PyResult<Self> {
        let mut tokenizer = Tokenizer::from_bytes(serialized.as_bytes()).map_err(error)?;
        tokenizer.set_encode_special_tokens(encode_special_tokens);
        Ok(Self { tokenizer })
    }

    fn truncate_many(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        cap: usize,
    ) -> PyResult<Vec<(String, usize)>> {
        py.detach(|| {
            let limit = cap
                .checked_mul(32)
                .ok_or_else(|| error("token cap overflow"))?;
            let mut current: Vec<String> = texts
                .into_iter()
                .map(|mut text| {
                    text.truncate(prefix_bytes(&text, limit));
                    text
                })
                .collect();
            let mut pending: Vec<usize> = (0..current.len()).collect();
            let mut counts = vec![0; current.len()];
            while !pending.is_empty() {
                let batch: Vec<&str> = pending.iter().map(|&i| current[i].as_str()).collect();
                let encoded = self
                    .tokenizer
                    .encode_batch_char_offsets(batch, false)
                    .map_err(error)?;
                let mut remaining = Vec::new();
                for (index, encoding) in pending.into_iter().zip(encoded) {
                    let count = encoding.len();
                    if count <= cap {
                        counts[index] = count;
                    } else {
                        let start = encoding
                            .get_offsets()
                            .get(cap)
                            .ok_or_else(|| error("missing cut offset"))?
                            .0;
                        let length = current[index].chars().count();
                        // Python's text[:min(start, len(text)-1)] for an empty string is still empty.
                        let cut =
                            prefix_bytes(&current[index], start.min(length.saturating_sub(1)));
                        current[index].truncate(cut);
                        remaining.push(index);
                    }
                }
                pending = remaining;
            }
            Ok(current.into_iter().zip(counts).collect())
        })
    }

    fn count_many(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<usize>> {
        py.detach(|| {
            let encoded = self
                .tokenizer
                .encode_batch_fast(texts, false)
                .map_err(error)?;
            Ok(encoded.iter().map(|encoding| encoding.len()).collect())
        })
    }
}

#[pymodule]
fn preparer_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeCounter>()
}
