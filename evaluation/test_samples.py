# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests of the wrapper, the inference isolation, the prompts and the sample generations (tiny model, CPU).
"""

import json
import os
from pathlib import Path

import pytest
import torch

from evaluation.prompts import (
    CONTINUATION,
    DEFAULT_PROMPTS,
    INSTRUCTION,
    Prompt,
    instruction_prompt,
    load_prompts,
    load_prompts_file,
)
from evaluation.samples import GeneratedSample, _sample_from, generate_and_save_samples, generate_samples, samples_path
from evaluation.wrapper import RECURRENCE_ENV, check_recurrence, hf_wrapper_around, isolated_inference
from model.model import RecurrentGPT
from training.data.tokenizer import Tokenizer


@pytest.fixture
def tokenizer(tiny_tokenizer_dir: Path) -> Tokenizer:
    return Tokenizer(tiny_tokenizer_dir)


def test_wrapper_shares_the_live_tensors(tiny_model: RecurrentGPT, tokenizer: Tokenizer) -> None:
    wrapper = hf_wrapper_around(tiny_model, tokenizer)
    assert not wrapper.training
    assert wrapper.model.transformer.wte.weight.data_ptr() == tiny_model.transformer.wte.weight.data_ptr()
    assert wrapper.model.lm_head.weight.data_ptr() == tiny_model.lm_head.weight.data_ptr()
    generation = wrapper.generation_config
    assert (generation.pad_token_id, generation.eos_token_id, generation.bos_token_id) == (
        tokenizer.pad_id,
        tokenizer.eos_id,
        tokenizer.bos_id,
    )


def test_isolated_inference_restores_rng_mode_and_env(tiny_model: RecurrentGPT, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RECURRENCE_ENV, raising=False)
    tiny_model.train()
    torch.manual_seed(7)
    before = torch.get_rng_state()
    with isolated_inference(tiny_model, [2, 2]):
        mode_inside: bool = tiny_model.training
        assert os.environ[RECURRENCE_ENV] == "2,2"
        torch.rand(5)
        tiny_model(torch.randint(1, 512, (1, 4)))
    mode_after: bool = tiny_model.training
    assert (mode_inside, mode_after) == (False, True)
    assert RECURRENCE_ENV not in os.environ
    assert torch.equal(torch.get_rng_state(), before)
    monkeypatch.setenv(RECURRENCE_ENV, "1")
    with isolated_inference(tiny_model, [3, 3]):
        assert os.environ[RECURRENCE_ENV] == "3,3"
    assert os.environ[RECURRENCE_ENV] == "1"
    with isolated_inference(tiny_model):  # no recurrence given: a value left over from elsewhere must not win
        assert RECURRENCE_ENV not in os.environ
    assert os.environ[RECURRENCE_ENV] == "1"


def test_check_recurrence_rejects_a_wrong_block_count(tiny_model: RecurrentGPT) -> None:
    check_recurrence(None, tiny_model)
    check_recurrence([4, 4], tiny_model)
    with pytest.raises(ValueError, match="2 blocks"):
        check_recurrence([4, 4, 4], tiny_model)  # the tiny model has two core blocks
    with pytest.raises(ValueError, match="positive"):
        check_recurrence([0, 4], tiny_model)


def test_default_prompts() -> None:
    assert instruction_prompt("Translate.", "  Hallo  ") == Prompt("Translate.\n\nHallo\n\n", INSTRUCTION)
    assert all(p.kind in (CONTINUATION, INSTRUCTION) for p in DEFAULT_PROMPTS)
    assert all(p.text.endswith("\n\n") for p in DEFAULT_PROMPTS if p.kind == INSTRUCTION)
    assert load_prompts() == list(DEFAULT_PROMPTS)


def test_prompts_file(tmp_path: Path) -> None:
    path = tmp_path / "prompts.txt"
    path.write_text("Once upon a time\n---\n# instruction\nSummarize the text.\n\nA long text.\n---\n\n# continuation\ndef f():\n")
    prompts = load_prompts_file(path)
    assert prompts == [
        Prompt("Once upon a time"),
        Prompt("Summarize the text.\n\nA long text.\n\n", INSTRUCTION),
        Prompt("def f():"),
    ]
    assert load_prompts(path) == prompts
    path.write_text("# poem\nRoses")
    with pytest.raises(ValueError, match="unknown prompt kind"):
        load_prompts_file(path)
    path.write_text("\n---\n")
    with pytest.raises(ValueError, match="no prompts"):
        load_prompts_file(path)


def test_sample_from_cuts_at_eos_and_keeps_generated_pad_ids(tokenizer: Tokenizer) -> None:
    eos, pad = tokenizer.eos_id, tokenizer.pad_id
    assert eos is not None and eos != pad
    prompt = Prompt("p")
    cut = _sample_from(prompt, [5, 6, eos, pad, pad], tokenizer)  # the filler after EOS goes with the cut
    assert cut == GeneratedSample("p", CONTINUATION, tokenizer.decode([5, 6], skip_special_tokens=True), 2, True)
    unfinished = _sample_from(prompt, [5, 6, pad, pad], tokenizer)  # no EOS: the pad ids are the model's own output
    assert (unfinished.new_tokens, unfinished.stopped_at_eos) == (4, False)


def test_generate_samples_greedy_is_deterministic_and_bounded(tiny_model: RecurrentGPT, tokenizer: Tokenizer) -> None:
    prompts = [Prompt("hello world"), instruction_prompt("say hi"), Prompt("a much longer prompt with more tokens in it")]
    tiny_model.train()
    torch.manual_seed(0)
    first = generate_samples(tiny_model, tokenizer, prompts, max_new_tokens=5, batch_size=2)
    torch.manual_seed(1)  # the isolated RNG is seeded inside: the global state does not reach the initial latent draw
    second = generate_samples(tiny_model, tokenizer, prompts, max_new_tokens=5, batch_size=3)
    assert first == second and len(first) == 3 and tiny_model.training  # neither batching nor the global RNG changes greedy output
    assert [sample.kind for sample in first] == [CONTINUATION, INSTRUCTION, CONTINUATION]
    assert all(0 <= sample.new_tokens <= 5 for sample in first)  # an untrained model may emit EOS at once
    sampled_a = generate_samples(tiny_model, tokenizer, prompts, max_new_tokens=12, temperature=1.5)
    sampled_b = generate_samples(tiny_model, tokenizer, prompts, max_new_tokens=12, temperature=1.5, seed=1)
    assert sampled_a != sampled_b and sampled_a == generate_samples(tiny_model, tokenizer, prompts, max_new_tokens=12, temperature=1.5)


def test_generate_and_save_samples_writes_jsonl(tiny_model: RecurrentGPT, tokenizer: Tokenizer, tmp_path: Path) -> None:
    path = samples_path(tmp_path, 12)
    assert path == tmp_path / "samples" / "step-00000012.jsonl"
    samples = generate_and_save_samples(
        tiny_model, tokenizer, path, step=12, max_new_tokens=4, batch_size=3, recurrences=[[1, 1], None]
    )
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")[:-1]]
    assert len(lines) == len(samples) == 2 * len(DEFAULT_PROMPTS)  # every prompt once per recurrence setting
    assert [line["prompt"] for line in lines] == 2 * [prompt.text for prompt in DEFAULT_PROMPTS]
    assert [line["recurrence"] for line in lines] == len(DEFAULT_PROMPTS) * [[1, 1]] + len(DEFAULT_PROMPTS) * [None]
    assert lines[0]["step"] == 12 and lines[0]["completion"] == samples[0].completion
    assert lines[0]["decoding"] == {"temperature": 0.0, "max_new_tokens": 4, "seed": 0}
    assert set(lines[0]) == {"step", "prompt", "kind", "completion", "new_tokens", "stopped_at_eos", "recurrence", "decoding"}
