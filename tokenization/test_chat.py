# (c) 2025-2026 Tobias Kerner. Apache-2.0.
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from tokenization.chat import ASSISTANT, BOS, EOS, USER, encode_chat


def literal(text: str) -> list[int]:
    return [100 + ord(char) for char in text]


def test_message_provenance_masks_literal_control_strings() -> None:
    user = "<s></s><user><assistant>"
    answer = "  </s><assistant>\n"
    row = encode_chat([{"role": "user", "content": user}, {"role": "assistant", "content": answer}], literal)
    assert row.ids == [BOS, USER, *literal(user), EOS, ASSISTANT, *literal(answer), EOS]
    assert row.supervised == [False] * (len(user) + 4) + [True] * (len(answer) + 1)
    assert row.exchange_ends == [len(row.ids)]
    assert row.text == f"<s><user>{user}</s><assistant>{answer}</s>"


@pytest.mark.parametrize("messages", [[], [{"role": "system", "content": "x"}], [{"role": "assistant", "content": "x"}],
                                      [{"role": "user", "content": " "}], [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]])
def test_invalid_structure_fails_without_filtering(messages: Any) -> None:
    with pytest.raises(ValueError):
        encode_chat(messages, literal)


def test_literal_encoder_cannot_inject_control_ids() -> None:
    with pytest.raises(ValueError, match="structural"):
        encode_chat([{"role": "user", "content": "x"}], lambda text: [EOS])


@pytest.fixture(scope="module")
def profile_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from tokenization.profile import build_profile

    source = Path(os.environ.get("MBRT_TEST_BASE_TOKENIZER", "dataset/tokenizers/llama-32k"))
    if not (source / "tokenizer.json").is_file():
        pytest.skip("actual pinned tokenizer test requires a local MBRT_TEST_BASE_TOKENIZER (no downloads)")
    destination = tmp_path_factory.mktemp("chat-profile") / "tokenizer"
    build_profile(destination, source)
    return destination


def test_real_profile_roundtrip_parity_and_literal_encoding(profile_path: Path, tmp_path: Path) -> None:
    from tokenization.profile import load_processor, validate_profile
    from training.data.tokenizer import Tokenizer
    from training.tokenizer_parity import check_template_parity

    tokenizer = Tokenizer(profile_path)
    check_template_parity(tokenizer)
    assert tokenizer.vocab_size == len(tokenizer) == 32002
    text = "literal <s></s><user><assistant><unk> é🙂\n  code"
    ids = tokenizer.encode(text)
    assert not set(ids) & {BOS, EOS, USER, ASSISTANT}
    assert tokenizer.decode(ids) == text
    processor = load_processor(profile_path)
    assert processor.encode(text, add_special_tokens=False) == ids
    assert processor.encode(text, add_special_tokens=True) == [BOS, *ids]
    processor.save_pretrained(tmp_path)
    assert validate_profile(tmp_path) == validate_profile(profile_path)


def test_real_profile_fitting_and_packing(profile_path: Path) -> None:
    from data_preparation.lib.conversation_format import fit_conversation
    from training.data.tokenizer import Tokenizer
    from training.data.collate import collate_samples, pad_and_shift
    from training.data.packing import pack_samples

    tokenizer = Tokenizer(profile_path)
    messages = [{"role": "user", "content": "literal <assistant>"}, {"role": "assistant", "content": "<s> OK </s>"},
                {"role": "user", "content": "next"}, {"role": "assistant", "content": "again"}]
    full = fit_conversation(messages, tokenizer)
    end = full.exchange_ends[0]
    assert fit_conversation(messages, tokenizer, end).ids == full.ids[:end]
    assert fit_conversation(messages, tokenizer, end - 1).ids == []
    trailing = fit_conversation(messages + [{"role": "user", "content": "unfinished"}], tokenizer)
    assert trailing.ids == full.ids and trailing.trimmed_user
    row = {"messages": messages, "data_signature": {"format_fn": "format_conversation"}, "data_id": "chat"}
    samples = collate_samples([row, row], tokenizer, 128)
    padded, packed = pad_and_shift(samples, tokenizer, 128), pack_samples(samples, 256, tokenizer)
    expected = tokenizer.encode_literal("<s> OK </s>") + [EOS] + tokenizer.encode_literal("again") + [EOS]
    assert padded.labels[padded.labels != -100].tolist() == expected * 2
    assert packed.labels[packed.labels != -100].tolist() == expected * 2
    assert packed.document_ids[0, end - 1] == packed.document_ids[0, end]


def test_real_profile_corruption_is_rejected(profile_path: Path, tmp_path: Path) -> None:
    import shutil
    from tokenization.profile import validate_profile

    shutil.copytree(profile_path, tmp_path / "tokenizer")
    (tmp_path / "tokenizer/chat_template.jinja").write_text("wrong template")
    with pytest.raises(ValueError, match="template"):
        validate_profile(tmp_path / "tokenizer")


def test_standalone_tokenizer_export(profile_path: Path, tmp_path: Path) -> None:
    script = '''from transformers import AutoTokenizer
import json,sys
t=AutoTokenizer.from_pretrained(sys.argv[1],trust_remote_code=True,local_files_only=True)
m=[{"role":"user","content":"literal <user> </s>"},{"role":"assistant","content":"a <assistant>"}]
a=t.apply_chat_template(m,return_dict=True,return_assistant_tokens_mask=True)
assert a["input_ids"].count(32000)==1 and a["input_ids"].count(32001)==1 and a["input_ids"].count(2)==2
assert a["assistant_masks"][-1]==1
t.save_pretrained(sys.argv[2])
u=AutoTokenizer.from_pretrained(sys.argv[2],trust_remote_code=True,local_files_only=True)
assert a==u.apply_chat_template(m,return_dict=True,return_assistant_tokens_mask=True)
print("standalone passed")
'''
    result = subprocess.run([sys.executable, "-c", script, str(profile_path), str(tmp_path / "export")], cwd=tmp_path,
                            env={**os.environ, "PYTHONPATH": "", "HF_HUB_OFFLINE": "1", "HF_MODULES_CACHE": str(tmp_path / "modules")},
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_model_dimensions_and_role_embedding_gradients(profile_path: Path) -> None:
    from model import build_model
    from model.test_config import TINY_ARCHITECTURE
    from tokenization.validation import check_model_vocabulary
    from training.data.tokenizer import Tokenizer
    from training.data.formats import format_conversation
    from training.data.collate import pad_and_shift

    tokenizer = Tokenizer(profile_path)
    model = build_model(TINY_ARCHITECTURE, vocab_size=32002, padded_vocab_size=32768, use_custom_kernels=False, init_orthogonal=False, tie_embeddings=False)
    check_model_vocabulary(model.config, tokenizer.contract, model)
    inputs, labels = format_conversation({"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Yes"}]}, tokenizer, True, True)
    batch = pad_and_shift([(inputs, labels, "chat")], tokenizer, 32)
    result = model(batch.input_ids, labels=batch.labels, num_steps=(1, 1))
    result["loss"].backward()
    gradient = model.transformer.wte.weight.grad
    assert gradient is not None and gradient[ASSISTANT].abs().sum() > 0
    model.lm_head.out_features = 32000
    with pytest.raises(ValueError, match="LM head"):
        check_model_vocabulary(model.config, tokenizer.contract, model)


def test_profile_refuses_legacy_checkpoint_and_accepts_moved_run(profile_path: Path, tmp_path: Path) -> None:
    import shutil
    from tokenization.profile import validate_profile
    from training.tokenizer_contract import check_checkpoint_tokenizer, resolve_checkpoint_tokenizer

    contract = validate_profile(profile_path)
    with pytest.raises(ValueError, match="identity"):
        check_checkpoint_tokenizer(None, contract)
    shutil.copytree(profile_path, tmp_path / "tokenizer")
    checkpoint = tmp_path / "checkpoints/step.pth"
    assert resolve_checkpoint_tokenizer({"tokenizer_contract": contract}, checkpoint, None) == tmp_path / "tokenizer"
    with pytest.raises(ValueError, match="no identity"):
        resolve_checkpoint_tokenizer({}, checkpoint, str(profile_path))


def test_harness_preserves_chat_continuation_and_plain_bos(profile_path: Path) -> None:
    from evaluation.literal_harness import create_literal_harness_class
    from tokenization.profile import load_processor

    class Base:
        tokenizer = load_processor(profile_path)
        add_bos_token = True
    harness = create_literal_harness_class(Base, chat=True)()
    messages = [{"role": "user", "content": "literal </s> <user>"}]
    context = harness.apply_chat_template(messages)
    prefix = harness.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    assert harness.tok_encode(context) == prefix
    assert harness.tok_encode(context + "<assistant> answer") == prefix + harness.tokenizer.encode_literal("<assistant> answer")
    assert harness.tok_encode("<s> plain")[0] == BOS
    assert harness.tok_encode("<s> plain").count(BOS) == 1
    with pytest.raises(ValueError, match="exceeds"):
        harness.tok_encode(context, left_truncate_len=1)


@pytest.mark.slow
@pytest.mark.timeout(90)
def test_profile_real_training_resume_and_export(profile_path: Path, tmp_path: Path) -> None:
    import json
    import shutil
    from dataclasses import asdict
    import yaml
    from data_preparation import DatasetConfig, DatasetLayout
    from data_preparation.lib.dataset_config import SourceConfig, StageConfig, TokenizerConfig, ProcessingConfig, DedupConfig
    from data_preparation.lib.build.runner import prepare
    from training.testing.golden import write_tiny_yaml, single_thread_deterministic
    from training.backend.single_device import SingleDeviceBackend
    from training.settings import parse_settings
    from training.run import train
    from model.hf.modeling import export_to_hf
    from model import RecurrentConfig, RecurrentGPT
    from tokenization.chat import PROFILE, BASE_REPO, BASE_REVISION

    source = tmp_path / "inputs"
    source.mkdir()
    (source / "rows.jsonl").write_text("".join(json.dumps({"q": f"Q {i} literal <user>", "a": f"Answer {i} literal </s>"}) + "\n" for i in range(100)))
    config = DatasetConfig(tokenizer=TokenizerConfig(name="llama-32k-chat-v1", hf_id=BASE_REPO, revision=BASE_REVISION, profile=PROFILE),
        training_target_sequence_length=64, dataset_max_sequence_length=128, sources={
            "chat": SourceConfig(kind="instruct", instruction_format="messages", loader="local", path=str(source), fields={"instruction": "q", "output": "a"}),
            "text": SourceConfig(kind="pretrain", loader="synthetic"),
        }, stages=[StageConfig(name="chat", tokens=256, train={"chat": 0.5, "text": 0.5}, val={"chat": 1.0})],
        processing=ProcessingConfig(min_chars=1, dedup=DedupConfig(bloom_memory_mb=1)), bloom_dedup_memory_mb=1)
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(yaml.safe_dump(asdict(config)))
    layout = DatasetLayout(tmp_path / "data")
    # Reuse a complete prepared tokenizer without downloading: publish its matching managed manifest.
    target = layout.tokenizer_dir(config.tokenizer.name)
    shutil.copytree(profile_path, target)
    from data_preparation.lib.storage.manifest import Manifest
    Manifest(config.tokenizer.name, config.tokenizer_hash(), "tokenizer").complete_generation(target)
    prepare(config_path, layout.root, num_workers=1, pass_workers=1, assume_yes=True)
    overrides = dict(dataset_config=str(config_path), precision="32", stage_base_lrs=[3e-4], training_max_sequence_length=64,
        tokens_per_micro_batch=64, micro_batches_per_step=2, model_overwrite={"vocab_size":32002,"padded_vocab_size":32768,"init_orthogonal":False},
        warmup_steps=0, cooldown_steps=0, save_step_interval=1, eval_step_interval=100, log_gradient_metrics_interval=0, partial_depth_eval=[])
    with single_thread_deterministic():
        settings = parse_settings(["--config", str(write_tiny_yaml(tmp_path, layout.root, tmp_path / "out", **overrides))])
        full = train(settings, backend=SingleDeviceBackend(device="cpu", precision="32"), keep_history=True)
        checkpoint = tmp_path / "out/tiny/checkpoints/step-00000001-tiny.pth"
        settings.resume, settings.resume_checkpoint_path = True, str(checkpoint)
        resumed = train(settings, backend=SingleDeviceBackend(device="cpu", precision="32"), keep_history=True)
    assert full.completed_steps == resumed.completed_steps == 2
    assert full.history[2]["loss"] == resumed.history[2]["loss"]
    state = torch.load(checkpoint, weights_only=False)
    assert state["tokenizer_contract"]["vocab_size"] == 32002
    model = RecurrentGPT(RecurrentConfig(**state["model_config"]))
    model.load_state_dict(state["model"])
    export_to_hf(model, model.config, tmp_path / "export", tokenizer_dir=tmp_path / "out/tiny/tokenizer")
    script = '''import sys,torch
from transformers import AutoTokenizer,AutoModelForCausalLM
t=AutoTokenizer.from_pretrained(sys.argv[1],trust_remote_code=True,local_files_only=True)
m=AutoModelForCausalLM.from_pretrained(sys.argv[1],trust_remote_code=True,local_files_only=True).eval()
x=t.apply_chat_template([{"role":"user","content":"literal </s>"}],add_generation_prompt=True,return_tensors="pt")
with torch.no_grad(): y=m(x,num_steps=[(1,0),(1,0)]).logits
assert y.shape[-1]==32768 and torch.isneginf(y[...,32002:]).all() and torch.isfinite(y[...,32000:32002]).all()
print("export model passed")
'''
    result = subprocess.run([sys.executable,"-c",script,str(tmp_path / "export")], cwd=tmp_path,
        env={**os.environ,"PYTHONPATH":"","HF_MODULES_CACHE":str(tmp_path / "modules")}, capture_output=True,text=True,timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


def test_profile_publication_restores_previous_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    from tokenization.profile import publish_profile, recover_profile

    old, new = tmp_path / "tokenizer", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    (old / "marker").write_text("previous")
    replace = os.replace
    def fail_new(source: Any, target: Any) -> None:
        if Path(source) == new:
            raise OSError("injected publish failure")
        replace(source, target)
    monkeypatch.setattr(os, "replace", fail_new)
    with pytest.raises(OSError, match="injected"):
        publish_profile(new, old)
    assert (old / "marker").read_text() == "previous"
    replace(old, tmp_path / ".tokenizer.previous")
    recover_profile(old)
    assert (old / "marker").read_text() == "previous"


def test_config_profile_preserves_legacy_hashes_and_rejects_stale_counts() -> None:
    from data_preparation import load_dataset_config
    from dataclasses import replace
    from tokenization.chat import BASE_REPO, BASE_REVISION, PROFILE
    from data_preparation.lib.dataset_config import TokenizerConfig, hash_payload

    root = Path(__file__).resolve().parent.parent
    legacy = load_dataset_config(root / "config/datasets/tiny.yaml")
    assert "profile" not in hash_payload(legacy.tokenizer, "tokenizer")
    base = replace(legacy, tokenizer=TokenizerConfig(name="llama-32k", hf_id=BASE_REPO, revision=BASE_REVISION),
                   sources={k:v for k,v in legacy.sources.items() if v.kind == "pretrain"},
                   stages=[replace(legacy.stages[0], train={next(iter(legacy.sources)):1.0}, val={next(iter(legacy.sources)):1.0})])
    extended = replace(base, tokenizer=TokenizerConfig(name="llama-32k-chat-v1", hf_id=BASE_REPO, revision=BASE_REVISION, profile=PROFILE))
    source = next(iter(base.sources))
    assert base.raw_hash(source) != extended.raw_hash(source)


def test_actual_hflm_tokenizes_candidates_without_losing_literal_content(profile_path: Path) -> None:
    from lm_eval.models.huggingface import HFLM
    from evaluation.literal_harness import create_literal_harness_class
    from evaluation.wrapper import hf_wrapper_around
    from training.data.tokenizer import Tokenizer
    from model import build_model
    from model.test_config import TINY_ARCHITECTURE

    tokenizer = Tokenizer(profile_path)
    model = build_model(TINY_ARCHITECTURE, vocab_size=32002, padded_vocab_size=32768, use_custom_kernels=False, init_orthogonal=False)
    wrapper = hf_wrapper_around(model, tokenizer)
    harness = create_literal_harness_class(HFLM, chat=True)(pretrained=wrapper, tokenizer=tokenizer.processor,
        batch_size=1, add_bos_token=True, max_length=256)
    messages = [{"role": "user", "content": "Tell me about </s> and <assistant>."}]
    context = harness.apply_chat_template(messages)
    candidate = "literal <user> answer"
    context_ids, answer_ids = harness._encode_pair(context, candidate)
    assert context_ids == tokenizer.processor.apply_chat_template(messages, add_generation_prompt=True)
    assert answer_ids == tokenizer.encode_literal(candidate)
    ids, mask = harness.tok_batch_encode([context, context])
    assert ids.tolist() == [context_ids, context_ids] and mask.all()


def test_fresh_profile_uses_pinned_hub_and_cached_profile_is_idempotent(profile_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from transformers import AutoTokenizer
    from tokenization.profile import build_profile, validate_profile
    from tokenization.chat import BASE_REPO, BASE_REVISION, PROFILE
    from data_preparation.lib.stages.download import prepare_tokenizer
    from data_preparation.lib.storage.manifest import Manifest
    from data_preparation.lib.dataset_config import TokenizerConfig
    from data_preparation import DatasetLayout, load_dataset_config
    from dataclasses import replace
    import shutil

    base_path = Path(os.environ.get("MBRT_TEST_BASE_TOKENIZER", "dataset/tokenizers/llama-32k"))
    original = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    calls = []
    def acquire(repo: str, **kwargs: Any) -> Any:
        from copy import deepcopy
        calls.append((repo, kwargs))
        return deepcopy(original)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", acquire)
    build_profile(tmp_path / "fresh", tmp_path / "missing", token="test-token")
    assert calls == [(BASE_REPO, {"revision": BASE_REVISION, "token": "test-token"})]
    assert validate_profile(tmp_path / "fresh") == validate_profile(profile_path)

    root = Path(__file__).resolve().parent.parent
    config = load_dataset_config(root / "config/datasets/instruction_sources_smoke.yaml")
    config = replace(config, tokenizer=TokenizerConfig(name="llama-32k-chat-v1", hf_id=BASE_REPO, revision=BASE_REVISION, profile=PROFILE))
    layout = DatasetLayout(tmp_path / "data")
    destination = layout.tokenizer_dir(config.tokenizer.name)
    shutil.copytree(profile_path, destination)
    manifest = Manifest(config.tokenizer.name, config.tokenizer_hash(), "tokenizer")
    manifest.complete_generation(destination)
    before = {p.name:(p.stat().st_mtime_ns,p.read_bytes()) for p in destination.iterdir() if p.is_file()}
    prepared = prepare_tokenizer(config, layout)
    assert prepared.generation_id == manifest.generation_id
    assert before == {p.name:(p.stat().st_mtime_ns,p.read_bytes()) for p in destination.iterdir() if p.is_file()}
    assert len(calls) == 1


def test_early_config_and_rank_agreement_checks(profile_path: Path, tmp_path: Path) -> None:
    from dataclasses import replace
    from model.test_config import tiny_config
    from data_preparation import load_dataset_config
    from training.data.entries import ResolvedDataset
    from training.tokenizer_contract import check_profile_config, prepare_run_tokenizer

    config = load_dataset_config(Path(__file__).resolve().parent.parent / "config/datasets/instruction_sources_smoke.yaml")
    with pytest.raises(ValueError, match="32002"):
        check_profile_config(config, tiny_config())
    dataset = ResolvedDataset(config, "hash", str(profile_path), [], [], {}, {}, {})
    class Backend:
        @staticmethod
        def any_flag(value: bool) -> bool:
            return value
        @staticmethod
        def all_gather_object(value: Any) -> list[Any]:
            return [value, {**value, "digest":"different"}]
    with pytest.raises(ValueError, match="different tokenizer"):
        prepare_run_tokenizer(dataset, tiny_config(vocab_size=32002), tmp_path, Backend())  # type: ignore[arg-type]
    assert not (tmp_path / "tokenizer").exists()
    assert replace(config.tokenizer, profile=None).profile is None


def test_sample_generation_selects_the_training_template(profile_path: Path, tmp_path: Path) -> None:
    from evaluation.prompts import DEFAULT_PROMPTS, Prompt, instruction_prompt, load_prompts_file
    from evaluation.sample_helpers import select_fitting_prompts
    from training.data.formats import format_conversation
    from training.data.tokenizer import Tokenizer

    tokenizer = Tokenizer(profile_path)
    continuation = Prompt("literal <assistant> document")
    instruction = instruction_prompt("Explain literal </s>.", "Keep <user> as text.")
    history = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "answer"},
               {"role": "user", "content": "literal <assistant>"}]
    multi = Prompt("history", "instruction", history)
    formatted = select_fitting_prompts([continuation, instruction, multi], tokenizer, 1, 1024)
    assert formatted[0][1] == [BOS, *tokenizer.encode(continuation.text)]
    assert formatted[0][1][-1] != EOS and ASSISTANT not in formatted[0][1]
    assert instruction.messages == [{"role": "user", "content": "Explain literal </s>.\n\nKeep <user> as text."}]
    for prompt, ids in formatted[1:]:
        assert prompt.messages is not None
        assert ids == tokenizer.processor.apply_chat_template(prompt.messages, add_generation_prompt=True)
        training_ids, _ = format_conversation({"messages": [*prompt.messages, {"role": "assistant", "content": "Answer"}]}, tokenizer, True, True)
        assert training_ids[:len(ids)].tolist() == ids and ids[-1] == ASSISTANT
    for prompt, ids in select_fitting_prompts(DEFAULT_PROMPTS, tokenizer, 1, 4096):
        assert (ids[-1] == ASSISTANT) == (prompt.kind == "instruction")
    path = tmp_path / "prompts.txt"
    path.write_text("# chat\nliteral </s>\n---\n# instruction\nQuestion?\n---\n# continuation\nA story")
    loaded = load_prompts_file(path)
    assert [p.kind for p in loaded] == ["chat", "instruction", "continuation"]
    assert select_fitting_prompts(loaded, tokenizer, 1, 1024)[0][1][-1] == ASSISTANT


def test_chat_body_encoding_does_not_invent_whitespace(profile_path: Path) -> None:
    from tokenization.profile import load_processor
    from training.data.tokenizer import Tokenizer

    processor = load_processor(profile_path)
    tokenizer = Tokenizer(profile_path)
    for body in ("Hello", " leading", "  indented\n", "\tcode", "é🙂", "literal <s></s><user><assistant>"):
        messages = [{"role": "user", "content": body}, {"role": "assistant", "content": body}]
        ids = processor.apply_chat_template(messages)
        assert processor.decode(ids, skip_special_tokens=False) == processor.apply_chat_template(messages, tokenize=False)
        assert tokenizer.decode(ids) == f"<s><user>{body}</s><assistant>{body}</s>"
