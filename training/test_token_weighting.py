"""Independent token-mean gradient/update oracles, including actual DDP averaging."""
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn
from torch import Tensor

from model import RecurrentGPT
from training.backend.ddp import DDPBackend
from training.backend.single_device import SingleDeviceBackend
from training.data.packing import PackedBatch
from training.evaluation import evaluate
from training.steps import NonFiniteLossError, TrainingProgress
from training.step import run_one_optimizer_step
from training.test_step import reference_settings, reference_stage_manager


class TokenOracle(RecurrentGPT):
    """Fixed per-token linear losses: no recurrent random draws or partition-dependent computation."""

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.register_buffer("freqs_cis", torch.empty(0))
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.config = type("Config", (), {"vocab_size": 8, "mean_recurrence": [1]})()
        self.ignore_index = -100
        self.calls = 0

    def forward(self, input_ids: Tensor, *args: Any, **kwargs: Any) -> dict[str, Tensor | None]:
        labels = cast(Tensor | None, kwargs.get("labels"))
        assert labels is not None
        self.calls += 1
        counted = self.mask_labels(labels) != self.ignore_index
        count = counted.sum()
        tokens = (self.weight * input_ids.float()).masked_fill(~counted, 0)
        numerator = tokens.sum()
        return {"loss_sum": numerator, "supervised_count": count, "loss": numerator / count, "token_losses": tokens}


def pack(value: int, count: int, length: int = 1024) -> PackedBatch:
    ids = torch.full((1, length), value, dtype=torch.int64)
    labels = torch.full_like(ids, -100)
    labels[:, :count] = 2  # EOS is supervised; ignored tails can contain the same input ID.
    return PackedBatch(ids, labels, ["s"], torch.arange(length).unsqueeze(0), torch.zeros_like(ids).int(),
                       length - count, [length - count])


def run_oracle(backend: SingleDeviceBackend, batches: list[PackedBatch], clip: float = 0.5) -> tuple[Tensor, Tensor, Tensor]:
    settings = reference_settings(tokens_per_micro_batch=1024, micro_batches_per_step=len(batches) * backend.world_size,
                                  log_gradient_metrics_interval=0, grad_clip=clip, warmup_steps=0, cooldown_steps=0)
    model = backend.setup_model(TokenOracle())
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    result = run_one_optimizer_step(settings, backend, model, optimizer, reference_stage_manager(settings),
                                    iter(batches), TrainingProgress(step=2))
    plain = backend.plain_model(model)
    assert isinstance(plain, TokenOracle)
    assert plain.calls == len(batches)
    # Independent single concatenation/mean + clipping + actual SGD step.
    return result.loss, result.grad_norm, plain.weight.detach()


@pytest.mark.parametrize("clip", [0.5, 10.0])
def test_token_mean_matches_independent_clipped_optimizer(clip: float) -> None:
    loss, norm, weight = run_oracle(SingleDeviceBackend("cpu", "32"), [pack(1, 100), pack(3, 900)], clip)
    reference = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([reference], lr=3e-4)
    expected_loss: Tensor = (reference * torch.tensor([1.0] * 100 + [3.0] * 900)).mean()
    torch.autograd.backward(expected_loss)
    expected_norm = torch.nn.utils.clip_grad_norm_([reference], clip)
    optimizer.step()
    torch.testing.assert_close(loss, torch.tensor(2.8))
    torch.testing.assert_close(norm, expected_norm)
    torch.testing.assert_close(weight, reference.detach(), rtol=0, atol=0)


def test_repartitioning_zero_contributions_and_zero_gradient_targets() -> None:
    backend = SingleDeviceBackend("cpu", "32")
    first = run_oracle(backend, [pack(0, 100), pack(3, 900), pack(7, 0)])
    second = run_oracle(backend, [pack(3, 450), pack(0, 100), pack(3, 450)])
    for actual, expected in zip(first, second):
        torch.testing.assert_close(actual, expected)
    third = run_oracle(backend, [pack(0, 100), pack(3, 900)])
    for actual, expected in zip(first, third):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(first[0], torch.tensor(2.7))


def test_empty_update_fails_before_optimizer() -> None:
    with pytest.raises(RuntimeError, match="No supervised target tokens"):
        run_oracle(SingleDeviceBackend("cpu", "32"), [pack(1, 0), pack(3, 0)])


def test_nonfinite_valid_targets_still_fail() -> None:
    class Nonfinite(TokenOracle):
        def forward(self, *args: Any, **kwargs: Any) -> dict[str, Tensor | None]:
            output = super().forward(*args, **kwargs)
            assert output["loss_sum"] is not None
            output["loss_sum"] = output["loss_sum"] * float("nan")
            return output
    backend = SingleDeviceBackend("cpu", "32")
    model = Nonfinite()
    settings = reference_settings(log_gradient_metrics_interval=0)
    with pytest.raises(NonFiniteLossError, match="Loss is nan"):
        run_one_optimizer_step(settings, backend, model, torch.optim.SGD(model.parameters(), lr=1),
                               reference_stage_manager(settings), iter([pack(1, 2), pack(3, 2)]), TrainingProgress(step=2))
    assert model.weight.item() == 1


def test_native_sum_statistics_masks_and_prompt_context(tiny_model: RecurrentGPT) -> None:
    tiny_model.eval()
    ids = torch.tensor([[20, 21, 22, 23]])
    labels = torch.tensor([[-100, -100, 2, tiny_model.config.vocab_size]])
    embedding_outputs: list[Tensor] = []

    def retain_context(module: torch.nn.Module, inputs: Any, output: Tensor) -> None:
        output.retain_grad()
        embedding_outputs.append(output)

    hook = tiny_model.transformer.wte.register_forward_hook(retain_context)
    torch.manual_seed(12)
    result = tiny_model(ids, labels=labels, num_steps=[(1, 0), (1, 0)], return_loss_statistics=True)
    assert result["supervised_count"] is not None and result["supervised_count"].item() == 1
    assert result["loss_sum"] is not None
    result["loss_sum"].backward()
    hook.remove()
    grad = embedding_outputs[0].grad
    assert grad is not None and grad[0, 0].abs().sum() > 0  # through context, independent of tied head gradients
    tiny_model.zero_grad(set_to_none=True)
    empty = tiny_model(ids, labels=torch.full_like(ids, -100), return_loss_statistics=True)
    assert empty["loss"] is not None and torch.isnan(empty["loss"])
    assert empty["loss_sum"] is not None and empty["loss_sum"].item() == 0
    empty["loss_sum"].backward()
    assert all(torch.count_nonzero(p.grad) == 0 for p in tiny_model.parameters() if p.grad is not None)


def test_validation_uses_token_counts_and_omits_empty_sources(caplog: pytest.LogCaptureFixture) -> None:
    from training.data.collate import Batch
    backend = SingleDeviceBackend("cpu", "32")
    settings = reference_settings(eval_iters=3, partial_depth_eval=[1, 2])
    batches = [Batch(p.input_ids, p.labels, [source])
               for p, source in [(pack(1, 100), "a"), (pack(3, 900), "b"), (pack(7, 0), "empty")]]
    result = evaluate(settings, backend, TokenOracle(), batches)
    assert "val_loss/empty" not in result
    assert "no supervised target tokens" in caplog.text
    for key in ("val_loss", "val_loss_1", "val_loss_2"):
        torch.testing.assert_close(result[key], torch.tensor(2.8))
    torch.testing.assert_close(result["val_ppl"], torch.tensor(2.8).exp())
    torch.testing.assert_close(result["val_loss"], (result["val_loss/a"] * 100 + result["val_loss/b"] * 900) / 1000)
    with pytest.raises(RuntimeError, match="no supervised target tokens"):
        evaluate(settings, backend, TokenOracle(), batches[-1:])


class GlooBackend(DDPBackend):
    """Use production reducer/no_sync with a bounded file rendezvous, without torchrun environment mutation."""
    def __init__(self, rank: int, rendezvous: str) -> None:
        SingleDeviceBackend.__init__(self, "cpu", "32")
        dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=40))
        self.rank, self.world_size, self.is_main = rank, 2, rank == 0


def ddp_worker(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    backend = GlooBackend(rank, rendezvous)
    try:
        for empty_rank in (False, True):
            batches = ([pack(1, 100), pack(0, 0)] if rank == 0 else [pack(3, 400), pack(3, 500)])
            if empty_rank and rank == 0:
                batches = [pack(1, 0), pack(0, 0)]
            loss, norm, weight = run_oracle(backend, batches)
            expected = 3.0 if empty_rank else 2.8
            torch.testing.assert_close(loss, torch.tensor(expected))
            torch.testing.assert_close(norm, torch.tensor(expected))
            replicas = backend.all_gather_object(weight)
            torch.testing.assert_close(replicas[0], replicas[1], rtol=0, atol=0)
            from training.data.collate import Batch
            settings = reference_settings(eval_iters=4, partial_depth_eval=[1, 2])
            validation = [Batch(p.input_ids, p.labels, ["s"]) for p in batches]
            metrics = evaluate(settings, backend, backend.setup_model(TokenOracle()), validation)
            for key in ("val_loss", "val_loss_1", "val_loss_2", "val_loss/s"):
                torch.testing.assert_close(metrics[key], torch.tensor(expected))
        with pytest.raises(RuntimeError, match="No supervised target tokens"):
            run_oracle(backend, [pack(1, 0), pack(0, 0)])
        invalid = pack(1, 1)
        if rank == 1:
            invalid = invalid._replace(input_ids=torch.full((1, 1024), float("nan")))
        with pytest.raises(NonFiniteLossError, match="Loss is nan"):
            run_oracle(backend, [invalid, pack(3, 2)])
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
def test_actual_two_rank_token_mean(tmp_path: Path) -> None:
    context = cast(Any, spawn(ddp_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=False))  # type: ignore[no-untyped-call]  # torch multiprocessing stub
    try:
        import time
        deadline = time.monotonic() + 90
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("two-rank token-weighting oracle exceeded 90 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def test_source_counts_remain_integer_above_float32_exact_range() -> None:
    from training.evaluation import _add_source_token_losses
    model = TokenOracle()
    sums = {"s": (torch.tensor(0.0), torch.tensor(2**24, dtype=torch.int64))}
    _add_source_token_losses(sums, model, torch.tensor([[0.0, 100.0, 100.0]]),
                             torch.tensor([[2, 8, -100]]), ["s"])
    assert sums["s"][1].dtype == torch.int64 and sums["s"][1].item() == 2**24 + 1
    assert sums["s"][0].item() == 0  # zero loss still counts; real-vocabulary-invalid labels do not


@pytest.mark.parametrize("options", [{}, {"labels": torch.tensor([[2]]), "return_logits": True},
                                      {"labels": torch.tensor([[2]]), "return_token_losses_chunked_nograd": True}])
def test_invalid_statistics_request_fails_before_rng_draw(tiny_model: RecurrentGPT, options: dict[str, Any]) -> None:
    before = torch.get_rng_state()
    with pytest.raises(ValueError, match="return_loss_statistics requires"):
        tiny_model(torch.tensor([[1]]), return_loss_statistics=True, **options)
    assert torch.equal(before, torch.get_rng_state())


def test_statistics_output_schema_is_opt_in(tiny_model: RecurrentGPT) -> None:
    ids, labels = torch.tensor([[20, 21]]), torch.tensor([[-100, 2]])
    standard_keys = {"loss", "logits", "token_losses", "log_ppl"}
    torch.manual_seed(18)
    default = tiny_model(ids, labels=labels)
    assert set(default) == standard_keys
    torch.manual_seed(18)
    statistics = tiny_model(ids, labels=labels, return_loss_statistics=True)
    assert set(statistics) == standard_keys | {"loss_sum", "supervised_count"}
    assert default["loss"] is not None and statistics["loss"] is not None
    torch.testing.assert_close(default["loss"], statistics["loss"])
    assert statistics["supervised_count"] is not None and statistics["supervised_count"].item() == 1
