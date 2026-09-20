# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Small real harness tasks with in-memory data, for offline qualification only."""
from typing import Any


def build_offline_task(kind: str = "multiple_choice", count: int = 3, name: str | None = None) -> Any:
    from datasets import Dataset, DatasetDict
    from lm_eval.api.task import ConfigurableTask

    class LocalTask(ConfigurableTask):  # type: ignore[misc]
        def download(self, dataset_kwargs: Any = None, **kwargs: Any) -> None:
            rows = [{"question": "tok_3 " * (1 + index % 3), "choices": ["tok_4", "tok_5 tok_6"],
                     "answer": index % 2, "text": "tok_3 tok_4", "target": "tok_4"} for index in range(count)]
            self.dataset = DatasetDict({"test": Dataset.from_list(rows), "train": Dataset.from_list(rows)})

    config: dict[str, Any] = {
        "task": name or f"offline_{kind}", "dataset_path": "local_fixture", "output_type": kind,
        "test_split": "test", "fewshot_split": "train", "num_fewshot": None,
        "doc_to_text": "question", "doc_to_target": "target",
    }
    if kind == "multiple_choice":
        config.update(doc_to_choice="choices", doc_to_target="answer")
    elif kind == "generate_until":
        config.update(generation_kwargs={"until": ["STOP"], "max_gen_toks": 2, "do_sample": False})
    elif kind == "loglikelihood_rolling":
        config.update(doc_to_text="", doc_to_target="text")
    return LocalTask(config=config)


def build_offline_group() -> Any:
    from lm_eval.api.group import Group
    from lm_eval.config.group import AggMetricConfig
    group = Group("offline_group", aggregate_metric_list=[AggMetricConfig(metric="acc")])
    group.add(build_offline_task("multiple_choice", 9, "offline_subject_a"))
    group.add(build_offline_task("multiple_choice", 5, "offline_subject_b"))
    return group
