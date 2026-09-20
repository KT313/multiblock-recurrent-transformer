# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Compare original/candidate preparation on bounded JSONL input in fresh offline processes.

Example: uv run --no-sync python -m tools.data_preparation.benchmark_download
  --tokenizer dataset/tokenizers/llama-32k --fixture /tmp/original-text.jsonl --output-root /tmp/replay
  --mode pipeline --threads 8 --jobs 8 --repeats 3
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from tools.data_preparation.replay import load_fixture, make_counter, run_pipeline, summarize_output


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--tokenizer", required=True, type=Path)
    result.add_argument("--fixture", required=True, type=Path)
    result.add_argument("--output-root", required=True, type=Path)
    result.add_argument("--mode", choices=("truncate", "count", "pipeline"), default="pipeline")
    result.add_argument("--threads", type=positive, default=4)
    result.add_argument("--jobs", type=positive, default=1)
    result.add_argument("--repeats", type=positive, default=3)
    result.add_argument("--cap", type=positive, default=16_384)
    result.add_argument("--max-input-mb", type=positive, default=64)
    result.add_argument("--max-rows", type=positive, default=100_000)
    result.add_argument("--max-rss-mb", type=positive, default=4096)
    result.add_argument("--max-output-mb", type=positive, default=1024)
    result.add_argument("--timeout", type=positive, default=300)
    variants = ("original", "candidate", "builtin", "rust")
    result.add_argument("--variants", nargs="+", choices=variants, default=["original", "candidate"])
    result.add_argument("--native-library", type=Path)
    result.add_argument("--worker", choices=variants, help=argparse.SUPPRESS)
    return result


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_fixture(args.fixture, args.max_input_mb << 20, args.max_rows)
    counter = make_counter(args.tokenizer, args.worker)
    counter.truncate_many([row.text for row in rows[:256]], args.cap - 2)
    started = time.perf_counter()
    cpu = time.process_time()
    digest = hashlib.sha256()
    tokens = 0
    results: list[tuple[str, int]] = []
    if args.mode == "pipeline":
        run_pipeline(rows, args.tokenizer, args.output_root / "shards", args.cap, args.jobs, args.worker)
    else:
        for start in range(0, len(rows), 256):
            texts = [row.text for row in rows[start:start + 256]]
            if args.mode == "count":
                counts = counter.count_many(texts)
                values = list(zip(texts, counts, strict=True))
            else:
                values = counter.truncate_many(texts, args.cap - 2)
            results.extend(values)
            del values
    elapsed, cpu_seconds = time.perf_counter() - started, time.process_time() - cpu
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux
    if args.mode == "pipeline":
        output = summarize_output(args.output_root / "shards")
        semantic = output["sources"]
        tokens = sum(source["tokens"] for source in semantic.values())
        compressed_bytes = output["compressed_bytes"]
    else:
        tokens = sum(count for _, count in results)
        for value in results:
            digest.update((json.dumps(value, ensure_ascii=False) + "\n").encode())
        semantic = {"digest": digest.hexdigest(), "rows": len(rows), "tokens": tokens}
        compressed_bytes = 0
    return {"variant": args.worker, "wall_seconds": elapsed, "cpu_seconds": cpu_seconds,
            "peak_rss_bytes": peak_rss, "input_rows": len(rows), "input_characters": sum(len(r.text) for r in rows),
            "tokens": tokens, "tokens_per_second": tokens / elapsed, "rows_per_second": len(rows) / elapsed,
            "compressed_bytes": compressed_bytes, "semantic": semantic}


def run_bounded(command: list[str], directory: Path, args: argparse.Namespace, environment: dict[str, str]) -> None:
    """Bound each isolated replay's wall time, RSS and scratch writes; retain failure logs."""
    started = time.monotonic()
    with (directory / "worker.log").open("w") as log:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    code = process.wait(timeout=0.25)
                    if code:
                        raise RuntimeError(f"benchmark worker exited {code}; see {directory / 'worker.log'}")
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() - started > args.timeout:
                        raise RuntimeError("benchmark exceeded wall-time limit") from None
                    try:
                        status = Path(f"/proc/{process.pid}/status").read_text()
                    except FileNotFoundError:
                        continue
                    rss = next((int(line.split()[1]) * 1024 for line in status.splitlines() if line.startswith("VmRSS:")), 0)
                    if rss > args.max_rss_mb << 20:
                        raise RuntimeError("benchmark exceeded RSS limit") from None
                    if output_size(directory) > args.max_output_mb << 20:
                        raise RuntimeError("benchmark exceeded output limit") from None
        finally:
            if process.poll() is None:
                process.kill()  # owned benchmark only; real preparation never spawns here
            process.wait()
    if output_size(directory) > args.max_output_mb << 20:
        raise RuntimeError("benchmark exceeded output limit")


def output_size(directory: Path) -> int:
    size = 0
    for path in directory.rglob("*"):
        try:
            if path.is_file():
                size += path.stat().st_size
        except FileNotFoundError:
            pass  # a writer atomically renamed its temporary between listing and stat
    return size


def read_environment() -> dict[str, Any]:
    import pyarrow as pa

    def git(*arguments: str) -> str:
        return subprocess.check_output(["git", *arguments], text=True).strip()

    cgroup = Path("/proc/self/cgroup").read_text()
    limits: dict[str, dict[str, str]] = {}
    relative = next((line[3:] for line in cgroup.splitlines() if line.startswith("0::")), None)
    if relative is not None:
        root = Path("/sys/fs/cgroup")
        current = root / relative.lstrip("/")
        while current.is_relative_to(root):
            limits[str(current)] = {name: (current / name).read_text() for name in ("cpu.max", "cpu.stat", "cpuset.cpus.effective") if (current / name).is_file()}
            current = current.parent
    names = ("TOKENIZERS_PARALLELISM", "RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "SLURM_CPUS_PER_TASK")
    return {"python": platform.python_version(), "git_revision": git("rev-parse", "HEAD"),
            "tracked_diff_sha256": hashlib.sha256(git("diff", "HEAD", "--", "data_preparation", "tokenization").encode()).hexdigest(),
            "versions": {name: importlib.metadata.version(name) for name in ("tokenizers", "pyarrow", "fsspec", "huggingface_hub")},
            "affinity": sorted(os.sched_getaffinity(0)), "environment": {name: os.environ.get(name) for name in names},
            "arrow_cpu_threads": pa.cpu_count(), "arrow_io_threads": pa.io_thread_count(), "cgroups": limits}


def main() -> None:
    args = parser().parse_args()
    if args.cap < 2:
        raise ValueError("--cap must leave room for BOS and EOS (>= 2)")
    if len(set(args.variants)) != len(args.variants):
        raise ValueError("--variants must be unique")
    if args.worker:
        before = read_environment()
        trial_report = run_worker(args)
        trial_report["environment_before"] = before
        trial_report["environment_after"] = read_environment()
        (args.output_root / "result.json").write_text(json.dumps(trial_report, indent=2) + "\n")
        return
    # Validate before creating anything. Only unique child directories are ever passed to writers.
    args.fixture, args.tokenizer, args.output_root = args.fixture.resolve(), args.tokenizer.resolve(), args.output_root.resolve()
    for protected in (args.fixture, args.tokenizer, Path("dataset").resolve()):
        if args.output_root == protected or args.output_root.is_relative_to(protected) or protected.is_relative_to(args.output_root):
            raise ValueError("output root overlaps input/tokenizer/production dataset")
    load_fixture(args.fixture, args.max_input_mb << 20, args.max_rows)
    if not (args.tokenizer / "tokenizer.json").is_file():
        raise ValueError("--tokenizer must contain a saved tokenizer.json")
    if args.threads > len(os.sched_getaffinity(0)):
        raise ValueError("--threads exceeds effective CPU affinity")
    args.output_root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="download-benchmark-", dir=args.output_root))
    environment = dict(os.environ, RAYON_NUM_THREADS=str(args.threads), TOKENIZERS_PARALLELISM="true",
                       HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", DATA_PREP_PROGRESS="0")
    if args.native_library is not None:
        environment["PREPARER_NATIVE_LIBRARY"] = str(args.native_library.resolve(strict=True))
    report: dict[str, Any] = {"schema_version": 1, "mode": args.mode, "threads": args.threads, "jobs": args.jobs,
                             "fixture_sha256": file_hash(args.fixture), "tokenizer_files": {}, "trials": []}
    report["tokenizer_files"] = {str(path.relative_to(args.tokenizer)): file_hash(path) for path in sorted(args.tokenizer.rglob("*")) if path.is_file()}
    tooling = Path(__file__).parent
    report["experiment_files"] = {str(path.relative_to(tooling)): file_hash(path) for path in sorted(tooling.rglob("*"))
                                  if path.is_file() and (path.suffix in (".py", ".rs") or path.name in ("Cargo.toml", "Cargo.lock"))}
    report["native_library_sha256"] = file_hash(args.native_library) if args.native_library is not None else None
    report["environment_before"] = read_environment()
    for repeat in range(args.repeats):
        for variant in (args.variants if repeat % 2 == 0 else list(reversed(args.variants))):
            directory = run / f"{repeat}-{variant}"
            directory.mkdir()
            command = [sys.executable, "-m", __spec__.name if __spec__ else "tools.data_preparation.benchmark_download",
                       "--worker", variant, "--tokenizer", str(args.tokenizer), "--fixture", str(args.fixture),
                       "--output-root", str(directory), "--mode", args.mode, "--cap", str(args.cap), "--jobs", str(args.jobs),
                       "--max-input-mb", str(args.max_input_mb), "--max-rows", str(args.max_rows)]
            run_bounded(command, directory, args, environment)
            trial = json.loads((directory / "result.json").read_text())
            if trial["peak_rss_bytes"] > args.max_rss_mb << 20:
                raise RuntimeError(f"benchmark exceeded RSS limit; retained diagnostics at {run}")
            if report["trials"] and trial["semantic"] != report["trials"][0]["semantic"]:
                raise RuntimeError(f"semantic mismatch; retained diagnostics at {run}")
            report["trials"].append(trial)
            print(f"{variant}: {trial['wall_seconds']:.3f}s, {trial['tokens_per_second']:.0f} tokens/s", flush=True)
    report["environment_after"] = read_environment()
    medians = {v: statistics.median(t["wall_seconds"] for t in report["trials"] if t["variant"] == v) for v in args.variants}
    report["median_seconds"] = medians
    baseline = args.variants[0]
    report["speedup_vs_first"] = {name: medians[baseline] / value for name, value in medians.items()}
    (run / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Exact output parity. Speedups versus {baseline}: {report['speedup_vs_first']}. Report: {run / 'report.json'}")


if __name__ == "__main__":
    main()
