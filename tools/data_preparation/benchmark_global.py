# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Offline, disposable final-pass comparison, including original-code baselines.

Run as a script, e.g. uv run --no-sync python tools/data_preparation/benchmark_global.py
--output-root /tmp/global-benchmark --baseline-repo /tmp/original-checkout.
Every trial has fresh candidates/output and an owned process group with time/RSS limits.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-root", type=Path, required=True, help="new disposable directory; must not exist")
    result.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    result.add_argument("--baseline-repo", type=Path, help="original checkout/archive; otherwise current code without reuse")
    result.add_argument("--cases", nargs="+", choices=("baseline", "reuse", "hash2", "hash4", "hash8"),
                        default=["baseline", "reuse", "hash2", "hash4"])
    result.add_argument("--rows", type=positive, default=24000, help="first-source rows; next two sources have rows/4 each")
    result.add_argument("--text-chars", type=positive, default=512)
    result.add_argument("--bloom-mb", type=positive, default=8)
    result.add_argument("--trials", type=positive, default=3)
    result.add_argument("--max-input-mb", type=positive, default=256)
    result.add_argument("--max-rss-mb", type=positive, default=4096)
    result.add_argument("--timeout", type=positive, default=180, help="seconds per trial, including fixture setup")
    result.add_argument("--worker", choices=("baseline", "reuse", "hash2", "hash4", "hash8"), help=argparse.SUPPRESS)
    return result


def run_worker(args: argparse.Namespace) -> None:
    # Defer all project imports so this same harness can exercise an original archive.
    sys.path.insert(0, str(args.repo.resolve()))
    from data_preparation.lib.dataset_config import DatasetConfig, SourceConfig, StageConfig, TokenizerConfig
    from data_preparation.lib.layout import DatasetLayout
    from data_preparation.lib.stages.global_build import build_global_source
    from data_preparation.lib.stages.global_dedup import GlobalAdmission, GlobalFrontier
    from data_preparation.lib.storage.manifest import Manifest
    from data_preparation.lib.storage.parquet import build_row_table, publish_shard

    names = ("a", "b", "c")
    config = DatasetConfig(
        tokenizer=TokenizerConfig(name="synthetic", kind="synthetic"),
        sources={name: SourceConfig(kind="pretrain", loader="synthetic") for name in names},
        stages=[StageConfig(name="pretrain", tokens=10, train=dict.fromkeys(names, 1 / 3), val={"a": 1.0})],
        training_target_sequence_length=64, token_count="estimate", bloom_dedup_memory_mb=args.bloom_mb,
    )
    root = args.output_root / "dataset"
    local, layout = DatasetLayout(root), DatasetLayout(root).for_config(config)
    fixture_digest = hashlib.sha256()
    counts = (args.rows, max(1, args.rows // 4), max(1, args.rows // 4))
    for name, count in zip(names, counts, strict=True):
        directory = local.processed_dir(name)
        manifest = Manifest(source=name, source_hash=config.processed_hash(name), stage="processed",
                            columns=["text", "source", "tokens", "hash"], token_count="estimate")
        for first in range(0, count, 4096):
            rows = []
            for index in range(first, min(first + 4096, count)):
                prefix = "shared" if index % 5 == 0 else name
                text = f"{prefix} {index} " + (" Mixed CASE\twords\n" * (args.text_chars // 18 + 1))[:args.text_chars]
                fixture_digest.update(text.encode())
                rows.append({"text": text, "source": name, "hash": index, "tokens": 64})
            path = publish_shard(build_row_table(rows), directory / f"shard_{first // 4096:05d}.parquet")
            manifest.add_shard(path.name, len(rows), len(rows) * 64)
        manifest.complete_generation(directory)
    restoration: list[dict[str, float | int]] = []
    original = GlobalAdmission.__init__

    def counted(self: GlobalAdmission, *pos: Any, **kw: Any) -> None:
        started = time.perf_counter()
        original(self, *pos, **kw)
        restoration.append({"keys": kw["frontier"].retained, "seconds": time.perf_counter() - started})

    GlobalAdmission.__init__ = counted  # type: ignore[method-assign]
    frontier = GlobalFrontier(names, args.bloom_mb)
    options: dict[str, Any] = {}
    hashing: Any = None
    if args.worker != "baseline":
        from data_preparation.lib.stages.global_session import GlobalAdmissionSession
        from data_preparation.lib.stages.global_hash_workers import GlobalHashWorkers
        options["session"] = GlobalAdmissionSession(config, layout, frontier, lambda: ())
        hashing = GlobalHashWorkers(int(args.worker[4:]) if args.worker.startswith("hash") else 1)
        options["hash_workers"] = hashing
    started = time.perf_counter()
    cpu_before = resource.getrusage(resource.RUSAGE_SELF)
    children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    by_source = {}
    try:
        for name in names:
            tick = time.perf_counter()
            frontier, _ = build_global_source(config, name, layout, frontier, rows_target=1, exhausted=True, **options)
            by_source[name] = time.perf_counter() - tick
    finally:
        if hashing is not None:
            hashing.close()
    elapsed = time.perf_counter() - started
    cpu = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    semantic = {}
    for name in names:
        directory = layout.processed_dir(name)
        output = Manifest.load(directory)
        assert output is not None
        semantic[name] = {
            "frontier": output.extra["global_frontier"], "stats": output.stats,
            "shards": [hashlib.sha256((directory / shard.name).read_bytes()).hexdigest() for shard in output.shards],
        }
    source_digest = hashlib.sha256()
    for filename in ("global_build.py", "global_dedup.py", "global_session.py", "global_hash_workers.py", "global_output.py", "exact_dedup.py"):
        source_file = args.repo / "data_preparation/lib/stages" / filename
        if source_file.exists():
            source_digest.update(filename.encode() + source_file.read_bytes())
    result = {
        "source_digest": source_digest.hexdigest(),
        "case": args.worker, "rows": sum(counts), "wall_seconds": elapsed, "source_seconds": by_source,
        "restorations": restoration, "rows_per_second": sum(counts) / elapsed,
        "cpu_seconds_parent_and_joined_children": cpu.ru_utime + cpu.ru_stime - cpu_before.ru_utime - cpu_before.ru_stime
        + children.ru_utime + children.ru_stime - children_before.ru_utime - children_before.ru_stime,
        "semantic": semantic, "fixture_digest": fixture_digest.hexdigest(), "config_hash": config.config_hash(),
        "versions": {name: importlib.metadata.version(name) for name in ("pyarrow", "rbloom")},
        "python": platform.python_version(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
    }
    (args.output_root / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def process_tree_rss(pid: int) -> int:
    """Linux sampled sum; shared pages count once per process (not unique physical RAM)."""
    pending, seen, total = [pid], set(), 0
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            status = Path(f"/proc/{current}/status").read_text()
            total += next((int(line.split()[1]) * 1024 for line in status.splitlines() if line.startswith("VmRSS:")), 0)
            for task in Path(f"/proc/{current}/task").iterdir():
                pending.extend(int(child) for child in (task / "children").read_text().split())
        except FileNotFoundError:
            pass  # process/thread exited between samples
    return total


def trial(args: argparse.Namespace, case: str, number: int) -> dict[str, Any]:
    directory = args.output_root / f"{case}-{number}"
    directory.mkdir()
    repo = args.baseline_repo if case == "baseline" and args.baseline_repo else args.repo
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", case, "--output-root", str(directory),
               "--repo", str(repo), "--rows", str(args.rows), "--text-chars", str(args.text_chars),
               "--bloom-mb", str(args.bloom_mb), "--max-input-mb", str(args.max_input_mb)]
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    peak = 0
    started = time.monotonic()
    with (directory / "worker.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
        try:
            while process.poll() is None:
                peak = max(peak, process_tree_rss(process.pid))
                if time.monotonic() - started > args.timeout or peak > args.max_rss_mb << 20:
                    raise RuntimeError(f"{case}: exceeded time/RSS limit; see {directory}")
                time.sleep(0.1)
            if process.returncode:
                raise RuntimeError(f"{case}: exited {process.returncode}; see {directory / 'worker.log'}")
        finally:
            # This group is owned by this isolated benchmark, never a live preparation job.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    result: dict[str, Any] = json.loads((directory / "result.json").read_text())
    result["sampled_peak_tree_rss_bytes"] = peak
    return result


def main() -> None:
    args = parser().parse_args()
    total_rows = args.rows + 2 * max(1, args.rows // 4)
    if total_rows > 1_000_000 or total_rows * (args.text_chars + 128) > args.max_input_mb << 20:
        raise ValueError("fixture exceeds the row/input-byte limit; reduce rows/text-chars")
    if args.worker:
        run_worker(args)
        return
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=False)
    results = [trial(args, case, number) for number in range(args.trials) for case in args.cases]
    reference = {key: results[0][key] for key in ("semantic", "fixture_digest", "config_hash")}
    if any(any(result[key] != value for key, value in reference.items()) for result in results):
        raise RuntimeError("benchmark output parity failed; inspect retained result.json files")
    summary = {case: {"median_seconds": statistics.median(r["wall_seconds"] for r in results if r["case"] == case),
                      "seconds": [r["wall_seconds"] for r in results if r["case"] == case]}
               for case in args.cases}
    filesystem = subprocess.run(["findmnt", "-T", str(args.output_root), "-n", "-o", "FSTYPE,SOURCE"],
                                check=True, capture_output=True, text=True).stdout.strip()
    report = {"filesystem": filesystem, "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "cache_conditions": "fresh fixture writes, warm input cache; no host cache clearing",
              "rss_method": "100ms sampled sum of process-tree VmRSS, including setup; shared pages counted per process",
              "summary": summary, "trials": results}
    (args.output_root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
