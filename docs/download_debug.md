# Live preparation diagnostics

Append `--debug` to a `prepare` or `download` command to log a pipeline overview every five seconds.
Use `--debug 2` for two-second reports; positive fractional intervals are supported. The flag has no
effect on a dry run and does not change dataset identity or preparation settings.

Reports appear in the terminal/dashboard and the dataset's `build.log`. Each identifies the process
PID, process name, source, and instrumented thread name/native ID. The main coordinator, download
threads, token workers, read callbacks, and spawned MinHash/decontamination workers are covered.

Timings describe the **latest interval**, including unfinished operations (`active` and `oldest`).
They are inclusive thread-wall times: nested stages and different threads overlap, so their sum can
exceed the interval. User/kernel CPU seconds are process-wide, including native tokenizer threads.

During quiet network periods, look for:

- `WAIT:queue_put`: fetching is blocked behind tokenization or storage.
- `WAIT:queue_get`: a token worker needs input; inspect `input_next`, `remote_read`, and decoding.
- `encode_batch` or `truncate_batch`: inspect process CPU alongside encoding/result-handling time.
- `parquet_compress_write`, `file_sync`, or `publish_manifest`: output work may be delaying progress.
- `WAIT:wait_jobs` on MainThread: the coordinator is waiting for workers, usually normal.
- `WAIT:worker_result_wait`: check the corresponding cleaning-process reports.

Read-byte counters describe completed application reads, not HTTP wire traffic. A long pending read
may show no completed bytes while its active duration increases. `pool_idle_or_dispatch` includes
IPC/deserialization and time outside worker tasks; it is not a measurement of pure kernel waiting.

Worker processes send reports to the parent through a bounded logging queue. Dropped reports are
counted when the queue fills; an abrupt process exit may lose its last report. Runtime helper
processes, such as multiprocessing's resource tracker, perform no instrumented pipeline stages.

Debugging adds timing overhead. If Python cannot run because of a GIL or kernel stall, reports may
arrive late; the sample timestamp and actual interval expose that delay. This is not a kernel profiler.
