# Configuration history when resuming training

`run_config.json` and `model_config.json` preserve the first available configuration evidence in a run directory.
Training never rewrites an existing sidecar, including during rejected or repeatedly accepted resumes. A fresh run
publishes missing sidecars after successful setup and before the first optimizer update. Constructing the model
alone creates no configuration files.

A fresh start requires a destination without established run evidence. After resolving the requested checkpoint,
training checks this under the run lock, before dataset resolution, model setup, or logger initialization. Existing
configuration sidecars, a training report, accepted resume records, or checkpoint files cause an explanatory refusal
when no checkpoint was selected. This applies both to `resume: false` and to `resume: true` when no checkpoint can be
found. Choose a new `run_name` or `out_dir`, or resume from a valid checkpoint. Merely precreated empty run, checkpoint,
and resume directories are allowed. Training neither deletes existing evidence nor silently treats it as the new run's
configuration.

Each accepted resume adds `resumes/<UTC timestamp>-<unique attempt ID>.json` (schema version 1). Acceptance means
model/optimizer restoration, compatibility checks, logger initialization, and stream restoration succeeded and
training may continue. It does not mean a new optimizer update completed. An immediate stop, or a resume from the
final checkpoint, can therefore have an accepted record and zero new updates. Checkpoint metadata identifies
completed optimizer steps; the resume record's source checkpoint step identifies where the attempt began.

A record contains the source checkpoint's absolute path and step, destination run name and directory, acceptance
time, requested settings, effective model configuration and stage schedule, checkpoint differences, and the explicit
`allow_settings_change` and `allow_dataset_change` acknowledgements. Differences include operational settings such as
output paths, even when they do not require an override. Missing checkpoint fields remain distinguishable from null
values; compatibility code's legacy defaults are not invented as historical observations.

`requested_settings.optim_config` records the constructor request. `effective.run_settings` excludes that constructor
configuration; `effective.optimizer.parameter_groups_at_acceptance` records the actual restored hyperparameters,
including per-group decay and available optimizer-specific options. The recorded group LR is the LR at acceptance:
each subsequent update replaces it with the current run's stage schedule. Tensor parameters and optimizer moments
are not serialized. Only the documented optimizer hyperparameter allowlist is recorded. Standard configuration
schemas are used, and environment variables are never collected.

The dataset build ID and checkpoint build ID are included when available. Legacy missing IDs are explicitly
unavailable. Code provenance records the local checkout's Git HEAD when obtainable; cleanliness is `not_checked`,
so the revision does not claim that all executing code matches that commit. A missing Git checkout produces an
unavailable revision instead of a guessed identity.

A resume into a new directory, or an old run missing a sidecar, reconstructs that file from checkpoint metadata.
The sidecar retains the checkpoint's flat configuration fields and adds `_provenance` with
`origin: checkpoint_reconstruction`, its source checkpoint and step, and `original_run_configuration: unknown`.
This proves the settings recorded at that checkpoint, which may itself follow earlier configuration changes; it does
not recover the run's birth configuration. Consumers of reconstructed sidecars should treat `_provenance` as metadata,
not a Settings or model option. Existing sidecars are neither modified nor retroactively marked as verified.

Publication runs on rank zero under the existing run lock. Peers exchange setup status and agree that publication
succeeded before training continues. Files are written to a sibling temporary file, flushed and atomically renamed;
existing public files are never replaced. A failed write stops setup and does not publish a partial accepted record.
Publication of several files is not a filesystem transaction: if a later write fails, an earlier complete missing
sidecar can remain. Its embedded origin still describes its evidence correctly, and the accepted record is always
published last. There is no mutable latest pointer and no failed-attempt record.
