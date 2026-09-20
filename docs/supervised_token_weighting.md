# Supervised-token loss normalization

Training and validation use `loss_normalization: supervised_token_v1`. Every target accepted by
`RecurrentGPT.mask_labels` receives equal weight. Ignored prompts, pack tails and IDs outside the real vocabulary
are excluded. A supervised EOS still counts; prompt representations remain available as context and receive
indirect gradients. Physical token budgets, source scheduling and learning-rate stages retain their existing units.

For one optimizer update, the objective is the sum of supervised losses across every local microbatch and rank,
divided by the global supervised count. The streaming backward uses each pack's differentiable loss sum divided by
`C = local_microbatches * physical_pack_length`. After DDP averages gradients, the trainer multiplies them in place
by `world_size * C / global_supervised_count` before checking/clipping gradients and stepping the optimizer.
Reported loss and perplexity use the same global numerator/count. Counts are accumulated and reduced as int64.

The additive `return_loss_statistics=True` model argument returns `loss_sum` and `supervised_count` without full
logits or per-token training losses. Native and custom heads accept `reduction="sum"`; their existing five-argument
mean calls remain valid. An all-ignored sum has a zero gradient, while the public all-ignored mean remains NaN.
A globally empty update or evaluation fails clearly. Nonfinite supervised losses remain errors.

Validation sums chunked token losses at each recurrence depth and divides once by the global target count.
Per-source counts use the same mask; empty sources have no defined mean and are omitted with a warning.

Run configs, checkpoint settings and training measurement configs store the policy. Historical settings without
this field mean `legacy_pack_v0`; they are not relabeled. Resuming them is refused unless
`allow_settings_change: true` explicitly acknowledges the objective transition. The warning explains the changed
trajectory; model/optimizer moments, saved data and RNG state are preserved. Existing `run_config.json` remains the
original run configuration, while subsequent checkpoints record current settings.

The recurrent RNG depends on microbatch grouping, so full-model repartitioning is not a numerical invariant.
