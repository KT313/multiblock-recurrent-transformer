# Modified from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f.
# Changes (c) 2025-2026 Tobias Kerner: multi-stage curriculum orchestration, checkpointing/resume overhaul, dataset mixing, DDP adaptations. See README and git history.
"""
This script is originally adapted from and inspired by the tinyllama.py and
redpajama.py scripts in the lit-gpt/pretrain directory, but not too much of the original structure remains.
"""

####################################################################################################
# Imports.
####################################################################################################

import time

global_start_time = time.time()
import math
import os
import random
import socket

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Optional, Iterator
import json

import torch
import torch.nn as nn

# Fix PyTorch 2.6+ checkpoint loading compatibility
# For trusted training checkpoints, we need to use weights_only=False
# This will be configured via custom checkpoint_io in Fabric initialization

# Anomaly detection causes 5-10x slowdown - only enable for debugging
if os.environ.get("DEBUG_ANOMALY") == "1":
    torch.autograd.set_detect_anomaly(True)
    print("[WARNING] Anomaly detection enabled - expect 5-10x slowdown")
torch.cuda.memory.set_per_process_memory_fraction(0.95)  # Increased from 0.90 for better utilization


def assert_cuda_visible(min_devices: int = 1) -> None:
    """
    Robust device check that works with Slurm/MIG/cgroups and doesn't depend on NVML/AMDSMI.
    Set RECUR_ALLOW_CPU=1 to bypass for CPU-only smoke tests.
    """
    allow_cpu = os.environ.get("RECUR_ALLOW_CPU", "0") == "1"
    try:
        is_avail = torch.cuda.is_available()
        count = torch.cuda.device_count() if is_avail else 0
    except Exception as e:
        print(f"[WARN] torch.cuda probe failed: {e}")
        is_avail, count = False, 0

    if count < min_devices and not allow_cpu:
        host = socket.gethostname()
        raise RuntimeError(
            f"No CUDA device visible to PyTorch on {host}. "
            f"is_available={is_avail}, device_count={count}. "
            f"Set RECUR_ALLOW_CPU=1 to bypass for CPU-only tests."
        )

    if is_avail and count > 0:
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "unknown"
        print(f"[INFO] CUDA OK: {count} device(s); device 0 = {name}")
    elif allow_cpu:
        print("[INFO] RECUR_ALLOW_CPU=1 set — continuing on CPU (for smoke tests).")

# call once at startup
assert_cuda_visible()


if TYPE_CHECKING:
    import torch.distributed
    import torch.version
    import torch._dynamo.config
from lightning.fabric.strategies import FSDPStrategy, DDPStrategy, SingleDeviceStrategy, DeepSpeedStrategy
from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.pytorch.loggers import WandbLogger
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean
from torch.distributed.checkpoint import state_dict as state_dict_helpers

import warnings

warnings.filterwarnings("ignore", message="The config.capture_autograd_function flag is deprecated")  # pytorch nightly
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`.*")  # our weights

from recpre.settings import CLISettings


from recpre.tokenizer import Tokenizer
from recpre.huggingface_dataset import HuggingfaceDataset, HuggingfaceCombinedDataset, ParquetStream, ParquetStreamPure, RandomTokensDataset
from recpre.data_loading_utils import generic_collate_fn
import recpre.utils
from recpre.data_scheduler_utils import DataSchedulerTracker, DataScheduler
from recpre.stage_manager import StageManager
from recpre.monitor import (
    enable_monitoring_on_step,
    disable_monitoring_and_retrieve_metrics,
    track_gradient_metrics,
    get_MFU_metrics,
)

from dataclasses import asdict, is_dataclass, dataclass
from jsonargparse import CLI
import re

RETRY_CACHE_INDUCTOR = False

if RETRY_CACHE_INDUCTOR:
    import torch._inductor.codecache

    torch._inductor.codecache.PyCodeCache.load_by_key_path = classmethod(recpre.utils.load_by_key_path_with_retry)

end_time = time.time()
if int(os.getenv("SLURM_PROCID", "0")) == 0:
    print(f"{time.ctime()[:-5]}: Time to load libraries: {end_time - global_start_time:.02f} seconds.")


####################################################################################################
# Custom Checkpoint IO for PyTorch 2.6+ compatibility
####################################################################################################
from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO


class TrustedCheckpointIO(TorchCheckpointIO):
    """Custom checkpoint IO that uses weights_only=False for trusted training checkpoints.

    PyTorch 2.6+ changed the default to weights_only=True for security, but our training
    checkpoints contain custom classes (DataLoaders, custom datasets, etc.) that need
    weights_only=False to load properly. Since these are our own trusted checkpoints,
    this is safe.
    """

    def load_checkpoint(self, path, map_location=None):
        """Load checkpoint with weights_only=False for compatibility."""
        from lightning.fabric.utilities.cloud_io import get_filesystem

        fs = get_filesystem(path)
        with fs.open(path, "rb") as f:
            return torch.load(f, map_location=map_location, weights_only=False)


####################################################################################################
# Setup functions.
####################################################################################################
Fabric = recpre.utils.LightningFabric | recpre.utils.SimpleFabric


def set_torch_flags(cfg):
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    # Do they AMD cards pick up on any of this? :
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # Should be true anyway

    # Dynamo + DDP primitives:
    torch._dynamo.config.optimize_ddp = cfg.dynamo_ddp_config
    # compilation choices
    torch._dynamo.config.compiled_autograd = cfg.compiled_autograd
    if cfg.fail_instead_of_recompile:
        torch._dynamo.config.error_on_recompile = True


def setup_fabric(cfg: CLISettings) -> Fabric:
    """Sets up the fabric and logger based on the cfg."""
    # Instantiate the logger.
    if cfg.logger_name == "wandb":
        # set offline dynamically from environemtn
        logger = WandbLogger(
            entity=os.environ.get("WANDB_ENTITY"), project=cfg.logger_project, name=cfg.run_name, save_dir=cfg.out_dir
        )
    else:
        raise ValueError(f"`logger={cfg.logger_name}` is not a valid option.")

    # Instantiate the fabric.
    # Use custom checkpoint IO for PyTorch 2.6+ compatibility with trusted checkpoints
    checkpoint_io = TrustedCheckpointIO()

    if cfg.fabric_strategy == "simple-ddp":
        fabric = recpre.utils.SimpleFabric(
            precision=cfg.fabric_precision,
            loggers=[logger],
        )
        fabric.print("Using simple fabric.")
    else:
        if "fsdp" in cfg.fabric_strategy:
            if "grad" in cfg.fabric_strategy:
                sharding_strategy = "SHARD_GRAD_OP"
                """ Gradients and optimizer states are sharded during computation, and additionally, parameters are sharded outside computation.
                    For the parameters, this strategy unshards before the forward, does not reshard them after the forward, and only reshards them
                    after the backward computation. The sharded optimizer states are updated locally per rank. Inside no_sync(), the parameters
                    are not resharded after the backward computation.
                """
            elif "full" in cfg.fabric_strategy:
                sharding_strategy = "FULL_SHARD"  # choose FULL_SHARD if oom
            elif "hybrid2" in cfg.fabric_strategy:
                sharding_strategy = "_HYBRID_SHARD_ZERO2"
                """  Apply SHARD_GRAD_OP within a node, and replicate parameters across nodes. This is like HYBRID_SHARD, except this may
                provide even higher throughput since the unsharded parameters are not freed after the forward pass, saving the all-gathers
                in the pre-backward.
                """
            else:
                sharding_strategy = "HYBRID_SHARD"  #  USE "HYBRID_SHARD" AT SCALE
            precision_strategy = derive_precision(cfg.fabric_precision, cfg.fabric)
            strategy = FSDPStrategy(
                auto_wrap_policy={cfg.model_config.Block},
                mixed_precision=precision_strategy,
                activation_checkpointing_policy={cfg.model_config.Block} if cfg.gradient_checkpointing else None,
                state_dict_type="full",
                sharding_strategy=sharding_strategy,  # type: ignore
                param_init_fn=((lambda x: x.to_empty(recurse=False)) if cfg.model_impl == "huggingface" else None),
                # use_orig_params=cfg.fabric.fsdp_use_original_params, # no
            )
        elif cfg.fabric_strategy == "ddp":
            strategy = DDPStrategy(
                find_unused_parameters=True,  # required for recurrent models with TBPTT
                checkpoint_io=checkpoint_io,
            )
        elif cfg.fabric_strategy == "deepspeed":
            strategy = DeepSpeedStrategy(
                # find_unused_parameters=True,  # required for recurrent models with TBPTT
                zero_optimization=True,
                stage=1,
                # checkpoint_io=checkpoint_io,
            )
        elif cfg.fabric_strategy == "single":
            strategy = SingleDeviceStrategy(
                device=torch.device("cuda:0") if torch.cuda.is_available() else "cpu",
                checkpoint_io=checkpoint_io,
            )
        elif cfg.fabric_strategy == "axonn_tp":
            from axonn.lightning import AxonnStrategy
            # dataloader world size mod not merged yet

            strategy = AxonnStrategy(
                G_intra_r=cfg.fabric.row_tensor_parallel_size,
                G_intra_c=cfg.fabric.col_tensor_parallel_size,  # this needs more integration!
                G_intra_d=cfg.fabric.depth_tensor_parallel_size,
                overlap_communication=cfg.fabric.optimize_communication,
                checkpoint_io=checkpoint_io,
            )
        else:
            raise ValueError(f"`fabric_strategy={cfg.fabric_strategy}` is not a valid option.")

        # Instantiate and launch/initialize the fabric distributed environment management.
        # Use LightningEnvironment plugin to override SLURM detection and properly use torchrun env vars
        # Instantiate and launch/initialize the fabric distributed environment management.
        fabric = recpre.utils.LightningFabric(
            devices=cfg.devices,
            strategy=strategy,
            precision=cfg.fabric_precision,
            loggers=[logger],
            num_nodes=cfg.num_nodes,
        )
        fabric.print(f"Using Lightning Fabric with strategy {cfg.fabric_strategy} ")
        fabric.launch()

        # Diagnostic logging to verify distributed setup
        fabric.print(f"[DISTRIBUTED] Rank {fabric.global_rank}/{fabric.world_size} on device {fabric.device}")
        fabric.print(f"[ENV] RANK={os.getenv('RANK', 'not set')}, LOCAL_RANK={os.getenv('LOCAL_RANK', 'not set')}, WORLD_SIZE={os.getenv('WORLD_SIZE', 'not set')}")
        fabric.print(f"[CONFIG] devices={cfg.devices}, num_nodes={cfg.num_nodes}")

    fabric.print(f"> gradient_accumulation_steps = {cfg.gradient_accumulation_steps}")
    fabric.print(f"> micro_batch_size = {cfg.micro_batch_size}")
    fabric.print(f"> global_batch_size = {cfg.world_batch_size}")

    return fabric


####################################################################################################
# Main driver functions.
####################################################################################################


def startup(fabric: recpre.utils.LightningFabric, cfg: CLISettings):
    """The main driver function for the training script."""
    start_time = time.time()

    # Get job remaining time
    if cfg.save_n_min_before_job_done is not None:
        if fabric.global_rank == 0:
            global_total_time = _get_time_from_slurm()
            fabric.print(f"Total job time: {global_total_time:.02f} seconds.")
        else:
            global_total_time = 0

        global_total_time = fabric.broadcast(global_total_time, 0)  # does this have to be a broadcast?
        cfg.global_total_time = global_total_time

    # Prepare directories for logging
    if fabric.global_rank == 0:
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
        (Path(cfg.out_dir) / fabric.get_prefix_for_checkpoint()).mkdir(parents=True, exist_ok=True)
        # Last step before we move on is to dump the cfg to a file in the out_dir.
        # This is is itself loadable as a config by passing like train.py --config run_config.json
        with open(f"{cfg.out_dir}/run_config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=4)
        with open(f"{cfg.out_dir}/model_config.json", "w") as f:
            json.dump(asdict(cfg.model_config) if is_dataclass(cfg.model_config) else cfg.model_config, f, indent=4)
    # Load tokenizer
    tokenizer_kwargs = {"max_length": cfg.loader_block_size}
    # Only pass special token IDs if they are explicitly set in config (not None)
    if hasattr(cfg.model_config, 'bos_token_id') and cfg.model_config.bos_token_id is not None:
        tokenizer_kwargs["bos_token_id"] = cfg.model_config.bos_token_id
    if hasattr(cfg.model_config, 'eos_token_id') and cfg.model_config.eos_token_id is not None:
        tokenizer_kwargs["eos_token_id"] = cfg.model_config.eos_token_id
    if hasattr(cfg.model_config, 'pad_token_id') and cfg.model_config.pad_token_id is not None:
        tokenizer_kwargs["pad_token_id"] = cfg.model_config.pad_token_id
    tokenizer = Tokenizer(cfg.tokenizer_path, **tokenizer_kwargs)
    tokenizer.pad_id = -100 # automatically ignored (ignore_id)
    # if tokenizer.pad_id is None:
    #     tokenizer.pad_id = -1
    if cfg.cache_attn:
        assert tokenizer.cache_token_id is not None
    if cfg.doc_block_attn:
        assert tokenizer.eod_token_id is not None

    # Initialize multi-stage training if enabled
    stage_manager = None
    stage_dataloaders = None
    if cfg.enable_multi_stage:
        fabric.print("=" * 80)
        fabric.print("Multi-stage training enabled")
        fabric.print("=" * 80)

        # Create stage manager
        stage_manager = StageManager(cfg)

        # Print stage configuration
        fabric.print(stage_manager.get_stage_summary())

        # Override max_steps with total steps from all stages
        cfg.max_steps = stage_manager.total_steps
        fabric.print(f"Total training steps: {cfg.max_steps:,}")
        fabric.print("=" * 80)

    # Create data objects
    t0 = time.time()
    # On block size, moved this here to be more explicit that this is happening ...
    if not cfg.ignore_block_size_mismatch:
        assert cfg.block_size == cfg.model_config.block_size, "cfg.block_size must match config.block_size"

    # Create dataloaders - either stage-based (multi-stage) or unified (single-stage)
    if cfg.enable_multi_stage:
        # NEW APPROACH: Separate dataloaders per stage with constant weights
        stage_dataloaders = create_stage_dataloaders(
            batch_size=cfg.micro_batch_size,
            block_size=cfg.loader_block_size,
            fabric=fabric,
            seed=(cfg.seed + fabric.global_rank),
            cfg=cfg,
            tokenizer=tokenizer,
        )
        # No data_scheduler needed with separate dataloaders
        data_scheduler = None
    else:
        # OLD APPROACH: Single unified dataloader (for non-multi-stage training)
        train_dataloader, val_dataloader, data_scheduler_tracker = create_dataloaders(
            batch_size=cfg.micro_batch_size,
            block_size=cfg.loader_block_size,
            fabric=fabric,
            seed=(cfg.seed + fabric.global_rank),
            cfg=cfg,
            tokenizer=tokenizer,
        )
        if data_scheduler_tracker is not None:
            data_scheduler = DataScheduler(data_scheduler_tracker, cfg.data_config["train_data"], cfg)
            data_scheduler.step(0)
        else:
            data_scheduler = None

    fabric.print(f"{time.ctime()[:-5]}: Time to instantiate and setup dataloaders: {time.time() - t0:.02f} seconds.")

    # Construct the model
    fabric.seed_everything(cfg.seed)  # same seed for every process to init model (FSDP)
    # if cfg.model_checkpoint is not None:
    #     recpre.utils.check_valid_checkpoint_dir(Path(cfg.model_checkpoint))
    fabric.print(f"Loading model with {cfg.model_config.__dict__}")

    # Set the objective
    objective = {
        "op": recpre.utils.chunked_cross_entropy,
        "label_smoothing": cfg.label_smoothing,
        "ignore_index": -100,
        "z_regularization": cfg.z_regularization,
    }

    # Initialize the model
    t0 = time.time()
    with fabric.init_module(empty_init=False):
        model = cfg.model_config.construct_model(
            objective=objective,
            tokenizer=tokenizer.processor,
            gradient_checkpointing=cfg.gradient_checkpointing and "fsdp" not in cfg.fabric_strategy,
        )
    fabric.print(f"{time.ctime()[:-5]}: Time to instantiate model: {time.time() - t0:.02f} seconds.")
    num_params = recpre.utils.num_parameters(model)
    fabric.log_to_summary({"num_parameters": num_params, "device": torch.cuda.get_device_name()})

    # With fabric and the model up, we can compute a few last derived cfg
    if cfg.max_steps is None:
        cfg.max_tokens_per_device = cfg.max_tokens // fabric.world_size
        cfg.tokens_per_step = cfg.micro_batch_size * cfg.block_size
        cfg.max_steps = cfg.max_tokens_per_device // cfg.tokens_per_step
        fabric.print(
            f"Based on block size {cfg.block_size}, expecting to take {cfg.max_steps} steps to reach "
            f"{cfg.max_tokens / 1e12:4.2f}T tokens.\nRunning {cfg.tokens_per_step * fabric.world_size} tok/step "
            f"for {cfg.max_tokens_per_device / 1e9:4.2f}B in total per device."
        )

    # Set up the final fabric+model details
    t0 = time.time()

    # DeepSpeed requires optimizer to be created BEFORE fabric.setup
    if "deepspeed" in cfg.fabric_strategy:
        param_groups = recpre.optim.get_param_groups(
            model.named_parameters(),
            cfg.no_weight_decay_for_bias_and_norm_params,
            weight_lr_scale=1 / getattr(model.config, "mup_model_scaling_factor", 1.0),
        )
        optimizer = recpre.optim.get_optimizer(
            cfg.optimizer,
            model,
            cfg.fabric.optim_sharding,
            allow_fusion=not torch.version.hip and "bf16" in cfg.fabric_precision,
            use_apex_adamw=cfg.fabric.use_apex_adamw,
        )(param_groups, **cfg.optim_config)
        fabric.print(f"{time.ctime()[:-5]}: Time to instantiate optimizer for DeepSpeed: {time.time() - t0:.02f} seconds.")
    else:
        optimizer = None  # Will be created after fabric.setup for DDP/FSDP

    # Use dynamic=True to handle variable sequence lengths (128, 256, ..., 2048) without recompiling
    if cfg.compile_model:
        model = torch.compile(model, dynamic=True)

    model, optimizer = fabric.setup(model=model, optimizer=optimizer, compile=False) # type: ignore

    fabric.print(f"Model with full setup is {model}")
    fabric.print(f"Total parameters: {num_params:,}")
    if hasattr(model.transformer, "core_blocks"):
        rec_params = sum([p.numel() for block in model.transformer.core_blocks for p in block.parameters()])
        static_params = num_params - rec_params
        if "fsdp" in cfg.fabric_strategy:
            rec_params, static_params = rec_params * cfg.devices, static_params * cfg.devices
        r = model.config.mean_recurrence
        # Handle both single value and list of mean_recurrence values
        r_display = r[0] if isinstance(r, list) else r
        r_mean = sum(r) / len(r) if isinstance(r, list) else r
        unfolded_params_mean = static_params + rec_params * r_mean
        unfolded_params_max = static_params + rec_params * 2 * r_mean
        fabric.print(f"Model initialized with {int(rec_params / 1e6):,}m parameters in recurrent block.")
        fabric.print(f"Will unfold to {int(unfolded_params_mean // 1e6):,}m mean parameters at test time ({r_display} rec).")
        fabric.print(f"Could unfold to {int(unfolded_params_max // 1e6):,}m parameters at test time ({2 * r_display} rec).")
    fabric.print(f"{time.ctime()[:-5]}: Time to setup model: {time.time() - t0:.02f} seconds.")
    t0 = time.time()

    # For non-DeepSpeed strategies, create optimizer AFTER fabric.setup
    # (DeepSpeed already has optimizer created and configured before fabric.setup)
    if "deepspeed" not in cfg.fabric_strategy:
        param_groups = recpre.optim.get_param_groups(
            model.named_parameters(),
            cfg.no_weight_decay_for_bias_and_norm_params,
            weight_lr_scale=1 / getattr(model.config, "mup_model_scaling_factor", 1.0),
        )
        optimizer = recpre.optim.get_optimizer(
            cfg.optimizer,
            model,
            cfg.fabric.optim_sharding,
            allow_fusion=not torch.version.hip and "bf16" in cfg.fabric_precision,
            use_apex_adamw=cfg.fabric.use_apex_adamw,
        )(param_groups, **cfg.optim_config)
        optimizer = fabric.setup_optimizers(optimizer)
        fabric.print(f"{time.ctime()[:-5]}: Time to instantiate and setup optimizers: {time.time() - t0:.02f} seconds.")

    # Build state dict - different structure for multi-stage vs single-stage
    if cfg.enable_multi_stage:
        state = {
            "model": model,
            "optimizer": optimizer,
            "tokenizer": tokenizer,
            "stage_dataloaders": stage_dataloaders,  # NEW: Container with all stage dataloaders
            "microbatch_step": 0,  # mbs steps
            "optimizer_step": 0,  # optimizer updates taken
            "is_accumulating": False,
            "metrics": {"lr": 0.0, "grad_norm": 0.0, "current_batch_size": 0},
            "should_exit_training": False,
            "model_config": asdict(cfg.model_config) if is_dataclass(cfg.model_config) else cfg.model_config,
            "stage_manager": stage_manager,  # Multi-stage training manager
            "resume_step": -1,  # Step at which training was resumed (for resume warmup), -1 if starting fresh
            "current_stage_idx": 0,  # Track current stage for checkpointing
            "save_after_first_optimizer_step": False,  # Will be set to True if resuming from checkpoint
        }
    else:
        state = {
            "model": model,
            "optimizer": optimizer,
            "tokenizer": tokenizer,
            "data_scheduler": data_scheduler,
            "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,  # val loader actually doesnt need state
            "microbatch_step": 0,  # mbs steps
            "optimizer_step": 0,  # optimizer updates taken
            "is_accumulating": False,
            "metrics": {"lr": 0.0, "grad_norm": 0.0, "current_batch_size": 0},
            "should_exit_training": False,
            "model_config": asdict(cfg.model_config) if is_dataclass(cfg.model_config) else cfg.model_config,
            "stage_manager": None,  # No stage manager for single-stage
            "resume_step": -1,  # Step at which training was resumed (for resume warmup), -1 if starting fresh
            "save_after_first_optimizer_step": False,  # Will be set to True if resuming from checkpoint
        }

    # If resuming or reloading, determines the checkpoint to resume from and loads it into the fabric
    checkpoint_was_loaded = load_checkpoint(fabric, state, cfg.out_dir, cfg.run_name, cfg.model_checkpoint, cfg.resume, cfg.resume_checkpoint_path, cfg)

    # Track if we should save after first optimizer step (for quick testing after resume)
    state["save_after_first_optimizer_step"] = checkpoint_was_loaded

    # MULTISTAGE: Recreate stage dataloaders after checkpoint load
    # With the new separate-dataloaders approach, we simply recreate all stage dataloaders
    if checkpoint_was_loaded and cfg.enable_multi_stage:
        fabric.print("")
        fabric.print("=" * 80)
        fabric.print("🔄 MULTISTAGE CHECKPOINT RESUME - RECREATING STAGE DATALOADERS")
        fabric.print("=" * 80)
        fabric.print("  Creating fresh dataloaders for all training stages")
        fabric.print("")

        # Recreate stage dataloaders (much simpler than old approach!)
        stage_dataloaders = create_stage_dataloaders(
            batch_size=cfg.micro_batch_size,
            block_size=cfg.loader_block_size,
            fabric=fabric,
            seed=(cfg.seed + fabric.global_rank),
            cfg=cfg,
            tokenizer=tokenizer,
            stateful=True,  # Can use stateful with separate dataloaders
        )

        # Update state with fresh stage dataloaders
        state["stage_dataloaders"] = stage_dataloaders

        # Update current_stage_idx based on resumed step
        current_step = state["microbatch_step"]
        stage_info = state["stage_manager"].get_current_stage_info(current_step)
        state["current_stage_idx"] = stage_info.stage_idx

        fabric.print(f"✓ Fresh stage dataloaders created for {len(cfg.training_stages)} stages")
        fabric.print(f"  Resumed at step {current_step}")
        fabric.print(f"  Current stage: {stage_info.stage_idx} ({stage_info.stage_name})")
        fabric.print("=" * 80)
        fabric.print("")

    if asdict(cfg.model_config) != state["model_config"]:
        fabric.print("-------------Warning, model config difference between checkpoint and model config!-------------")

    # Report the full cfg set for the run.
    fabric.print(f"cmdline + derived cfg:\n{json.dumps(cfg.__dict__, default=lambda x: x.__dict__, indent=4)}")
    fabric.logger.log_hyperparams(cfg.__dict__)

    fabric.barrier()
    end_time = time.time()
    fabric.print(f"{time.ctime()[:-5]}: Total time to run main func setups: {end_time - start_time:.02f} seconds.")

    return state


@torch.no_grad()
def validate(
    fabric: Fabric, model: nn.Module, val_dataloader: DataLoader, tokenizer: Tokenizer, cfg
) -> dict[str, torch.Tensor]:
    fabric.print(f"Validating for {cfg.eval_iters} steps ...")
    model.eval()

    def loss_fn(logits, labels):
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.view(-1), ignore_index=model.objective["ignore_index"]
        )

    metrics = {}

    # Don't merge this block into main, works only for a few models
    # Handle mean_recurrence being a list (multi-block) or scalar (single block)
    mean_rec = model.config.mean_recurrence
    if isinstance(mean_rec, list):
        mean_rec_label = f"{mean_rec}"  # Use string representation for multi-block
    else:
        mean_rec_label = mean_rec

    losses = torch.zeros(cfg.eval_iters, len(cfg.partial_depth_eval) + 1, device=fabric.device)
    for idx, depth in enumerate(cfg.partial_depth_eval + [mean_rec_label]):
        for k, (input_ids, labels, _) in enumerate(val_dataloader):
            if k >= cfg.eval_iters:
                break

            print(f"doing val step k={k}, idx={idx}, depth={depth}", flush=True)

            input_ids = input_ids.to(fabric.device, non_blocking=True)
            labels = labels.to(fabric.device, non_blocking=True)

            mask, positions = get_attention_mask(input_ids, tokenizer, cfg.cache_attn, cfg.doc_block_attn)

            # Handle depth being either int or list (for multi-block models)
            if isinstance(depth, str):
                # This is the mean_rec_label string representation - use actual mean_recurrence
                rec_steps = mean_rec if isinstance(mean_rec, list) else torch.as_tensor([mean_rec, 0])
            elif isinstance(depth, list):
                # Already a list for multi-block - use as-is
                rec_steps = depth
            else:
                # Scalar depth - replicate for all blocks if multi-block model
                if isinstance(mean_rec, list):
                    # Multi-block model: replicate depth for all blocks
                    rec_steps = [(depth, 0) for _ in range(len(mean_rec))]
                else:
                    # Single block model: use as scalar tuple
                    rec_steps = torch.as_tensor([depth, 0])

            outputs = model(
                input_ids, position_ids=positions, attention_mask=mask, return_logits=True, num_steps_pair=rec_steps
            )
            losses[k, idx] = loss_fn(outputs["logits"], labels)

    # print(f"{time.ctime()[:-5]}: Validation forward passes complete on rank ({fabric.global_rank}/{fabric.world_size})")
    # Communicate
    global_val_loss = fabric.all_reduce(losses.mean(dim=0))  # dim-0 is the mbs dimension, dim-1 is kept after comms
    metrics["val_loss"] = global_val_loss[-1]
    metrics["val_ppl"] = global_val_loss[-1].exp()
    for idx, depth in enumerate(
        cfg.partial_depth_eval + [mean_rec_label]
    ):  # duplicate mean depth value key for ease of use
        metrics[f"val_loss_{depth}"] = global_val_loss[idx]
        metrics[f"val_ppl_{depth}"] = global_val_loss[idx].exp()

    model.train()
    return metrics


def train_step(input_ids, labels, fabric, state, running_loss, running_ppl, cfg):
    """Separate scope for a single train step, encapsulating the part that is actual work"""
    model = state["model"]
    optimizer = state["optimizer"]
    data_scheduler = state.get("data_scheduler")  # May be None in multi-stage mode
    tokenizer = state["tokenizer"]
    metrics = state["metrics"]

    state["microbatch_step"] += 1
    model.step = state["microbatch_step"]  # propagate this to the model
    # Goldfishing on CPU
    if cfg.goldfish.strategy is not None:
        labels, _ = recpre.utils.apply_tld(labels=labels, settings=cfg.goldfish, ignore_index=tokenizer.pad_id)

    # Realize the input and labels tensors.
    input_ids = input_ids.to(fabric.device, non_blocking=True)
    labels = labels.to(fabric.device, non_blocking=True)
    mask, positions = get_attention_mask(input_ids, tokenizer, cfg.cache_attn, cfg.doc_block_attn)


    # Prepare for step
    if state["microbatch_step"] < cfg.shape_watching_steps:
        bsz, seq_len = input_ids.shape
        fabric.print(f"bsz: {bsz} | seq_len: {seq_len}")
        fabric.print(f"input_ids.shape: {input_ids.shape} | labels.shape: {labels.shape}")
    elif state["microbatch_step"] == cfg.shape_watching_steps and cfg.shape_watching_steps > 0:
        fabric.print("Silencing shape watching ...")
    state["is_accumulating"] = state["microbatch_step"] % cfg.gradient_accumulation_steps != 0
    monitor_step = cfg.model_telemetry and state["microbatch_step"] % cfg.log_step_interval == 0
    if monitor_step and not state["is_accumulating"]:
        model.module.apply(enable_monitoring_on_step)

    # The actual compute step of  Forward, loss, and backward computation:
    def tightly_scoped_fwd_bwd(model, input_ids, positions, labels, mask):
        with fabric.no_backward_sync(model, enabled=state["is_accumulating"]):
            outputs = model(input_ids, position_ids=positions, labels=labels, attention_mask=mask)
            fabric.backward(outputs["loss"] / cfg.gradient_accumulation_steps, model=model)
            return outputs["loss"].detach(), outputs["log_ppl"].detach()

    loss, log_ppl = tightly_scoped_fwd_bwd(model, input_ids, positions, labels, mask)
    # Record metrics
    metrics["mbs_loss"] = loss
    running_loss.update(loss)
    running_ppl.update(log_ppl)

    # Guardrails
    if not cfg.allow_nonfinite_loss and not torch.isfinite(loss):
        fabric.print(f"Loss is {loss} on {socket.gethostname()}. Terminating ...")
        state["should_exit_training"] = True

    # Update data scheduler (single-stage only) OR log stage info (multi-stage)
    if cfg.enable_multi_stage:
        # Multi-stage: Log current stage info periodically
        if state["microbatch_step"] % 100 == 0:
            stage_info = state["stage_manager"].get_current_stage_info(state["microbatch_step"])
            fabric.print(f"[STAGE INFO] Step {state['microbatch_step']}: "
                       f"Stage {stage_info.stage_idx} ({stage_info.stage_name}), "
                       f"in_transition={stage_info.in_transition}, "
                       f"transition_progress={stage_info.transition_progress:.3f}")
    else:
        # Single-stage: Update data scheduler if present
        data_scheduler = state.get("data_scheduler")
        if data_scheduler is not None:
            data_scheduler.step(state["microbatch_step"])

    # Take an optimization step if not accumulating.
    if not state["is_accumulating"]:
        # LR scheduler (now a pre-increment, so that this step runs on exactly this scheduled lr)
        if state["stage_manager"] is not None:
            current_step_lr = get_lr_multistage(
                step=state["microbatch_step"],
                max_steps=cfg.max_steps,
                cfg=cfg,
                stage_manager=state["stage_manager"],
                resume_step=state.get("resume_step", -1)  # Pass resume_step for resume warmup
            )
        else:
            current_step_lr = get_lr(
                step=state["microbatch_step"],
                max_steps=cfg.max_steps,
                cfg=cfg,
                resume_step=state.get("resume_step", -1)  # Pass resume_step for resume warmup
            )
        for param_group in optimizer.param_groups:
            param_group["lr"] = torch.as_tensor(current_step_lr * param_group["base_lr"])

        metrics["grad_norm"] = fabric.clip_gradients(model, optimizer, max_norm=cfg.grad_clip, error_if_nonfinite=False)
        if torch.isfinite(metrics["grad_norm"]):
            if state["optimizer_step"] > 0:  # Skip first step if compiling or autotuning
                if cfg.compile_optimizer:
                    for param_group in optimizer.param_groups:
                        for param in param_group["params"]:
                            if param.grad is not None:
                                torch._dynamo.decorators.mark_static_address(param.grad)  # yolo
                    torch.compile(optimizer.step, mode="max-autotune-no-cudagraphs")()
                else:
                    optimizer.step()
        else:
            if cfg.skip_nonfinite_grads:
                fabric.print(f"Grad norm non-finite! Optimizer step {state['optimizer_step'] + 1} skipped.")
            else:
                fabric.print(f"Grad norm non-finite! Optimizer step {state['optimizer_step'] + 1}. Terminating ...")
                state["should_exit_training"] = True

        if monitor_step:  # Monitor triggers after update (to check it), but before grads are wiped
            track_gradient_metrics(model, optimizer, metrics)
            model.module.apply(partial(disable_monitoring_and_retrieve_metrics, metrics=metrics))
        optimizer.zero_grad(set_to_none=not (cfg.fabric.use_apex_adamw or cfg.compile_optimizer))
        state["optimizer_step"] += 1

        # Quick checkpoint save test after resume: save immediately after first optimizer step
        # This helps catch pickling issues early instead of waiting hours for the next checkpoint
        if state.get("save_after_first_optimizer_step", False):
            fabric.print("=" * 80)
            fabric.print("QUICK CHECKPOINT TEST: Saving after first optimizer step post-resume")
            fabric.print("=" * 80)
            maybe_save_checkpoint(fabric, state, cfg, is_accumulating=False, force_save=True)
            state["save_after_first_optimizer_step"] = False  # Only do this once

        # Data scheduler - MOVED OUTSIDE OPTIMIZER BLOCK (see line 710)
        # DataScheduler now updates EVERY microbatch instead of only on optimizer steps
        # This was causing delayed updates and missed transition boundaries
        # Batch size scheduler
        cfg.gradient_accumulation_steps = get_batch_size(state["microbatch_step"], cfg)

        metrics["lr"] = current_step_lr
        metrics["current_batch_size"] = cfg.gradient_accumulation_steps * cfg.micro_batch_size * cfg.replicas


def train(fabric, state, cfg):
    """The main training loop."""
    warmup_or_early_fail_allreduce(fabric)
    state["initial_step"] = state["last_logged_step"] = state["microbatch_step"]  # "initial_step" in this chunk

    # Create train iterator - different approach for multi-stage vs single-stage
    if cfg.enable_multi_stage:
        # Multi-stage: Create generator that yields batches from stage dataloaders
        fabric.print(f"")
        fabric.print(f"[MULTI-STAGE TRAINING] Using separate dataloaders per stage")
        fabric.print(f"  Number of stages: {len(cfg.training_stages)}")
        fabric.print(f"  Batch sampling: Probabilistic mixing during transitions")
        fabric.print(f"")

        def multi_stage_batch_generator():
            """Generator that yields batches from appropriate stage dataloader(s).

            Yields:
                Tuples of (input_ids, labels, metadata) where metadata contains
                dataset IDs for composition tracking.
            """
            step_count = state["microbatch_step"]
            while True:
                input_ids, labels, metadata = get_batch_from_stage_dataloaders(
                    stage_dataloaders=state["stage_dataloaders"],
                    stage_manager=state["stage_manager"],
                    current_step=step_count,
                    fabric=fabric,
                )
                step_count += 1
                yield input_ids, labels, metadata

        train_iterator = multi_stage_batch_generator()
    else:
        # Single-stage: Use traditional unified dataloader
        train_iterator = iter(state["train_dataloader"])

    # Set up global loss monitor.
    running_loss = RunningMean(window=cfg.log_step_interval, sync_on_compute=False).to(fabric.device)
    running_log_ppl = RunningMean(window=cfg.log_step_interval, sync_on_compute=False).to(fabric.device)

    # Dataset composition tracking (counts actual samples from each dataset)
    from collections import Counter
    dataset_sample_counter = Counter()  # Accumulates counts since last log

    first_validation_passed = False
    fabric.barrier()
    state["total_t0"] = time.time()  # this is the start time for this chunk of training
    fabric.print(f"{time.ctime()[:-5]}: Training preparations finished, starting to iterate train data now.")

    # Length-based batching iterator
    def length_sorted_batches(dataloader, micro_batch_size, gradient_accumulation_steps, tokenizer, device):
        """
        Yields mini-batches sorted by length within each global batch.
        Dramatically reduces padding waste while preserving gradient accumulation.

        Buffers gradient_accumulation_steps mini-batches (= 1 global batch),
        sorts all samples by sequence length, then re-batches and yields.
        """
        buffer = []
        pad_id = tokenizer.pad_id or 0

        for batch in dataloader:
            buffer.append(batch)

            # Process when we have a full global batch
            if len(buffer) == gradient_accumulation_steps:
                # Flatten all mini-batches into individual samples
                all_samples = []
                for input_ids, labels, metadata in buffer:
                    for i in range(input_ids.size(0)):
                        seq_len = (input_ids[i] != pad_id).sum().item()
                        all_samples.append((input_ids[i], labels[i],
                                          metadata[i] if metadata else None, seq_len))

                # Sort by sequence length (ascending order)
                all_samples.sort(key=lambda x: x[3])

                # Re-batch into mini-batches of micro_batch_size
                for i in range(0, len(all_samples), micro_batch_size):
                    mini_batch = all_samples[i:i+micro_batch_size]
                    batch_input_ids = torch.stack([s[0] for s in mini_batch]).to(device)
                    batch_labels = torch.stack([s[1] for s in mini_batch]).to(device)
                    batch_metadata = [s[2] for s in mini_batch]
                    yield batch_input_ids, batch_labels, batch_metadata

                buffer = []

        # Handle remaining samples (last incomplete global batch)
        if buffer:
            all_samples = []
            for input_ids, labels, metadata in buffer:
                for i in range(input_ids.size(0)):
                    seq_len = (input_ids[i] != pad_id).sum().item()
                    all_samples.append((input_ids[i], labels[i],
                                      metadata[i] if metadata else None, seq_len))
            all_samples.sort(key=lambda x: x[3])

            for i in range(0, len(all_samples), micro_batch_size):
                mini_batch = all_samples[i:i+micro_batch_size]
                batch_input_ids = torch.stack([s[0] for s in mini_batch]).to(device)
                batch_labels = torch.stack([s[1] for s in mini_batch]).to(device)
                batch_metadata = [s[2] for s in mini_batch]
                yield batch_input_ids, batch_labels, batch_metadata

    # Set up optional profiling
    enable_profiler = int(os.environ.get("ENABLE_TORCH_PROFILER", "0")) == 1
    profiler_steps_str = os.environ.get("PROFILER_STEPS", "5,10")
    try:
        profiler_start, profiler_end = map(int, profiler_steps_str.split(","))
    except:
        profiler_start, profiler_end = 5, 10

    profiler_context = None
    if enable_profiler and fabric.global_rank == 0:
        from torch.profiler import profile, ProfilerActivity, schedule
        fabric.print(f"[PROFILER] Enabled: will profile steps {profiler_start}-{profiler_end}")
        profiler_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=max(0, profiler_start-1), warmup=1, active=profiler_end-profiler_start, repeat=1),
            on_trace_ready=lambda prof: prof.export_chrome_trace(f"{cfg.out_dir}/trace_step{profiler_start}-{profiler_end}.json"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        profiler_context.__enter__()

    # Main training loop.
    step_time = 0
    step_num = state["microbatch_step"]  # Initialize from current step (0 for fresh training, or resumed step from checkpoint)

    # Apply length-based batching if enabled
    if cfg.sort_batches_by_length:
        fabric.print("Length-based batching enabled: sorting samples by length within each global batch to reduce padding waste")
        train_iterator = length_sorted_batches(
            train_iterator,
            cfg.micro_batch_size,
            cfg.gradient_accumulation_steps,
            state["tokenizer"],
            fabric.device
        )

    for input_ids, labels, batch_metadata in train_iterator:
        # DEBUG: Inspect what we're actually getting
        if cfg.enable_multi_stage and step_num % 100 == 0:
            fabric.print(f"[DEBUG METADATA] Step {step_num}:")
            fabric.print(f"  batch_metadata is None: {batch_metadata is None}")
            if batch_metadata is not None:
                fabric.print(f"  batch_metadata type: {type(batch_metadata)}")
                fabric.print(f"  batch_metadata length: {len(batch_metadata)}")
                fabric.print(f"  First 5 items: {batch_metadata[:5]}")
                metadata_count = len([m for m in batch_metadata if m is not None])
                fabric.print(f"  Non-None count: {metadata_count}/{len(batch_metadata)}")

        # Track dataset composition from actual batch samples
        if batch_metadata is not None:
            for data_id in batch_metadata:
                if data_id is not None:
                    dataset_sample_counter[data_id] += 1

        # Store sample for logging (first example from batch)
        # Only store when we're about to log (to save memory)
        if cfg.log_step_interval > 0 and (step_num + 1) % cfg.log_step_interval == 0:
            state["sample_input_ids"] = input_ids[0].detach().cpu()
            state["sample_labels"] = labels[0].detach().cpu()

        # Main train work
        t0 = time.time()  # measure average time over last log_step steps
        # print(f"doing step with step_num {step_num}", flush=True)
        train_step(input_ids, labels, fabric, state, running_loss, running_log_ppl, cfg=cfg)
        step_num += 1
        step_time += time.time() - t0
        step = state["microbatch_step"]

        # Check for stage transitions (only at optimizer steps, not every mini-batch)
        if not state["is_accumulating"] and state["stage_manager"] is not None:
            stage_info = state["stage_manager"].get_current_stage_info(step)
            # Detect if we just started a transition
            if stage_info.in_transition and stage_info.transition_progress < 0.01:  # Early in transition
                fabric.print("=" * 80)
                fabric.print(f"Starting transition from stage {stage_info.prev_stage_idx} to stage {stage_info.stage_idx}")
                fabric.print(f"Stage: {stage_info.stage_name}")
                if stage_info.prev_base_lr is not None:
                    fabric.print(f"LR will transition from {stage_info.prev_base_lr:.2e} to {stage_info.base_lr:.2e}")
                else:
                    fabric.print(f"LR will transition to {stage_info.base_lr:.2e}")
                fabric.print("=" * 80)
            # Detect if we just completed a transition
            elif not stage_info.in_transition and stage_info.stage_progress < 0.01 and stage_info.stage_idx > 0:
                fabric.print("=" * 80)
                fabric.print(f"Transition complete! Now in stage {stage_info.stage_idx}: {stage_info.stage_name}")
                fabric.print(f"Base LR: {stage_info.base_lr:.2e}")
                fabric.print("=" * 80)

        # Step profiler if enabled
        if profiler_context is not None:
            profiler_context.step()
            if step == profiler_end:
                fabric.print(f"[PROFILER] Completed profiling, trace saved to {cfg.out_dir}/trace_step{profiler_start}-{profiler_end}.json")

        # Regular validation (comes before log)
        validate_regular = not state["is_accumulating"] and step % cfg.eval_step_interval == 0
        validate_at_the_end = step >= cfg.max_steps - 1
        if validate_regular or validate_at_the_end:
            t0 = time.time()

            # Select validation dataloader based on training mode
            if cfg.enable_multi_stage:
                # Multi-stage: Use validation loader for current stage
                stage_info = state["stage_manager"].get_current_stage_info(step)
                current_stage_idx = stage_info.stage_idx
                current_stage_name = stage_info.stage_name

                val_dataloader_to_use = state["stage_dataloaders"].val_loaders[current_stage_idx]
                fabric.print(f"Validating on stage {current_stage_idx} ('{current_stage_name}') validation data")

                val_metrics = validate(fabric, state["model"], val_dataloader_to_use, state["tokenizer"], cfg=cfg)
            else:
                # Single-stage: Use unified validation dataloader
                val_dataloader_to_use = state.get("val_dataloader")
                if val_dataloader_to_use is not None:
                    val_metrics = validate(fabric, state["model"], val_dataloader_to_use, state["tokenizer"], cfg=cfg)
                else:
                    # Skip validation if no data available
                    val_metrics = {"val_loss": torch.tensor(float('nan'))}

            td = time.time() - t0
            val_metrics["val_time"] = torch.as_tensor(td)

            fabric.print(f"Step {step}: Val loss {val_metrics['val_loss'].item():.4f}, Val time: {td:.2f}s")
            state["metrics"] |= val_metrics
            if not first_validation_passed:
                # This is the first moment that all potential compilation calls have resolved
                fabric.log_to_summary({"first_validation_passed": time.time() - global_start_time})
                first_validation_passed = True
            fabric.barrier()

        # Communicate exit flags from all devices BEFORE logging conditional on flag
        if torch.distributed.is_initialized() and (state["microbatch_step"] % 32) == 0:  # Exit is comm'd every 32 steps
            state["should_exit_training"] = torch.as_tensor([state["should_exit_training"]], device=fabric.device)
            torch.distributed.all_reduce(state["should_exit_training"], torch.distributed.ReduceOp.MIN, async_op=False)

        # Log at an interval.
        if step % cfg.log_step_interval == 0 or (state["should_exit_training"] and (step % 32) == 0):
            # Calculate actual dataset composition from samples
            if dataset_sample_counter:
                total_samples = sum(dataset_sample_counter.values())
                dataset_composition = {
                    data_id: count / total_samples * 100.0
                    for data_id, count in dataset_sample_counter.items()
                }
                state["dataset_composition"] = dataset_composition
                state["dataset_sample_counts"] = dict(dataset_sample_counter)
            else:
                state["dataset_composition"] = {}
                state["dataset_sample_counts"] = {}

            log_step(fabric, state, running_loss, running_log_ppl, step_time, state.get("data_scheduler"), cfg)
            step_time = 0

            # Reset counter for next logging interval
            dataset_sample_counter.clear()

        if state["should_exit_training"] and (state["microbatch_step"] % 32) == 0:  # Exit is checked every 32 steps.
            fabric.print(f"{time.ctime()[:-5]}: Exiting training early in step {step} due to error signal received.")
            break

        # Maybe save, this needs to come after the error signal exit
        maybe_save_checkpoint(fabric, state, cfg, is_accumulating=state["is_accumulating"])

        if step >= cfg.max_steps - 1:
            fabric.print(f"{time.ctime()[:-5]}: Exiting training orderly after completion of {step + 1} steps.")
            break

    # Export to HuggingFace format if configured
    if cfg.export_to_hf:
        from recpre.hf_export import export_from_fabric_state
        export_from_fabric_state(fabric, state, cfg)


####################################################################################################
# Train loop sub-routines.
####################################################################################################


def log_step(
    fabric: Fabric,
    state: dict,
    running_loss: RunningMean,
    running_log_ppl: RunningMean,
    accumulated_step_time: float,
    data_scheduler: Optional[DataScheduler],
    cfg: CLISettings,
):
    """Log at this microbatch step and compute the throughput."""
    loss = running_loss.compute()
    log_ppl = running_log_ppl.compute()
    t1 = time.time()


    # Load metrics here:
    metrics = state["metrics"]

    avg_time_per_step = accumulated_step_time / (state["microbatch_step"] - state["last_logged_step"])
    tokens_per_step = cfg.micro_batch_size * cfg.block_size * fabric.world_size
    tokens_per_second = tokens_per_step / avg_time_per_step

    metrics |= {
        "local_loss": loss,
        "local_ppl": log_ppl.exp(),
        "microbatch_step": state["microbatch_step"],
        "optimizer_step": state["optimizer_step"],
        "steps/second": 1 / avg_time_per_step,
        "seconds/step": avg_time_per_step,
        "tokens/second": tokens_per_second,
        "remaining_time": (
            (t1 - state["total_t0"])
            / (state["microbatch_step"] - state["initial_step"])
            * (cfg.max_steps - state["microbatch_step"])
        ),
        "total_tokens": state["microbatch_step"] * tokens_per_step,
        "total_time": t1 - state["total_t0"],
    }
    if cfg.measure_utilization:
        max_memory_allocated_per_gpu = torch.cuda.max_memory_allocated(fabric.device) / 1024**3
        max_mem_reserved_per_gpu = torch.cuda.max_memory_reserved(fabric.device) / 1024**3
        torch.cuda.reset_peak_memory_stats(fabric.device)
        model_flops, tflops, mfu = get_MFU_metrics(tokens_per_second, fabric, state["model"], cfg.fabric_precision)
        metrics |= {
            "total_FLOPs": state["microbatch_step"] * tokens_per_step * model_flops,
            "FLOP/S": tflops,
            "model_flop_utilization": mfu,
            "max_mem_per_gpu": max_memory_allocated_per_gpu,
            "max_mem_reserved_per_gpu": max_mem_reserved_per_gpu,
        }

    # Update loss and grad_norm with all_reduce
    if "grad_norm" in metrics and metrics["grad_norm"] is not None:
        grad_norm = fabric.all_reduce(metrics["grad_norm"])
        metrics["global_grad_norm"] = grad_norm
    else:
        metrics["global_grad_norm"] = None
    metrics["global_loss"] = fabric.all_reduce(loss)

    # This is the only place where this guardrail can be checked without incurring another sync
    if cfg.loss_guardrail_active:
        total_tokens = state["microbatch_step"] * cfg.micro_batch_size * cfg.block_size * fabric.world_size
        if total_tokens > 10_000_000_000 and metrics["global_loss"] > 6:  # after 10b tokens we're in slow descent
            fabric.print(
                f"Loss guard activated with loss {metrics['global_loss']} in step {state['microbatch_step']}. "
                f"Terminating ..."
            )
            state["should_exit_training"] = True

    metrics["global_train_ppl"] = fabric.all_reduce(log_ppl).exp()

    if data_scheduler is not None:
        curr_data_weights = data_scheduler.get_data_weights()
        # Extract correct dataset names (with stage prefixes) from data_scheduler
        dataset_names = [entry.prefix for entry in data_scheduler.data_config]
        curr_data_weights = dict(zip(dataset_names, curr_data_weights))

        curr_sample_count = data_scheduler.get_sample_count()
        curr_sample_count = fabric.all_reduce(curr_sample_count, reduce_op="sum")

        curr_epoch_count = data_scheduler.get_epoch_count()
        curr_epoch_count = fabric.all_reduce(curr_epoch_count, reduce_op="mean")

        for i, x in enumerate(curr_data_weights.keys()):
            metrics["data_scheduler_weight/" + x] = curr_data_weights[x]
            metrics["data_scheduler_norm_weight/" + x] = curr_data_weights[x] / sum(list(curr_data_weights.values()))
            metrics["data_scheduler_sample_count/" + x] = curr_sample_count[i]
            metrics["data_scheduler_epoch_count/" + x] = curr_epoch_count[i]

            state["data_scheduler_weight/" + x] = metrics["data_scheduler_weight/" + x]
            state["data_scheduler_norm_weight/" + x] = metrics["data_scheduler_norm_weight/" + x]
            state["data_scheduler_sample_count/" + x] = metrics["data_scheduler_sample_count/" + x]
            state["data_scheduler_epoch_count/" + x] = metrics["data_scheduler_epoch_count/" + x]

    # Log actual dataset composition (ground truth from batch samples)
    # print(f"'dataset_composition' in state: {'dataset_composition' in state}", flush=True)
    # print(f"state['dataset_composition']: {state['dataset_composition']}", flush=True)

    if "dataset_composition" in state and state["dataset_composition"]:
        fabric.print("\n" + "=" * 80)
        fabric.print("DATASET COMPOSITION (Actual samples in batches)")
        fabric.print("=" * 80)

        # Sort by percentage for readability
        sorted_composition = sorted(
            state["dataset_composition"].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for data_id, percentage in sorted_composition:
            count = state["dataset_sample_counts"].get(data_id, 0)
            fabric.print(f"  {data_id:40s}: {percentage:6.2f}% ({count:6d} samples)")

        # Also show scheduled weights for comparison if available
        if data_scheduler is not None:
            fabric.print("\nScheduled weights (from DataScheduler):")
            scheduled_weights = data_scheduler.get_data_weights()
            total_weight = sum(scheduled_weights)
            if total_weight > 0:
                # Extract correct dataset names (with stage prefixes) from data_scheduler
                dataset_names = [entry.prefix for entry in data_scheduler.data_config]
                # Create mapping from dataset name to scheduled weight
                weight_by_name = dict(zip(dataset_names, scheduled_weights))
                # Show ALL datasets with scheduled weights, not just sampled ones
                total_samples = sum(state["dataset_sample_counts"].values()) if state["dataset_sample_counts"] else 1
                for data_id, scheduled_weight in sorted(weight_by_name.items(), key=lambda x: -x[1]):
                    if scheduled_weight > 0.001:  # Only show datasets with meaningful weight
                        weight_pct = (scheduled_weight / total_weight * 100.0) if total_weight > 0 else 0.0
                        actual_count = state["dataset_sample_counts"].get(data_id, 0)
                        actual_pct = (actual_count / total_samples * 100.0) if total_samples > 0 else 0.0
                        status = "✓ sampled" if actual_count > 0 else "✗ NOT SAMPLED"
                        fabric.print(f"  {data_id:40s}: {weight_pct:6.2f}% (sched) | {actual_pct:6.2f}% (actual) | {status}")

        fabric.print("=" * 80 + "\n")

    # Log multi-stage training info
    if state["stage_manager"] is not None:
        stage_info = state["stage_manager"].get_current_stage_info(state["microbatch_step"])
        metrics["stage/current_stage"] = stage_info.stage_idx
        metrics["stage/base_lr"] = stage_info.base_lr
        metrics["stage/in_transition"] = 1.0 if stage_info.in_transition else 0.0
        metrics["stage/transition_progress"] = stage_info.transition_progress
        metrics["stage/stage_progress"] = stage_info.stage_progress

        state["stage/current_stage"] = metrics["stage/current_stage"]
        state["stage/stage_name"] = stage_info.stage_name

    fabric.log_dict(metrics, step=state["microbatch_step"])
    state["last_logged_step"] = state["microbatch_step"]

    # Log some metrics to the console.
    step_timing = (
        f" steps/sec: {metrics['steps/second']:4.2f}  |"
        if metrics["steps/second"] >= 1.0
        else f" secs/step: {metrics['seconds/step']:4.2f}  |"
    )
    lr_str = f"{metrics['lr']:2.4e}" if "lr" in metrics and metrics["lr"] is not None else ""
    grad_norm_str = f"{metrics['global_grad_norm']:6.4e}" if metrics["global_grad_norm"] is not None else ""

    fabric.print(
        f"{time.ctime()[:-5]}\n"
        f"Step {metrics['microbatch_step']:>8}    | Loss: {metrics['global_loss']:7.4f} | {metrics['global_train_ppl']:9.2f} PPL     |"
        f" Update {metrics['optimizer_step']:>8}     |\n"
        f"{'(optimizer.step)' if not state['is_accumulating'] else ' ' * 16}"
        f" | LR: {lr_str:>10}| Grad norm: {grad_norm_str:>11} |{' ' * 19}|\n"
        f"                 | MFU : {metrics.get('model_flop_utilization', 0):6.2%}  | TFLOP/S : {metrics.get('FLOP/S', 0):5.2f}  |"
        f" tok/sec: {metrics['tokens/second']:8.1f} | {step_timing}\n"
        f"                 | Max mem allocated: {metrics.get('max_mem_per_gpu', 0):4.2f} GB       "
        f"| Max mem reserved: {metrics.get('max_mem_reserved_per_gpu', 0):4.2f} GB            |\n"
        f"                 | Tokens: {metrics['total_tokens'] / 1e9: 4.1f}B | exaFLOP: {metrics.get('total_FLOPs', 0) / 1e18:8.5f} |"
        f" Remaining time: {metrics['remaining_time'] / 3600 / 24:.2f} days             |"
    )

    # Log sample batch for verification (only during finetuning)
    if cfg.log_step_interval > 0:
        log_sample_batch(fabric, state, cfg)

    # Reset metrics after logging them
    state["metrics"] = {}


def log_sample_batch(fabric: Fabric, state: dict, cfg: CLISettings):
    """Log a decoded sample showing chat template application and label masking.

    This helps verify during finetuning that:
    1. Chat template is applied correctly (Llama-2 style)
    2. User prompt tokens are masked (not trained)
    3. Assistant response tokens are trained
    4. Training statistics look reasonable
    """
    if "sample_input_ids" not in state or "sample_labels" not in state:
        return

    input_ids = state["sample_input_ids"]
    labels = state["sample_labels"]
    tokenizer = state["tokenizer"]

    # Move tensors to CPU for decoding
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.cpu()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu()

    try:
        # Decode full input sequence
        input_text = tokenizer.decode(input_ids, skip_special_tokens=False)

        # Decode only trained tokens (where labels != -100)
        trained_mask = labels != -100
        if trained_mask.any():
            trained_ids = input_ids[trained_mask]
            trained_text = tokenizer.decode(trained_ids, skip_special_tokens=False)
        else:
            trained_text = "(no tokens trained - all masked)"

        # Compute training statistics
        num_trained_tokens = trained_mask.sum().item()
        num_total_tokens = len(input_ids)
        num_masked_tokens = num_total_tokens - num_trained_tokens
        pct_trained = 100 * num_trained_tokens / num_total_tokens if num_total_tokens > 0 else 0
        pct_masked = 100 * num_masked_tokens / num_total_tokens if num_total_tokens > 0 else 0

        # Print formatted sample
        fabric.print("=" * 80)
        fabric.print("SAMPLE FROM CURRENT BATCH")
        fabric.print("=" * 80)
        fabric.print(f"\nFull Input Sequence ({num_total_tokens} tokens):")
        fabric.print(f"  {input_text[:500]}{'...' if len(input_text) > 500 else ''}")
        fabric.print(f"\nTrained Tokens Only (assistant response, {num_trained_tokens} tokens):")
        fabric.print(f"  {trained_text[:500]}{'...' if len(trained_text) > 500 else ''}")
        fabric.print(f"\nTraining Statistics:")
        fabric.print(f"  Total tokens:   {num_total_tokens}")
        fabric.print(f"  Trained tokens: {num_trained_tokens} ({pct_trained:.1f}%)")
        fabric.print(f"  Masked tokens:  {num_masked_tokens} ({pct_masked:.1f}%)")
        print(end="", flush=True)

        # Show token-level breakdown for first 20 tokens (debugging)
        fabric.print(f"\nFirst 20 Tokens (for verification):")
        fabric.print(f"{'Idx':<5} {'Token':<20} {'Status':<10}")
        fabric.print("-" * 40)
        for i in range(min(20, len(input_ids))):
            token_text = tokenizer.decode(input_ids[i], skip_special_tokens=False)
            token_text = token_text.replace("\n", "\\n").replace("\t", "\\t")
            status = "TRAIN" if labels[i] != -100 else "MASK"
            fabric.print(f"{i:<5} {token_text[:18]:<20} {status:<10}")

        fabric.print("=" * 80 + "\n")

    except Exception as e:
        fabric.print(f"⚠️  Error logging sample batch: {e}\n")


####################################################################################################
# Data utility functions.
####################################################################################################


@dataclass
class StageDataloaders:
    """Container for per-stage dataloaders in multi-stage training.

    Instead of using a single unified dataloader with dynamic weight modification,
    this approach creates separate dataloaders for each training stage with constant weights.
    During phase transitions, samples are mixed from adjacent stage dataloaders.
    """
    train_loaders: list[StatefulDataLoader | DataLoader]  # One dataloader per stage
    val_loaders: list[DataLoader]                          # One validation loader per stage
    train_iterators: list[Optional[Iterator]]              # Cached iterators (lazy init)
    val_iterators: list[Optional[Iterator]]                # Cached validation iterators

    def get_train_iterator(self, stage_idx: int) -> Iterator:
        """Get or create train iterator for a given stage."""
        if self.train_iterators[stage_idx] is None:
            self.train_iterators[stage_idx] = iter(self.train_loaders[stage_idx])
        return self.train_iterators[stage_idx]

    def get_val_iterator(self, stage_idx: int) -> Iterator:
        """Get or create validation iterator for a given stage."""
        if self.val_iterators[stage_idx] is None:
            self.val_iterators[stage_idx] = iter(self.val_loaders[stage_idx])
        return self.val_iterators[stage_idx]


def create_stage_dataloaders(
    batch_size: int,
    block_size: int,
    fabric: Fabric,
    seed: int,
    *,
    cfg: CLISettings,
    tokenizer: Tokenizer,
    stateful: bool = True,
) -> StageDataloaders:
    """Create separate dataloaders for each training stage.

    This replaces the unified dataloader approach with separate dataloaders per stage.
    Each stage's dataloader has constant weights (no dynamic modification needed).

    Args:
        batch_size: Micro batch size
        block_size: Sequence block size
        fabric: Lightning Fabric instance
        seed: Random seed
        cfg: Configuration with training_stages
        tokenizer: Tokenizer instance
        stateful: Whether to use StatefulDataLoader

    Returns:
        StageDataloaders object with train_loaders and val_loaders for each stage
    """
    fabric.print(f"Creating separate dataloaders for {len(cfg.training_stages)} training stages...")
    fabric.print(f"  Seed: {seed}")
    fabric.print(f"  Stateful: {stateful}")

    train_loaders = []
    val_loaders = []

    for stage_idx, stage in enumerate(cfg.training_stages):
        fabric.print(f"\n  Stage {stage_idx} ({stage.name}):")
        fabric.print(f"    Train datasets: {len(stage.train_data)}")
        fabric.print(f"    Val datasets: {len(stage.val_data)}")

        # Create train dataloader for this stage
        # NOTE: We pass the stage's train_data directly (no unified config needed)
        # NOTE: We DON'T pass scheduler parameter (constant weights)
        # NOTE: force_return_data_id=True ensures composition tracking works
        train_loader, _ = create_dataloader(
            data_config=stage.train_data,
            batch_size=batch_size,
            block_size=block_size,
            n_chunks=cfg.n_chunks,
            fabric=fabric,
            data_dir=cfg.train_data_dir,
            seed=seed,
            cfg=cfg,
            tokenizer=tokenizer,
            stateful=stateful,
            force_return_data_id=True,
        )
        train_loaders.append(train_loader)

        # Create validation dataloader for this stage
        val_loader, _ = create_dataloader(
            data_config=stage.val_data,
            batch_size=batch_size,
            block_size=block_size,
            n_chunks=cfg.n_chunks,
            fabric=fabric,
            data_dir=cfg.val_data_dir,
            seed=seed,
            cfg=cfg,
            tokenizer=tokenizer,
            stateful=False,  # Validation is never stateful
            force_return_data_id=True,
        )
        val_loaders.append(val_loader)

    fabric.print(f"\n✓ Created {len(train_loaders)} stage dataloaders")

    # Initialize iterator lists (lazy initialization - created on first use)
    train_iterators = [None] * len(train_loaders)
    val_iterators = [None] * len(val_loaders)

    return StageDataloaders(
        train_loaders=train_loaders,
        val_loaders=val_loaders,
        train_iterators=train_iterators,
        val_iterators=val_iterators,
    )


def get_batch_from_stage_dataloaders(
    stage_dataloaders: StageDataloaders,
    stage_manager,
    current_step: int,
    fabric: Fabric,
) -> tuple[torch.Tensor, torch.Tensor, list]:
    """Sample a batch from the appropriate stage dataloader(s) based on current step.

    During pure stage periods, samples come from a single stage's dataloader.
    During transitions between stages, samples are probabilistically mixed from
    the two adjacent stages based on transition progress.

    Args:
        stage_dataloaders: Container with dataloaders for all stages
        stage_manager: StageManager instance for determining current stage
        current_step: Current training step
        fabric: Fabric instance for logging

    Returns:
        Tuple of (input_ids, labels, metadata) sampled from appropriate stage(s).
        Metadata is a list of dataset IDs (data_id strings) for composition tracking.
    """
    stage_info = stage_manager.get_current_stage_info(current_step)

    if not stage_info.in_transition:
        # Pure stage sampling - all samples from current stage
        iterator = stage_dataloaders.get_train_iterator(stage_info.stage_idx)
        try:
            batch = next(iterator)
        except StopIteration:
            # Dataset exhausted, recreate iterator (cycle)
            fabric.print(f"[STAGE DATALOADER] Stage {stage_info.stage_idx} dataloader exhausted, cycling...")
            stage_dataloaders.train_iterators[stage_info.stage_idx] = None
            iterator = stage_dataloaders.get_train_iterator(stage_info.stage_idx)
            batch = next(iterator)

        return batch

    else:
        # Transition sampling - probabilistic mixing between two stages
        # Use transition_progress as probability of sampling from current (new) stage
        prev_stage_idx = stage_info.stage_idx - 1

        # Sample from current stage with probability = transition_progress
        # This gives smooth interpolation: 0% → 100% over the transition period
        if random.random() < stage_info.transition_progress:
            # Sample from new stage
            iterator = stage_dataloaders.get_train_iterator(stage_info.stage_idx)
            try:
                batch = next(iterator)
            except StopIteration:
                fabric.print(f"[STAGE DATALOADER] Stage {stage_info.stage_idx} dataloader exhausted, cycling...")
                stage_dataloaders.train_iterators[stage_info.stage_idx] = None
                iterator = stage_dataloaders.get_train_iterator(stage_info.stage_idx)
                batch = next(iterator)
        else:
            # Sample from previous stage
            iterator = stage_dataloaders.get_train_iterator(prev_stage_idx)
            try:
                batch = next(iterator)
            except StopIteration:
                fabric.print(f"[STAGE DATALOADER] Stage {prev_stage_idx} dataloader exhausted, cycling...")
                stage_dataloaders.train_iterators[prev_stage_idx] = None
                iterator = stage_dataloaders.get_train_iterator(prev_stage_idx)
                batch = next(iterator)

        return batch


def create_dataloader(
    data_config: list[recpre.settings.DataEntry],
    batch_size: int,
    block_size: int,
    n_chunks: int,
    data_dir: str,
    fabric: Fabric,
    seed: int = 1337,
    *,
    cfg: CLISettings,
    tokenizer: Tokenizer,
    stateful: bool = True,
    force_return_data_id: bool = False,
) -> tuple[StatefulDataLoader | DataLoader, Optional[DataSchedulerTracker]]:
    global_data_dir = data_dir
    datasets = []
    for curr_config in data_config:
        # Override return_data_id if requested (needed for multi-stage composition tracking)
        effective_return_data_id = curr_config.return_data_id or force_return_data_id
        if curr_config.type == "hfds":
            assert tokenizer is not None, "tokenizer must be provided for HuggingfaceDataset"
            assert curr_config.data_dir is not None, "data_dir must be provided for HuggingfaceDataset"
            dataset = HuggingfaceDataset(
                ds_name_or_path=curr_config.data_dir,  # this is a path to a previously save_to_disk'd hfds
                seed=seed,
                num_processes=fabric.world_size,
                process_rank=fabric.global_rank,
                data_id=curr_config.prefix,  # this is provided for logging, and schedule purposes
                return_data_id=effective_return_data_id,
                data_signature=curr_config.data_signature or cfg.data_signature,  # specification of the data fmt
                repetitions=curr_config.repetitions,  # repeat the dataset a number of times
            )
        elif "pqds" in curr_config.type:
            ParquetImpl = ParquetStreamPure if curr_config.type == "pqds-pure" else ParquetStream
            dataset = ParquetImpl(
                dataset_folder_path=curr_config.data_dir if curr_config.data_dir is not None else global_data_dir,
                seed=seed,
                shuffle=cfg.shuffle_blocks,
                shuffle_filenames=cfg.shuffle_filenames,
                num_processes=fabric.world_size,
                process_rank=fabric.global_rank,
                data_id=curr_config.prefix,
                data_signature=curr_config.data_signature or cfg.data_signature,
                repetitions=None,
                return_data_id=effective_return_data_id,
                prefix=curr_config.prefix,
                stateful=stateful,
            )
        elif curr_config.type == "rngds":  # debug option
            dataset = RandomTokensDataset(seed=seed, vocab_size=tokenizer.vocab_size, block_size=block_size)
        else:
            raise ValueError(f"Unsupported dataset type: {curr_config.type}")

        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(f"No data found at {data_dir}.")

    if len(datasets) > 1:
        # Multiple datasets - combine with weights
        weights = [curr_config.weight for curr_config in data_config]
        # Create tracker first (needed by dataset for dynamic weight sampling)
        data_scheduler_tracker = DataSchedulerTracker(weights=weights)
        combined_dataset = HuggingfaceCombinedDataset(
            datasets=datasets,
            seed=seed,
            weights=weights,
            data_telemetry=cfg.data_telemetry,
            tracker=data_scheduler_tracker,  # Pass tracker for dynamic weights
        )
    else:
        combined_dataset = datasets[0]
        data_scheduler_tracker = None

    parametrized_collate_fn = partial(
        generic_collate_fn,
        tokenizer=tokenizer,
        block_size=cfg.loader_block_size,
        pad_to_block_size=cfg.pad_to_block_size,
        sequence_padding_multiple=cfg.sequence_padding_multiple,
        add_bos=cfg.add_bos,
        add_eos=cfg.add_eos,
        collate_checks_enabled=cfg.collate_checks_enabled,
        all_block_size_tensors=cfg.all_block_size_tensors,
    )

    loader_class = StatefulDataLoader if stateful else DataLoader
    return (
        loader_class(
            combined_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=parametrized_collate_fn,
            num_workers=cfg.dataloader_num_workers,
            prefetch_factor=4 if cfg.dataloader_num_workers > 0 else None,
        ),
        data_scheduler_tracker,
    )


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric: Fabric,
    seed: int = 1337,
    *,
    cfg: CLISettings,
    tokenizer: Tokenizer,
    stateful: bool = True,
) -> Tuple[StatefulDataLoader | DataLoader, Optional[DataLoader], DataSchedulerTracker]:
    fabric.print(f"Creating dataloaders with seed: {seed}")
    train_dataloader, data_scheduler_tracker = create_dataloader(
        cfg.data_config["train_data"],
        batch_size=batch_size,
        block_size=block_size,
        n_chunks=cfg.n_chunks,
        fabric=fabric,
        data_dir=cfg.train_data_dir,
        seed=seed,
        cfg=cfg,
        tokenizer=tokenizer,
        stateful=stateful,
    )
    val_dataloader, _ = (
        create_dataloader(
            cfg.data_config["val_data"],
            batch_size=batch_size,
            block_size=block_size,
            n_chunks=cfg.n_chunks,
            fabric=fabric,
            data_dir=cfg.val_data_dir,
            seed=seed,
            cfg=cfg,
            tokenizer=tokenizer,
            stateful=False,
        )
        if "val_data" in cfg.data_config
        else (None, None)
    )
    return train_dataloader, val_dataloader, data_scheduler_tracker  # type: ignore


####################################################################################################
# Train utility functions.
####################################################################################################


def derive_precision(precision, strategy_details):
    """ "Precision setup for torch fsdp"""
    import torch.distributed.fsdp

    param_dtype = torch.bfloat16 if "bf16" in precision else torch.float16 if "16" in precision else torch.float32
    reduce_dtype = torch.float32 if "mixed" in precision else param_dtype
    if r := strategy_details.all_reduce_dtype is not None:
        reduce_dtype = (
            torch.float16
            if r in ["16", "fp16", "fp16-mixed"]
            else torch.bfloat16
            if r in ["bf16", "bf16-mixed"]
            else torch.float32
        )
    return torch.distributed.fsdp.MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=torch.float32,
        keep_low_precision_grads=False,
        # cast_forward_inputs=False,
    )


def get_attention_mask(input_ids, tokenizer, cache_attn=True, doc_block_attn=True):
    mask, position_ids = None, None
    return mask, position_ids


# learning rate decay schedulers
def get_lr(step: int, max_steps: int, cfg: CLISettings, resume_step: int = -1) -> float:
    base_lr = cfg.optim_config["lr"]

    # Resume handling: freeze period followed by warmup
    # Useful when optimizer state is not fully restored (e.g., ranks 1-3 after migration)
    if resume_step >= 0 and (cfg.resume_freeze_steps > 0 or cfg.resume_warmup_steps > 0):
        steps_since_resume = step - resume_step

        # Phase 1: Freeze period (LR = 0) - optimizer builds up state without updating weights
        if cfg.resume_freeze_steps > 0 and steps_since_resume < cfg.resume_freeze_steps:
            return 0.0

        # Phase 2: Warmup period - gradually ramp up from min_lr to target_lr
        if cfg.resume_warmup_steps > 0:
            warmup_start_step = steps_since_resume - cfg.resume_freeze_steps
            if warmup_start_step < cfg.resume_warmup_steps:
                # Compute target LR without resume adjustments
                target_lr = get_lr(step, max_steps, cfg, resume_step=-1)
                # Ramp from min_lr to target_lr over resume_warmup_steps
                warmup_factor = warmup_start_step / cfg.resume_warmup_steps
                return cfg.min_lr + warmup_factor * (target_lr - cfg.min_lr)

    # 1) linear warmup and cooldown
    if step < cfg.warmup_steps:
        return base_lr * step / cfg.warmup_steps
    if step > (max_steps - cfg.cooldown_steps):
        return max(base_lr * (max_steps - step) / cfg.cooldown_steps, cfg.min_lr)
    # 2) if step > max_steps, return min learning rate
    if step > max_steps:
        return cfg.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (step - cfg.warmup_steps) / (max_steps - cfg.warmup_steps)
    assert 0 <= decay_ratio <= 1
    if cfg.lr_schedule == "linear":
        return base_lr - decay_ratio * (base_lr - cfg.min_lr)
    elif cfg.lr_schedule in ["constant", "trapezoid"]:
        return base_lr
    elif cfg.lr_schedule == "cosine":
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return cfg.min_lr + coeff * (base_lr - cfg.min_lr)
    else:
        raise ValueError(f"Unsupported lr_schedule: {cfg.lr_schedule}")


def get_lr_multistage(step: int, max_steps: int, cfg: CLISettings, stage_manager, resume_step: int = -1) -> float:
    """
    Multi-stage aware learning rate scheduler.

    Uses global warmup (at beginning) and cooldown (at end).
    During stage transitions, interpolates between stage LRs.
    Within stages, applies cfg.lr_schedule (constant/cosine/linear).

    Args:
        step: Current training step
        max_steps: Total training steps
        cfg: Configuration object
        stage_manager: StageManager instance
        resume_step: Step at which training was resumed (for resume warmup), -1 if not applicable

    Returns:
        Learning rate for the current step
    """
    # Resume handling: freeze period followed by warmup
    # Useful when optimizer state is not fully restored (e.g., ranks 1-3 after migration)
    if resume_step >= 0 and (cfg.resume_freeze_steps > 0 or cfg.resume_warmup_steps > 0):
        steps_since_resume = step - resume_step

        # Phase 1: Freeze period (LR = 0) - optimizer builds up state without updating weights
        if cfg.resume_freeze_steps > 0 and steps_since_resume < cfg.resume_freeze_steps:
            return 0.0

        # Phase 2: Warmup period - gradually ramp up from min_lr to target_lr
        if cfg.resume_warmup_steps > 0:
            warmup_start_step = steps_since_resume - cfg.resume_freeze_steps
            if warmup_start_step < cfg.resume_warmup_steps:
                # Compute target LR without resume adjustments
                target_lr = get_lr_multistage(step, max_steps, cfg, stage_manager, resume_step=-1)
                # Ramp from min_lr to target_lr over resume_warmup_steps
                warmup_factor = warmup_start_step / cfg.resume_warmup_steps
                return cfg.min_lr + warmup_factor * (target_lr - cfg.min_lr)

    # Get current stage info
    stage_info = stage_manager.get_current_stage_info(step)

    # Global warmup (beginning of first stage)
    if step < cfg.warmup_steps:
        return stage_info.base_lr * step / cfg.warmup_steps

    # Global cooldown (end of last stage)
    if step > (max_steps - cfg.cooldown_steps):
        final_lr = cfg.training_stages[-1].base_lr
        return max(final_lr * (max_steps - step) / cfg.cooldown_steps, cfg.min_lr)

    # If past max_steps, return min LR
    if step > max_steps:
        return cfg.min_lr

    # During transition: interpolate between stage LRs
    if stage_info.in_transition and stage_info.prev_base_lr is not None:
        interpolated_lr = (
            stage_info.prev_base_lr +
            (stage_info.base_lr - stage_info.prev_base_lr) * stage_info.transition_progress
        )
        return max(interpolated_lr, cfg.min_lr)

    # Within stage: apply base schedule
    current_lr = stage_info.base_lr

    if cfg.lr_schedule == "constant" or cfg.lr_schedule == "trapezoid":
        return max(current_lr, cfg.min_lr)
    elif cfg.lr_schedule == "cosine":
        # Cosine decay within the stage
        coeff = 0.5 * (1.0 + math.cos(math.pi * stage_info.stage_progress))
        return max(cfg.min_lr + coeff * (current_lr - cfg.min_lr), cfg.min_lr)
    elif cfg.lr_schedule == "linear":
        # Linear decay within the stage
        return max(current_lr - stage_info.stage_progress * (current_lr - cfg.min_lr), cfg.min_lr)
    else:
        raise ValueError(f"Unsupported lr_schedule: {cfg.lr_schedule}")


# only linear batch size schedules for now
def get_batch_size(step: int, cfg: CLISettings) -> int:
    if step > cfg.batch_size_ramp:
        gradient_accumulation_steps = cfg.batch_size // cfg.micro_batch_size
    else:
        slope = step / cfg.batch_size_ramp
        gradient_accumulation_steps = math.ceil(slope * cfg.batch_size / cfg.micro_batch_size)
    return gradient_accumulation_steps


def load_checkpoint(fabric, state, out_dir, run_name, model_checkpoint, resume=True, resume_checkpoint_path=None, cfg=None):
    resume_ckpt = None
    t0 = time.time()
    if resume:
        fabric.print("================================================================================")
        fabric.print("MODEL LOAD TRIGGERED - CHECKPOINT RESUME MODE")
        fabric.print("================================================================================")

        # PRIORITY 1: Manual checkpoint path (if specified)
        if resume_checkpoint_path is not None:
            fabric.print(f"Using manually specified checkpoint path:")
            fabric.print(f"  Path: {resume_checkpoint_path}")
            resume_ckpt = Path(resume_checkpoint_path)

            # Detailed validation
            fabric.print(f"Validating checkpoint path...")
            if not resume_ckpt.exists():
                error_msg = f"ERROR: Manual checkpoint path does not exist!"
                fabric.print("=" * 80)
                fabric.print(error_msg)
                fabric.print(f"  Searched at: {resume_ckpt.absolute()}")
                fabric.print(f"  Parent dir: {resume_ckpt.parent}")
                fabric.print(f"  Parent dir exists: {resume_ckpt.parent.exists()}")
                if resume_ckpt.parent.exists():
                    siblings = list(resume_ckpt.parent.iterdir())
                    fabric.print(f"  Files in parent directory ({len(siblings)} total):")
                    for f in sorted(siblings)[:20]:  # Show first 20
                        size_mb = f.stat().st_size / (1024 * 1024) if f.is_file() else 0
                        file_type = "DIR" if f.is_dir() else "FILE"
                        fabric.print(f"    [{file_type}] {f.name} ({size_mb:.2f} MB)")
                    if len(siblings) > 20:
                        fabric.print(f"    ... and {len(siblings) - 20} more files")
                fabric.print("=" * 80)
                raise FileNotFoundError(error_msg)

            fabric.print(f"✓ Checkpoint path exists")

            if resume_ckpt.is_dir():
                # Directory checkpoint (multi-file format)
                dir_files = list(resume_ckpt.iterdir())
                fabric.print(f"  Checkpoint is a directory with {len(dir_files)} files:")
                for f in sorted(dir_files)[:10]:
                    size_mb = f.stat().st_size / (1024 * 1024) if f.is_file() else 0
                    fabric.print(f"    - {f.name} ({size_mb:.2f} MB)")
                if len(dir_files) > 10:
                    fabric.print(f"    ... and {len(dir_files) - 10} more files")
            else:
                # Single file checkpoint
                size_mb = resume_ckpt.stat().st_size / (1024 * 1024)
                fabric.print(f"  Checkpoint is a single file: {size_mb:.2f} MB")

        # PRIORITY 2: Automatic search (existing behavior)
        else:
            base_for_glob = Path(out_dir) / fabric.get_prefix_for_checkpoint()
            fabric.print(f"Searching for checkpoints automatically:")
            fabric.print(f"  Base directory: {base_for_glob}")
            fabric.print(f"  Base directory exists: {base_for_glob.exists()}")

            # Updated pattern to match actual checkpoint naming (no .pth extension, no underscore requirement)
            ckpt_pattern = f"*/*-{run_name}_*" if fabric.strategy_name == "axonn_tp" else f"*-{run_name}*"
            fabric.print(f"  Glob pattern: {ckpt_pattern}")

            ckpt_paths = list(base_for_glob.glob(ckpt_pattern))

            if len(ckpt_paths) == 0:
                fabric.print(f"✗ No checkpoint found matching pattern!")
                fabric.print(f"  Tried pattern: {ckpt_pattern}")
                fabric.print(f"  In directory: {base_for_glob}")
                if base_for_glob.exists():
                    all_files = list(base_for_glob.iterdir())
                    fabric.print(f"  Files in directory ({len(all_files)} total):")
                    for f in sorted(all_files)[:20]:
                        fabric.print(f"    - {f.name}")
                    if len(all_files) > 20:
                        fabric.print(f"    ... and {len(all_files) - 20} more files")
            else:
                fabric.print(f"✓ Found {len(ckpt_paths)} checkpoint(s):")
                for p in sorted(ckpt_paths):
                    step_num = int(p.name.split("-")[1])
                    size_mb = p.stat().st_size / (1024 * 1024) if p.is_file() else 0
                    fabric.print(f"    [step {step_num:08d}] {p.name} ({size_mb:.2f} MB)")

                # Extract step number from checkpoint filename (format: step-NNNNNNNN-run_name[-suffix])
                resume_ckpt = max(ckpt_paths, key=(lambda p: int(p.name.split("-")[1])))
                fabric.print(f"  Selected checkpoint (highest step): {resume_ckpt.name}")

                filename, directory = str(resume_ckpt.name), resume_ckpt.parents[0]
                filename = filename[filename.find("step") :]
                # For multi-rank strategies, strip off rank suffix; for single-device, use as-is
                if f"-{run_name}_" in filename:
                    filename = filename.split(f"-{run_name}_")[0] + f"-{run_name}"  # split off rank info and .pth
                if fabric.strategy_name == "axonn_tp":
                    directory = Path(out_dir) / fabric.get_prefix_for_checkpoint()
                resume_ckpt = directory / filename

        # LOAD THE CHECKPOINT
        if resume_ckpt is not None:
            fabric.print(f"")
            fabric.print(f"Attempting to load checkpoint from: {resume_ckpt}")
            fabric.print(f"  Full path: {resume_ckpt.absolute()}")

            try:
                # For multistage training, don't restore dataloader state (it may be outdated/incomplete)
                # The checkpoint might have been saved before all stages were configured
                if cfg is not None and cfg.enable_multi_stage:
                    fabric.print(f"")
                    fabric.print(f"⚠ Multistage training enabled - SKIPPING dataloader state restoration")
                    fabric.print(f"  Reason: Checkpoint may contain incomplete stage configuration")
                    fabric.print(f"  Action: Using fresh unified dataloader with all stages")
                    fabric.print(f"")

                    # Load everything EXCEPT dataloaders
                    state_to_load = {
                        "model": state["model"],
                        "optimizer": state["optimizer"],
                        "microbatch_step": state["microbatch_step"],
                        "optimizer_step": state["optimizer_step"],
                    }
                    fabric.load(resume_ckpt, state_to_load, strict=False)

                    # Update the state dict with loaded values
                    state["microbatch_step"] = state_to_load["microbatch_step"]
                    state["optimizer_step"] = state_to_load["optimizer_step"]
                else:
                    # Normal resume - load full state including dataloaders
                    fabric.load(resume_ckpt, state, strict=False)

                # Set resume_step for resume warmup functionality
                state["resume_step"] = state["microbatch_step"]
                if cfg.resume_warmup_steps > 0:
                    fabric.print(f"")
                    fabric.print(f"📊 Resume warmup enabled:")
                    fabric.print(f"  Resume step: {state['resume_step']}")
                    fabric.print(f"  Warmup steps: {cfg.resume_warmup_steps}")
                    fabric.print(f"  Will warmup LR until step: {state['resume_step'] + cfg.resume_warmup_steps}")
                    fabric.print(f"")

                fabric.print(f"✓ Checkpoint loaded successfully!")
            except Exception as e:
                error_msg = f"FAILED TO LOAD CHECKPOINT!"
                fabric.print("=" * 80)
                fabric.print(error_msg)
                fabric.print(f"  Exception type: {type(e).__name__}")
                fabric.print(f"  Exception message: {str(e)}")
                fabric.print(f"")
                fabric.print(f"  Checkpoint path: {resume_ckpt.absolute()}")
                fabric.print(f"  Checkpoint exists: {resume_ckpt.exists()}")
                if resume_ckpt.exists():
                    if resume_ckpt.is_dir():
                        dir_files = list(resume_ckpt.iterdir())
                        fabric.print(f"  Checkpoint is directory with {len(dir_files)} files:")
                        for f in sorted(dir_files):
                            size_mb = f.stat().st_size / (1024 * 1024) if f.is_file() else 0
                            fabric.print(f"    - {f.name} ({size_mb:.2f} MB)")
                    else:
                        size_mb = resume_ckpt.stat().st_size / (1024 * 1024)
                        fabric.print(f"  Checkpoint is single file: {size_mb:.2f} MB")

                # Print state keys to help debug
                fabric.print(f"")
                fabric.print(f"  Expected state keys: {list(state.keys())}")
                fabric.print("=" * 80)
                raise RuntimeError(error_msg) from e

    if resume_ckpt is None and model_checkpoint is not None:
        fabric.print("-------------------- Pretrained Checkpoint Load triggered ------------------------------")
        fabric.print(f"Loading model and optimizer from pretrained checkpoint: {model_checkpoint}")

        # For finetuning, only load model and optimizer - NOT dataloader state
        # Create a state dict with only the components we want to restore
        finetuning_state = {
            "model": state["model"],
            "optimizer": state["optimizer"],
        }
        fabric.load(model_checkpoint, finetuning_state, strict=False)

        # Reset the step counter for finetuning
        fabric.print("Note: Starting finetuning from step 0 (dataloader state not restored)")
        state["microbatch_step"] = 0
    if resume_ckpt or model_checkpoint:
        # Dataloaders should have resumed automatically, due to reloading the entire previous state. Let's print the state
        # as well as dataloader internal state:
        fabric.print(f"Loaded state is from step {state['microbatch_step']}")
        # fabric.print(f"Train Data Loader State: {state['train_dataloader'].state_dict()}")# deadlock
        fabric.print(f"{time.ctime()[:-5]} : Time to load ckpt state: {time.time() - t0:.02f} seconds.")
        fabric.print("-------------------- Checkpoint loaded    ------------------------------")
    else:
        fabric.print("-------------------- No Checkpoint loaded ------------------------------")

    # Return whether a checkpoint was loaded (for multistage dataloader recreation)
    checkpoint_was_loaded = resume_ckpt is not None
    return checkpoint_was_loaded


def maybe_save_checkpoint(fabric, state, cfg, is_accumulating=False, force_save=False):
    # Pathing for various save conditions.
    t0 = time.time()
    prefix = fabric.get_prefix_for_checkpoint()

    # Check if we should save before stage transition
    save_before_stage_transition = False
    stage_checkpoint_suffix = ""
    if state["stage_manager"] is not None:
        should_save, checkpoint_name = state["stage_manager"].should_save_stage_checkpoint(state["microbatch_step"])
        if should_save:
            save_before_stage_transition = True
            stage_checkpoint_suffix = f"-{checkpoint_name}"

    fully_qualified_checkpoint_path = f"{cfg.out_dir}/{prefix}/step-{state['microbatch_step']:08d}-{cfg.run_name}{stage_checkpoint_suffix}"

    # Check the save conditions:
    save_at_interval = not is_accumulating and state["microbatch_step"] % cfg.save_step_interval == 0
    if cfg.save_n_min_before_job_done is not None and (state["microbatch_step"] % 32) == 0:
        time_spent = time.time() - global_start_time
        remaining_time = cfg.global_total_time - time_spent
        remaining_time = remaining_time / 60.0
        remaining_time = fabric.all_reduce(remaining_time, reduce_op="mean")  # slowdown?
        save_before_timeout = remaining_time <= cfg.save_n_min_before_job_done
        if save_before_timeout:
            fabric.print(f"{time.ctime()[:-5]}: Saving at {remaining_time:.02f} minutes left")
            cfg.save_n_min_before_job_done = None  # reset
    else:
        save_before_timeout = False

    save_at_first_step = cfg.save_first_step and (state["microbatch_step"] == 0)
    save_at_last_step = cfg.save_last_step and (state["microbatch_step"] >= (cfg.max_steps - 1))

    if save_at_interval or save_at_last_step or save_at_first_step or save_before_timeout or save_before_stage_transition or force_save:
        fabric.print(f"--------------------- {time.ctime()[:-5]} Model Save triggered --------")
        fabric.print(f"Saving to {str(fully_qualified_checkpoint_path)!r}")

        # Filter state to exclude unpicklable objects (dataloaders, iterators)
        filtered_state = form_save_state(state)
        fabric.save(fully_qualified_checkpoint_path, filtered_state)
        fabric.print(f"------------------- {time.ctime()[:-5]} Checkpoint saved ({time.time() - t0:.02f} seconds)")


def form_save_state(state):
    """
    Create a filtered state dict for checkpointing.

    Excludes unpicklable objects like dataloaders, iterators, and Fabric objects.
    Only saves what's needed for resuming training: model, optimizer, counters, RNG states.
    """
    save_state_dict = {}

    # Model and optimizer are handled specially by Fabric
    save_state_dict["model"] = state["model"]
    save_state_dict["optimizer"] = state["optimizer"]

    # List of keys to exclude (unpicklable or recreated on resume)
    exclude_keys = {
        "model", "optimizer",  # Already handled above
        "train_dataloader", "val_dataloader",  # Unpicklable iterators
        "stage_dataloaders",  # Contains unpicklable iterators
        "data_scheduler",  # Contains references to dataloaders
        "fabric",  # Fabric object
        "tokenizer",  # Recreated on resume
        "sample_input_ids", "sample_labels",  # Just for logging
        "dataset_composition",  # Accumulated stats, not critical
        "save_after_first_optimizer_step",  # Transient flag, not needed in checkpoint
    }

    # Save everything else (counters, RNG states, manager objects)
    for key, value in state.items():
        if key not in exclude_keys:
            save_state_dict[key] = value

    return save_state_dict


def load_save_state(state, resume_ckpt):
    checkpoint_state = torch.load(resume_ckpt, map_location=torch.device("cpu"))
    state_dict_helpers.set_state_dict(
        state["model"],
        state["optimizer"],
        model_state_dict=checkpoint_state["model"],
        optim_state_dict=checkpoint_state["optimizer"],
        options=None,
    )
    state["train_dataloader"].load_state_dict(checkpoint_state["train_dataloader"])
    for key, value in checkpoint_state.items():
        if key not in ["optimizer", "model"] and "dataloader" not in key:
            state[key] = value


def warmup_or_early_fail_allreduce(fabric):
    if torch.distributed.is_initialized():
        fabric.print("Staging allreduce warmup")
        device = fabric.device
        # Creating random data for warmup
        flat_params = torch.randn(128 * 1024 * 1024 // 4, device=device)
        num_stages = 8
        chunk_size = flat_params.numel() // num_stages

        for i in range(num_stages):
            end = min((i + 1) * chunk_size, flat_params.numel())
            chunk = flat_params[:end]
            torch.distributed.all_reduce(chunk)
            torch.cuda.current_stream().synchronize()  # Force completion
            fabric.print(f"Warmup stage {i} [{chunk.numel() // (1024 * 1024 // 4)} MB] really completed")

        torch.distributed.barrier()
        fabric.print(f"{time.ctime()[:-5]}: All warmup stages passed")


def _get_time_from_slurm() -> int:
    try:
        global_total_str_parse = os.popen("squeue -h -j $SLURM_JOBID -o %L").read()  # this is slow
        global_total_str_parse = global_total_str_parse.strip("\n")
        global_total_str_parse = [int(i) for i in re.split(":|-", global_total_str_parse)]
        if len(global_total_str_parse) == 4:
            global_total_time = (
                24 * 3600 * global_total_str_parse[0]
                + 3600 * global_total_str_parse[1]
                + 60 * global_total_str_parse[2]
                + global_total_str_parse[3]
            )
        elif len(global_total_str_parse) == 3:
            global_total_time = (
                3600 * global_total_str_parse[0] + 60 * global_total_str_parse[1] + global_total_str_parse[2]
            )
        elif len(global_total_str_parse) == 2:
            global_total_time = 60 * global_total_str_parse[0] + global_total_str_parse[1]
    except Exception as e:
        print(e)
        global_total_time = 9999999999999999
    return global_total_time


####################################################################################################
# Main control loop
####################################################################################################
import sys
import datetime


def main():
    """Encapsulates main scope away from import calls."""

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "17777"
    device_count_str = str(torch.cuda.device_count())
    os.environ["WORLD_SIZE"] = device_count_str

    # Configuration loader
    cfg: CLISettings = CLI(CLISettings)  # type: ignore

    # Print system setup
    if int(os.getenv("SLURM_PROCID", "0")) == 0:
        print("--------------------------------------------------------------------")
        print(f"------------------ Launching run {cfg.run_name}------------------")
        print("--------------------------------------------------------------------")
        print("--------------------------------------------------------------------")
        print(f"Platform: {sys.platform}, Python: {sys.version.split(' (')[0]}, PyTorch: {torch.__version__}")
        print(f"CPU threads: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.")
        driver = f"HIP/ROCM {torch.version.hip}" if torch.version.hip else f"CUDA: {torch.version.cuda}"
        print(f"GPU : {torch.cuda.get_device_name()}. {driver}.")

    set_torch_flags(cfg)  # should come before fabric setup
    # Next we set up the fabric and logger.
    fabric = setup_fabric(cfg)

    # Now we call the main function with the fabric and cfg.
    state = startup(fabric, cfg)

    # Now we call the train function with the fabric, state, and dataloaders.
    train_time = time.time()
    train(fabric, state, cfg)

    # Now exit
    fabric.print("--------------------------------------------------------------------")
    fabric.print(f"Training time: {str(datetime.timedelta(seconds=time.time() - train_time))} ")
    fabric.log_to_summary(
        {"train_time": time.time() - global_start_time, "total_time": time.time() - global_start_time}
    )
    if fabric.device.type == "cuda":
        max_alloc = f"{torch.cuda.max_memory_allocated(fabric.device) / float(1024**3):,.3f} GB"
        max_reserved = f"{torch.cuda.max_memory_reserved(fabric.device) / float(1024**3):,.3f} GB"
        fabric.print(f"Max. Mem allocated: {max_alloc}. Max. Mem reserved: {max_reserved}.")
    fabric.print("--------------------------------------------------------------------")
    if torch.distributed.is_initialized():
        # torch.distributed.barrier()  # this could be very good or very bad
        torch.distributed.destroy_process_group()  # Force a clean exit
    if int(os.getenv("SLURM_PROCID", "0")) == 0:
        print(f"Run {cfg.run_name} finished without error.")
        print(f"---------Total time: {str(datetime.timedelta(seconds=time.time() - global_start_time))} ---------")
        print("-----------------Shutdown complete.--------------------------")


def guarded_main():
    try:
        main()
    except BaseException:  # gate around hell to guarantee NCCL deconstruction
        if torch.distributed.is_initialized():
            # torch.distributed.barrier()  # this could be very good or very bad
            torch.distributed.destroy_process_group()  # Force a clean exit
        if int(os.getenv("SLURM_PROCID", "0")) == 0:
            print("Run finished with errors.")
            print(f"---------Total time: {str(datetime.timedelta(seconds=time.time() - global_start_time))} ---------")
            print("-----------------Shutdown complete.--------------------------")

            raise


if __name__ == "__main__":
    guarded_main()
