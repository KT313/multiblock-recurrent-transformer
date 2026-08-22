#!/usr/bin/env python3
"""
Debug script to inspect optimizer checkpoint structure.

Usage:
    python cluster/scripts/inspect_checkpoint.py /path/to/checkpoint_base_name

Example:
    python cluster/scripts/inspect_checkpoint.py /path/to/fast_storage/recpre/outputs/212/checkpoints-DDPStrategy/step-00091551-recur1b-mig-212-stage-0_end

This will automatically check for rank-specific files (e.g., checkpoint_0.pth, checkpoint_1.pth, etc.)
"""

import torch
import sys
from pathlib import Path


def inspect_checkpoint(ckpt_path):
    """Inspect a checkpoint file and print optimizer structure."""
    ckpt_path = Path(ckpt_path)

    if not ckpt_path.exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return False

    print(f"\n{'='*80}")
    print(f"Inspecting: {ckpt_path.name}")
    print(f"{'='*80}")

    try:
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        print(f"\n📦 Checkpoint keys: {list(checkpoint.keys())}")

        # Inspect optimizer state if present
        if 'optimizer' in checkpoint:
            optim_state = checkpoint['optimizer']
            print(f"\n🔧 Optimizer state type: {type(optim_state)}")

            if isinstance(optim_state, list):
                print(f"   Optimizer state is a list with {len(optim_state)} elements")

                for idx, state in enumerate(optim_state):
                    print(f"\n   Element {idx}:")
                    if isinstance(state, dict) and 'param_groups' in state:
                        print(f"     Contains {len(state['param_groups'])} parameter groups:")
                        for pg_idx, pg in enumerate(state['param_groups']):
                            print(f"       Group {pg_idx}: {len(pg['params'])} params, lr={pg.get('lr', 'N/A')}")
                            # Show first few param IDs for debugging
                            if len(pg['params']) > 0:
                                param_ids = pg['params'][:3]
                                print(f"         First param IDs: {param_ids}{'...' if len(pg['params']) > 3 else ''}")
                    else:
                        print(f"     Type: {type(state)}, Keys: {list(state.keys()) if isinstance(state, dict) else 'N/A'}")

            elif isinstance(optim_state, dict):
                print(f"   Optimizer state is a dict")
                if 'param_groups' in optim_state:
                    print(f"   Contains {len(optim_state['param_groups'])} parameter groups:")
                    for pg_idx, pg in enumerate(optim_state['param_groups']):
                        print(f"     Group {pg_idx}: {len(pg['params'])} params, lr={pg.get('lr', 'N/A')}")
                else:
                    print(f"   Keys: {list(optim_state.keys())}")

        # Inspect other state
        if 'model' in checkpoint:
            print(f"\n🏗️  Model state present: {type(checkpoint['model'])}")

        if 'microbatch_step' in checkpoint:
            print(f"📊 Microbatch step: {checkpoint['microbatch_step']}")

        if 'optimizer_step' in checkpoint:
            print(f"📊 Optimizer step: {checkpoint['optimizer_step']}")

        print(f"\n{'='*80}\n")
        return True

    except Exception as e:
        print(f"❌ Error loading checkpoint: {type(e).__name__}: {str(e)}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python cluster/scripts/inspect_checkpoint.py <checkpoint_base_path>")
        print("\nExample:")
        print("  python cluster/scripts/inspect_checkpoint.py /path/to/fast_storage/recpre/outputs/212/checkpoints-DDPStrategy/step-00091551-recur1b-mig-212-stage-0_end")
        sys.exit(1)

    base_path = Path(sys.argv[1])

    print(f"\n🔍 Searching for checkpoint files...")
    print(f"   Base path: {base_path}")
    print(f"   Base path exists: {base_path.exists()}")

    # Check for rank-specific files (DDPStrategy format)
    rank_files = []
    for rank_idx in range(16):  # Check up to 16 ranks
        # Try different naming patterns
        patterns = [
            f"{base_path}_{rank_idx}.pth",
            f"{base_path}_{rank_idx}",
            base_path.parent / f"{base_path.name}_{rank_idx}.pth",
            base_path.parent / f"{base_path.name}_{rank_idx}",
        ]

        for pattern in patterns:
            if Path(pattern).exists():
                rank_files.append((rank_idx, Path(pattern)))
                break

    # Check for single file (SingleDeviceStrategy format)
    single_file = None
    if base_path.exists() and base_path.is_file():
        single_file = base_path
    elif Path(str(base_path) + ".pth").exists():
        single_file = Path(str(base_path) + ".pth")

    # Inspect found files
    if rank_files:
        print(f"\n✓ Found {len(rank_files)} rank-specific checkpoint files (DDPStrategy)")
        for rank_idx, ckpt_path in sorted(rank_files):
            size_mb = ckpt_path.stat().st_size / (1024 * 1024)
            print(f"   Rank {rank_idx}: {ckpt_path.name} ({size_mb:.2f} MB)")

        print(f"\n" + "="*80)
        print(f"INSPECTING EACH RANK'S CHECKPOINT")
        print(f"="*80)

        for rank_idx, ckpt_path in sorted(rank_files):
            print(f"\n{'─'*80}")
            print(f"RANK {rank_idx}")
            print(f"{'─'*80}")
            inspect_checkpoint(ckpt_path)

    elif single_file:
        print(f"\n✓ Found single checkpoint file (SingleDeviceStrategy)")
        size_mb = single_file.stat().st_size / (1024 * 1024)
        print(f"   {single_file.name} ({size_mb:.2f} MB)")
        inspect_checkpoint(single_file)

    else:
        print(f"\n❌ No checkpoint files found!")
        print(f"   Looked for:")
        print(f"     - Single file: {base_path}")
        print(f"     - Single file: {base_path}.pth")
        print(f"     - Rank files: {base_path}_[0-15].pth")

        # Show what's in the parent directory
        if base_path.parent.exists():
            print(f"\n   Files in {base_path.parent}:")
            for f in sorted(base_path.parent.iterdir())[:20]:
                print(f"     - {f.name}")


if __name__ == "__main__":
    main()
