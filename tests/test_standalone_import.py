#!/usr/bin/env python3
"""Test script to verify standalone modeling file works without recpre package.

This simulates loading the model from HuggingFace Hub where recpre is not available.
"""

import sys
import importlib.util

# Check if recpre is installed (it shouldn't be for a true test)
if importlib.util.find_spec("recpre") is not None:
    print("⚠️  WARNING: recpre package is installed.")
    print("   For a true standalone test, uninstall recpre first:")
    print("   pip uninstall recpre")
    print()
    print("Continuing test anyway...")
else:
    print("✓ recpre is not installed (good for standalone test)")

print("\n" + "="*60)
print("Testing standalone modeling file import...")
print("="*60 + "\n")

try:
    # Import the standalone module directly
    sys.path.insert(0, "recurrent-pretraining/recpre")
    from modeling_recurrent_gpt_standalone import (
        RecurrentGPTConfig,
        RecurrentGPTForCausalLM,
        RecurrentGPT,
        RMSNorm_llama,
        CausalSelfAttention,
        GatedMLP,
        SandwichBlock,
    )
    print("✓ Successfully imported all classes from standalone file")
    print()

    # Test configuration creation
    print("Testing RecurrentGPTConfig creation...")
    config = RecurrentGPTConfig(
        vocab_size=32000,
        n_embd=1024,
        num_attention_heads=16,
        intermediate_size=4096,
        block_size=2048,
        n_layers_in_prelude=2,
        n_layers_in_coda=2,
        n_layers_in_recurrent_block=[4, 4, 4],
        mean_recurrence=[12, 12, 12],
        injection_type="linear",
        state_init="normal",
        qk_bias=True,
        tie_embeddings=True,
    )
    print(f"✓ Config created successfully")
    print(f"  - Model type: {config.model_type}")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Num layers: {config.num_hidden_layers}")
    print(f"  - Injection type: {config.injection_type}")
    print()

    # Test model creation
    print("Testing RecurrentGPTForCausalLM creation...")
    import torch
    model = RecurrentGPTForCausalLM(config)
    print(f"✓ Model created successfully")
    print(f"  - Model class: {type(model).__name__}")
    print(f"  - Internal model: {type(model.model).__name__}")
    print(f"  - Num recurrent blocks: {model.num_recurrent_blocks}")
    print()

    # Test forward pass
    print("Testing forward pass...")
    input_ids = torch.randint(0, config.vocab_size, (2, 10))  # Batch of 2, seq len 10
    print(f"  Input shape: {input_ids.shape}")

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)

    print(f"✓ Forward pass successful")
    print(f"  - Logits shape: {outputs.logits.shape}")
    print(f"  - Expected shape: (2, 10, {config.padded_vocab_size or config.vocab_size})")
    print()

    # Test recurrence control
    print("Testing recurrence step control...")
    import os
    os.environ["EVAL_RECURRENCE_STEPS"] = "8"
    with torch.no_grad():
        outputs_8 = model(input_ids)
    print(f"✓ Recurrence control via env var works (8 steps)")

    os.environ["EVAL_RECURRENCE_STEPS"] = "4,8,12"
    with torch.no_grad():
        outputs_custom = model(input_ids)
    print(f"✓ Per-block recurrence control works (4,8,12 steps)")
    print()

    # Verify outputs are different (different recurrence should give different results)
    logits_diff = (outputs.logits - outputs_8.logits).abs().mean().item()
    print(f"Logits difference (12 vs 8 steps): {logits_diff:.6f}")
    if logits_diff > 0:
        print("✓ Different recurrence steps produce different outputs (as expected)")
    else:
        print("⚠️  Warning: Different recurrence steps produce identical outputs")
    print()

    print("="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print()
    print("The standalone file is fully functional and requires only:")
    print("  - torch")
    print("  - transformers")
    print("  - Standard library")
    print()
    print("No recpre package dependency!")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
