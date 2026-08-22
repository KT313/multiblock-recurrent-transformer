#!/usr/bin/env python3
"""
Interactive text generation script for the converted HuggingFace model.

Load the model once, then accept text prompts from the terminal and generate
completions. Type 'exit' or 'quit' to stop, or press Ctrl+C.
"""

import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM
from typing import Optional
import time

# Add recurrent-pretraining to path for tokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "recurrent-pretraining"))
from recpre.tokenizer import Tokenizer


# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = "/path/to/shared_storage/recpre/outputs/hf_models/final-model-step-00151060-standalone"
TOKENIZER_PATH = "/path/to/shared_storage/recpre/artifacts/tokenizer_llama32k"

MAX_NEW_TOKENS = 512 # 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Command-line Arguments
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive text generation with recurrent model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default recurrence (12,12,12)
  python3 test_interactive_generation.py

  # Fast generation with minimal recurrence
  python3 test_interactive_generation.py --steps "1,1,1"

  # Medium recurrence
  python3 test_interactive_generation.py --steps "8,8,8"

  # Full recurrence
  python3 test_interactive_generation.py --steps "12,12,12"

  # Per-block control
  python3 test_interactive_generation.py --steps "4,8,12"
        """
    )

    parser.add_argument(
        "--steps",
        type=str,
        default="12,12,12",
        help='Recurrence steps per block (comma-separated, e.g., "1,1,1" or "12,12,12"). Default: "12,12,12"'
    )

    parser.add_argument(
        "--script-mode",
        action="store_true",
        help="Run automated benchmark script with 4 prompts × 8 step configurations (32 total outputs)"
    )

    return parser.parse_args()


# ============================================================================
# Progress Callback for Generation
# ============================================================================

class GenerationProgressCallback:
    """Callback to show generation progress."""

    def __init__(self, max_new_tokens, batch_size=1, update_interval=10):
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.start_time = None
        self.last_update_tokens = 0

    def __call__(self, input_ids, scores, **kwargs):
        """Called after each token generation."""
        if self.start_time is None:
            self.start_time = time.time()
            self.last_update_tokens = 0

        # Calculate tokens generated (total length - input length)
        current_length = input_ids.shape[1]

        # Update every N tokens or at the end
        if current_length - self.last_update_tokens >= self.update_interval:
            elapsed = time.time() - self.start_time
            tokens_per_sec = (current_length - self.last_update_tokens) / elapsed if elapsed > 0 else 0

            # Print progress on same line
            progress_pct = min(100, (current_length / (current_length + self.max_new_tokens)) * 200)
            print(f"\r[Progress] Tokens generated: {current_length} | Speed: {tokens_per_sec:.1f} tok/s",
                  end='', flush=True)

            self.last_update_tokens = current_length

        return False  # Don't stop generation


# ============================================================================
# Model Loading
# ============================================================================

def load_model_and_tokenizer():
    """Load the model and tokenizer."""
    print("Loading model and tokenizer...")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Tokenizer: {TOKENIZER_PATH}")
    print(f"  Device: {DEVICE}")
    print()

    # Load tokenizer
    tokenizer = Tokenizer(TOKENIZER_PATH)

    # Set pad_id to -100 (same as train.py line 317)
    # This is the ignore_index for loss calculation
    tokenizer.pad_id = -100
    print(f"  Set pad_id to -100 (same as training)")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # Note: transformers may show deprecation warning, ignore it
    )
    model = model.to(DEVICE)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print()

    return model, tokenizer


# ============================================================================
# Generation Functions
# ============================================================================

def manual_generate(model, input_ids, max_new_tokens, eos_token_id):
    """
    Manual autoregressive generation loop.
    Fallback when model.generate() has issues.
    Uses greedy decoding (temperature=0.7).
    """
    generated_ids = input_ids.clone()

    print("[Manual generation with progress tracking]")
    for i in range(max_new_tokens):
        # Show progress every 10 tokens
        if i % 10 == 0:
            print(f"\r[Progress] Tokens generated: {generated_ids.shape[1]} / {input_ids.shape[1] + max_new_tokens}", end='', flush=True)

        # Forward pass
        outputs = model(generated_ids)
        logits = outputs.logits

        # Get logits for the last token
        next_token_logits = logits[:, -1, :]

        # Greedy decoding: pick the token with highest probability
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Append to sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        # Stop if EOS token is generated
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    print()  # New line after progress
    return generated_ids


def batch_generate_text(model, tokenizer, prompts: list, max_new_tokens: int = MAX_NEW_TOKENS):
    """Generate text continuations for multiple prompts in a batch."""

    print(f"[DEBUG] Starting batch_generate_text() with {len(prompts)} prompts")

    # Tokenize all prompts
    all_input_ids = []
    for i, prompt in enumerate(prompts):
        print(f"[DEBUG] Tokenizing prompt {i+1}/{len(prompts)}")
        input_ids = tokenizer.encode(prompt, bos=True, eos=False)

        # Handle both list and tensor returns
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        elif torch.is_tensor(input_ids):
            input_ids = input_ids.clone().detach()

        all_input_ids.append(input_ids)

    # Find max length for padding
    max_len = max(ids.shape[0] for ids in all_input_ids)
    print(f"[DEBUG] Max input length: {max_len}")

    # Pad all sequences to same length
    # IMPORTANT: Use LEFT padding for decoder-only models (autoregressive generation)
    padded_input_ids = []
    attention_masks = []

    for input_ids in all_input_ids:
        seq_len = input_ids.shape[0]
        padding_length = max_len - seq_len

        # Pad with eos_id on the LEFT (beginning of sequence)
        pad_id = tokenizer.eos_id if tokenizer.eos_id is not None else 2
        padding = torch.full((padding_length,), pad_id, dtype=torch.long)
        padded_ids = torch.cat([padding, input_ids])  # LEFT padding

        # Create attention mask (0 for padding on left, 1 for real tokens on right)
        mask = torch.cat([
            torch.zeros(padding_length, dtype=torch.long),  # 0 for padding
            torch.ones(seq_len, dtype=torch.long)           # 1 for real tokens
        ])

        padded_input_ids.append(padded_ids)
        attention_masks.append(mask)

    # Stack into batch
    input_ids_batch = torch.stack(padded_input_ids).to(DEVICE)
    attention_mask_batch = torch.stack(attention_masks).to(DEVICE)

    print(f"[DEBUG] Batch input_ids shape: {input_ids_batch.shape}")
    print(f"[DEBUG] Batch attention_mask shape: {attention_mask_batch.shape}")

    # Verify attention mask is correct (left-padded)
    for i in range(min(2, len(prompts))):  # Check first 2 prompts
        mask = attention_masks[i]
        first_one = (mask == 1).nonzero(as_tuple=True)[0][0].item() if (mask == 1).any() else 0
        print(f"[DEBUG] Prompt {i+1}: padding_len={first_one}, real_tokens={mask.sum().item()}")
        print(f"[DEBUG]   First 5 tokens: {input_ids_batch[i, :5].tolist()}")
        print(f"[DEBUG]   First 5 mask:   {attention_mask_batch[i, :5].tolist()}")
        print(f"[DEBUG]   Last 5 tokens:  {input_ids_batch[i, -5:].tolist()}")
        print(f"[DEBUG]   Last 5 mask:    {attention_mask_batch[i, -5:].tolist()}")

    print(f"[Generating {max_new_tokens} tokens for {len(prompts)} prompts with temperature=0.7...]")
    print()

    # Create position_ids that account for left-padding (critical for RoPE!)
    # For left-padded sequences, position_ids should start from 0 at the first real token
    position_ids = attention_mask_batch.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask_batch == 0, 0)  # Padding positions get 0
    print(f"[DEBUG] Created position_ids for left-padding (fixes RoPE)")
    print(f"[DEBUG]   Prompt 1 position_ids: {position_ids[0].tolist()}")
    print(f"[DEBUG]   Prompt 2 position_ids: {position_ids[1].tolist()}")

    # Create progress callback
    progress_callback = GenerationProgressCallback(max_new_tokens, batch_size=len(prompts), update_interval=10)

    # Generate
    with torch.no_grad():
        try:
            output_ids = model.generate(
                input_ids_batch,
                attention_mask=attention_mask_batch,
                position_ids=position_ids,  # Explicitly pass position_ids for correct RoPE with left-padding
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                temperature=None,
                pad_token_id=tokenizer.eos_id,  # Use eos_id=2 (valid embedding index)
                eos_token_id=tokenizer.eos_id,  # 2
                stopping_criteria=[progress_callback],
            )
            print()  # New line after progress
            print(f"[DEBUG] Batch generation succeeded! output shape: {output_ids.shape}")
        except Exception as e:
            print(f"[DEBUG] Batch generation failed: {e}")
            print(f"[DEBUG] Falling back to sequential generation...")

            # Fallback to sequential generation
            results = []
            for i, prompt in enumerate(prompts):
                print(f"[DEBUG] Generating {i+1}/{len(prompts)} sequentially...")
                full_output, generated_only = generate_text(model, tokenizer, prompt, max_new_tokens)
                results.append((full_output, generated_only))
            return results

    # Decode all outputs
    results = []
    for i in range(len(prompts)):
        print(f"[DEBUG] Decoding output {i+1}/{len(prompts)}")

        # Get original input length (before padding)
        original_len = all_input_ids[i].shape[0]
        padding_len = max_len - original_len

        # With left-padding, the structure is:
        # [padding: padding_len] [original_input: original_len] [generated: N tokens]

        # Skip padding tokens for full output (starts at padding_len)
        full_output = tokenizer.decode(output_ids[i][padding_len:].cpu())

        # Generated tokens start after padding + original input (at index max_len)
        generated_ids = output_ids[i][max_len:]
        generated_only = tokenizer.decode(generated_ids.cpu())

        results.append((full_output, generated_only))

    print("[DEBUG] batch_generate_text() completed successfully")
    return results


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS):
    """Generate text continuation from a prompt."""

    print("[DEBUG] Starting generate_text()")
    print(f"[DEBUG] Prompt: {repr(prompt)}")

    # Tokenize input
    print("[DEBUG] Calling tokenizer.encode()...")
    input_ids = tokenizer.encode(prompt, bos=True, eos=False)
    print(f"[DEBUG] tokenizer.encode() returned: type={type(input_ids)}, value={input_ids}")

    # Handle both list and tensor returns from tokenizer
    if isinstance(input_ids, list):
        print("[DEBUG] input_ids is a list, converting to tensor...")
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        print(f"[DEBUG] After conversion: type={type(input_ids)}, shape={input_ids.shape}, dtype={input_ids.dtype}")
    elif torch.is_tensor(input_ids):
        print("[DEBUG] input_ids is already a tensor, cloning...")
        print(f"[DEBUG] Before clone: type={type(input_ids)}, shape={input_ids.shape}, dtype={input_ids.dtype}")
        input_ids = input_ids.clone().detach()
        print(f"[DEBUG] After clone: type={type(input_ids)}, shape={input_ids.shape}, dtype={input_ids.dtype}")
    else:
        print(f"[DEBUG] ERROR: input_ids is neither list nor tensor! type={type(input_ids)}")

    # Ensure correct shape and device
    print(f"[DEBUG] Checking shape... ndim={input_ids.ndim}")
    if input_ids.ndim == 1:
        print("[DEBUG] ndim=1, unsqueezing...")
        input_ids = input_ids.unsqueeze(0)
        print(f"[DEBUG] After unsqueeze: shape={input_ids.shape}")

    print(f"[DEBUG] Moving to device {DEVICE}...")
    input_ids = input_ids.to(DEVICE)
    print(f"[DEBUG] After to(device): shape={input_ids.shape}, dtype={input_ids.dtype}, device={input_ids.device}")

    print(f"\n[Input tokens: {input_ids.shape[1]}]")
    print(f"[Generating {max_new_tokens} tokens with temperature=0.7 (greedy)...]")
    print()

    # Create attention mask (all ones for input tokens)
    print("[DEBUG] Creating attention mask...")
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    print(f"[DEBUG] attention_mask: shape={attention_mask.shape}, dtype={attention_mask.dtype}, device={attention_mask.device}")

    # Create progress callback
    progress_callback = GenerationProgressCallback(max_new_tokens, batch_size=1, update_interval=10)

    # Generate
    print("[DEBUG] Entering generation with torch.no_grad()...")
    with torch.no_grad():
        try:
            print("[DEBUG] About to call model.generate()...")
            print(f"[DEBUG] Arguments:")
            print(f"[DEBUG]   input_ids: type={type(input_ids)}, shape={input_ids.shape}")
            print(f"[DEBUG]   attention_mask: type={type(attention_mask)}, shape={attention_mask.shape}")
            print(f"[DEBUG]   max_new_tokens: {max_new_tokens}")
            print(f"[DEBUG]   do_sample: True")
            print(f"[DEBUG]   temperature: 0.7")
            print(f"[DEBUG]   pad_token_id: {tokenizer.eos_id}")  # 2 (valid embedding index)
            print(f"[DEBUG]   eos_token_id: {tokenizer.eos_id}")  # 2

            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                pad_token_id=tokenizer.eos_id,  # Use eos_id=2 (valid embedding index)
                eos_token_id=tokenizer.eos_id,  # 2
                stopping_criteria=[progress_callback],
            )
            print()  # New line after progress
            print(f"[DEBUG] model.generate() succeeded! output_ids type={type(output_ids)}, shape={output_ids.shape}")
        except Exception as e:
            print(f"[DEBUG] Exception caught in model.generate()!")
            print(f"[DEBUG] Exception type: {type(e)}")
            print(f"[DEBUG] Exception message: {e}")
            import traceback
            print("[DEBUG] Full traceback:")
            traceback.print_exc()
            print(f"[DEBUG] Falling back to manual generation loop...")
            output_ids = manual_generate(
                model, input_ids, max_new_tokens, tokenizer.eos_id
            )
            print(f"[DEBUG] manual_generate() completed! output_ids type={type(output_ids)}, shape={output_ids.shape}")

    # Decode full output (input + generated)
    # Note: tokenizer.decode() expects a tensor, not a list
    print("[DEBUG] Calling tokenizer.decode() for full output...")
    output_text = tokenizer.decode(output_ids[0].cpu())
    print(f"[DEBUG] Decoded full output, length: {len(output_text)}")

    # Also decode just the generated part
    print("[DEBUG] Extracting generated tokens...")
    generated_ids = output_ids[0][input_ids.shape[1]:]
    print(f"[DEBUG] generated_ids type: {type(generated_ids)}, shape: {generated_ids.shape if torch.is_tensor(generated_ids) else 'N/A'}")

    print("[DEBUG] Calling tokenizer.decode() for generated part...")
    generated_text = tokenizer.decode(generated_ids.cpu())
    print(f"[DEBUG] Decoded generated text, length: {len(generated_text)}")

    print("[DEBUG] generate_text() completed successfully")
    return output_text, generated_text


# ============================================================================
# Script Mode (Automated Benchmark)
# ============================================================================

def run_script_mode(model, tokenizer):
    """Run automated benchmark with multiple prompts and step configurations."""

    # Define prompts
    prompts = [
        "Please write a python program that calculates the first 10 prime numbers.",
        "Please write a cpp program that calculates the first 10 prime numbers.",
        "Please write a java program that calculates the first 10 prime numbers.",
        "Please tell me funny cat facts.",
        "How to bake a tasty fruit cake:",
        "How can i bake a tasty fruit cake?",
        "Please tell me how to bake a tasty fruit cake.",
        "What are recent world news?"
    ]

    # Define step configurations
    step_configs = [
        # "12,12,12",
        "1,1,1",
        "2,2,2",
        "4,4,4",
        "8,8,8",
        "16,16,16",
        "32,32,32",
        "4,4,16",
        "4,16,4",
        "16,4,4"
    ]

    # Store all results for final overview
    results = []

    print("=" * 80)
    print("SCRIPT MODE: Running automated benchmark (SEQUENTIAL)")
    print("=" * 80)
    print(f"Prompts: {len(prompts)}")
    print(f"Step configurations: {len(step_configs)}")
    print(f"Total outputs: {len(prompts) * len(step_configs)}")
    print("=" * 80)
    print()

    total = len(prompts) * len(step_configs)
    counter = 0

    # Run sequentially: step configs first, then prompts
    for step_idx, steps in enumerate(step_configs, 1):

        print("\n" + "=" * 80)
        print(f"Step configuration [{step_idx}/{len(step_configs)}]: {steps}")
        print("=" * 80)

        # Set recurrence steps for this configuration
        os.environ["EVAL_RECURRENCE_STEPS"] = steps

        # Generate each prompt sequentially
        for prompt_idx, prompt in enumerate(prompts, 1):
            counter += 1

            print("\n" + "-" * 80)
            print(f"[{counter}/{total}] Prompt: {prompt[:60]}...")
            print(f"         Steps: {steps}")
            print("-" * 80)

            # Generate
            try:
                full_output, generated_only = generate_text(model, tokenizer, prompt)

                # Store result
                results.append({
                    "prompt": prompt,
                    "steps": steps,
                    "full_output": full_output,
                    "generated_only": generated_only
                })

                # Print output
                print("Generated output:")
                print("-" * 80)
                print(generated_only)
                print("-" * 80)

            except Exception as e:
                print(f"\nERROR: Generation failed")
                print(f"Exception: {e}")
                import traceback
                traceback.print_exc()

                # Store error result
                results.append({
                    "prompt": prompt,
                    "steps": steps,
                    "full_output": None,
                    "generated_only": f"ERROR: {e}"
                })

        print(f"\nCompleted step configuration {step_idx}/{len(step_configs)}")
        print()

    # Print final overview
    print("\n\n" + "=" * 80)
    print("FINAL OVERVIEW - ALL OUTPUTS")
    print("=" * 80)
    print()

    for idx, result in enumerate(results, 1):
        print(f"\n{'=' * 80}")
        print(f"[{idx}/{total}]")
        print(f"Prompt: {result['prompt']}")
        print(f"Steps:  {result['steps']}")
        print(f"{'-' * 80}")
        print("Generated output:")
        print(f"{'-' * 80}")
        print(result['generated_only'])
        print()

    print("=" * 80)
    print("Script mode completed!")
    print("=" * 80)


# ============================================================================
# Interactive Loop
# ============================================================================

def main():
    """Main entry point - either interactive or script mode."""

    # Parse command-line arguments
    args = parse_args()

    # Load model once
    print("=" * 80)
    print("Loading model...")
    print("=" * 80)
    print()
    model, tokenizer = load_model_and_tokenizer()

    # Check if script mode is enabled
    if args.script_mode:
        # Run automated benchmark
        run_script_mode(model, tokenizer)

    else:
        # Run interactive mode
        # Set recurrence steps via environment variable
        os.environ["EVAL_RECURRENCE_STEPS"] = args.steps

        print("=" * 80)
        print("Interactive Text Generation")
        print("=" * 80)
        print()
        print(f"Configuration:")
        print(f"  Recurrence steps: {args.steps}")
        print(f"  Max new tokens: {MAX_NEW_TOKENS}")
        print()

        print("=" * 80)
        print("Ready! Enter text prompts to generate continuations.")
        print("Type 'exit' or 'quit' to stop, or press Ctrl+C.")
        print("=" * 80)
        print()

        try:
            while True:
                # Get input from user
                try:
                    prompt = input(">>> ")
                except EOFError:
                    # Handle Ctrl+D
                    print("\nExiting...")
                    break

                # Check for exit commands
                if prompt.strip().lower() in ["exit", "quit", ""]:
                    if prompt.strip() == "":
                        continue
                    print("Exiting...")
                    break

                # Generate
                try:
                    full_output, generated_only = generate_text(model, tokenizer, prompt)

                    print("-" * 80)
                    print("Full output (prompt + generated):")
                    print("-" * 80)
                    print(full_output)
                    print()
                    print("-" * 80)
                    print("Generated only:")
                    print("-" * 80)
                    print(generated_only)
                    print()
                    print("=" * 80)
                    print()

                except Exception as e:
                    print(f"Error during generation: {e}")
                    print()

        except KeyboardInterrupt:
            # Handle Ctrl+C
            print("\n\nInterrupted by user. Exiting...")

    # Cleanup (runs for both modes)
    print("\nUnloading model...")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Done!")


if __name__ == "__main__":
    main()
