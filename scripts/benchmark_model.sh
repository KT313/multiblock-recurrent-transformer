#!/bin/bash
# (c) 2025-2026 Tobias Kerner, part of the multi-block recurrent thesis work.
# Released under Apache-2.0 alongside seal-rg/recurrent-pretraining code. See LICENSE.

# Script to benchmark a HuggingFace model with lm-eval-harness
# Usage: ./benchmark_model.sh <path_to_model> [tasks] [device] [recurrence_steps]

set -e  # Exit on error

# Check if model path is provided
if [ -z "$1" ]; then
    echo "Error: Model path not provided"
    echo ""
    echo "Usage: $0 <path_to_model> [tasks] [device] [recurrence_steps]"
    echo ""
    echo "Arguments:"
    echo "  path_to_model     Path to the HuggingFace model directory (required)"
    echo "  tasks             Comma-separated benchmark tasks (default: mmlu,hellaswag,arc_challenge)"
    echo "  device            Device to use (default: cuda:0)"
    echo "  recurrence_steps  Recurrence steps configuration (optional):"
    echo "                    - Omit for model default (mean_recurrence from config)"
    echo "                    - Single value: '12' (same for all blocks)"
    echo "                    - Per-block: '4,4,4' or '4,12,4' (comma-separated, one per block)"
    echo ""
    echo "Examples:"
    echo "  # Use default recurrence steps from model config"
    echo "  $0 outputs/hf_models/my_model"
    echo ""
    echo "  # Specify custom tasks and device"
    echo "  $0 outputs/hf_models/my_model mmlu,hellaswag cuda:0"
    echo ""
    echo "  # Fast inference with 4 steps per block"
    echo "  $0 outputs/hf_models/my_model mmlu,hellaswag cuda:0 4,4,4"
    echo ""
    echo "  # Ablation: different steps per block"
    echo "  $0 outputs/hf_models/my_model mmlu cuda:0 4,12,4"
    exit 1
fi

MODEL_PATH="$1"
TASKS="${2:-mmlu,hellaswag,arc_challenge}"  # Default tasks if not provided
DEVICE="${3:-cuda:0}"  # Default device if not provided
RECURRENCE_STEPS="${4:-}"  # Optional: recurrence steps configuration

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

# Extract model name from path (last directory name)
MODEL_NAME=$(basename "$MODEL_PATH")

# Create output directory with recurrence steps suffix if custom steps provided
if [ -n "$RECURRENCE_STEPS" ]; then
    # Convert "4,4,4" to "4_4_4" for directory name
    STEPS_SUFFIX=$(echo "$RECURRENCE_STEPS" | tr ',' '_')
    OUTPUT_DIR="outputs/benchmark/${MODEL_NAME}_steps_${STEPS_SUFFIX}"
else
    OUTPUT_DIR="outputs/benchmark/${MODEL_NAME}"
fi
mkdir -p "$OUTPUT_DIR"

echo "================================================================================"
echo "Benchmarking Model"
echo "================================================================================"
echo "Model: $MODEL_PATH"
echo "Model Name: $MODEL_NAME"
echo "Tasks: $TASKS"
echo "Device: $DEVICE"
if [ -n "$RECURRENCE_STEPS" ]; then
    echo "Recurrence Steps: $RECURRENCE_STEPS (custom)"
else
    echo "Recurrence Steps: default (from model config)"
fi
echo "Output Directory: $OUTPUT_DIR"
echo "================================================================================"

# Set HuggingFace cache directory
export HF_HOME=/path/to/fast_storage/.cache

# Set recurrence steps if provided
if [ -n "$RECURRENCE_STEPS" ]; then
    export EVAL_RECURRENCE_STEPS="$RECURRENCE_STEPS"
    echo "Using custom recurrence steps: $RECURRENCE_STEPS"
fi

# Run lm-eval and save output
echo ""
echo "Running evaluation..."
lm-eval --model hf \
    --model_args pretrained="${MODEL_PATH}",trust_remote_code=True \
    --tasks "$TASKS" \
    --device "$DEVICE" \
    --output_path "$OUTPUT_DIR" \
    --batch_size auto \
    --log_samples 2>&1 | tee "${OUTPUT_DIR}/eval_log.txt"

echo ""
echo "================================================================================"
echo "Evaluation Complete!"
echo "================================================================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"
echo "================================================================================"
