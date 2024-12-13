#!/bin/bash

# Define sample inputs
TASKS=("code_translation" "assert_generation" "code_repair_short" "code_repair_long" "summarization")
MODEL_DIRS=("codet5-base" "codet5p-220m" "codet5p-770m")
TUNING_METHODS=("concatpervector" "full-finetuning" "no-finetuning" "no-gnn" "lora" "prefix-tuning" "prompt-tuning")
SEEDS=("seed_8_1" "seed_18_1" "seed_99_1")
OUTPUT_DIR="../compute_metrics_outputs"

# Convert arrays to space-separated strings
TASKS_STR="${TASKS[@]}"
MODEL_DIRS_STR="${MODEL_DIRS[@]}"
TUNING_METHODS_STR="${TUNING_METHODS[@]}"
SEEDS_STR="${SEEDS[@]}"

# Run the Python script with the sample inputs
python ../compute_metrics_per_instance.py --tasks $TASKS_STR --model_dirs $MODEL_DIRS_STR --tuning_methods $TUNING_METHODS_STR --seeds $SEEDS_STR --output_dir $OUTPUT_DIR
