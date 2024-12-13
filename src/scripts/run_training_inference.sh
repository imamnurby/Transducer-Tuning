#!/bin/bash
# echo "Waiting for 1h to run the scipt!"
# sleep 2h
# Need to check

DO_TRAINING="1"
DO_INFERENCE="1"

TASK="code_translation"

## TRAINING
project_dir_bases=(    
    "../experiments/$TASK/codet5p-220m/transducer_tuning"
    "../experiments/$TASK/codet5p-220m/full_finetuning"
    "../experiments/$TASK/codet5p-220m/lora"
    # "../experiments/$TASK/codet5p-220m/no_finetuning"
    "../experiments/$TASK/codet5p-220m/linear_adapter"
    "../experiments/$TASK/codet5p-220m/prefix-tuning"
    "../experiments/$TASK/codet5p-220m/prompt-tuning"

    "../experiments/$TASK/codet5p-770m/transducer_tuning"
    "../experiments/$TASK/codet5p-770m/full_finetuning"
    "../experiments/$TASK/codet5p-770m/lora"
    # "../experiments/$TASK/codet5p-770m/no_finetuning"
    "../experiments/$TASK/codet5p-770m/linear_adapter"
    "../experiments/$TASK/codet5p-770m/prefix-tuning"
    "../experiments/$TASK/codet5p-770m/prompt-tuning"
)
config_generation_value="../experiments/$TASK/generation.yml"
project_name_value=""
sapt_classnames=(
    "SAPTModelConcatPerVectorLinear"
    "full-finetuning"
    "lora"
    # "no-finetuning"
    "SAPTModelNoGNN"
    "prefix-tuning"
    "prompt-tuning"

    "SAPTModelConcatPerVectorLinear"
    "full-finetuning"
    "lora"
    # "no-finetuning"
    "SAPTModelNoGNN"
    "prefix-tuning"
    "prompt-tuning"
)
random_seeds=(18 99)
# Default
is_debug_value=0
# Dataset
dataset_path_value="/data/datasets/fix/$TASK/codet5_dataset_ready"
max_seq_len_value=400
dataset_portion_values=(1)
# Checkpoint
save_total_limit_value=1
checkpointing_must_save_steps_value="-1"    
checkpointing_must_save_epochs_value="1,"
resume_from_checkpoint_value=""
# Validation Metric
validation_metric_value="no_validation"
# Batch Size
validation_batch_size_value=32
training_batch_size_value=8
# Tracking
with_tracking_value=0
resume_tracking_value=""
notes_value="" 
run_name_value=""
tags_value=""
# Learning
lr_value=0.0003
num_epochs_value=1
max_grad_norm_value=1.0
gradient_accumulation_steps_value=1
num_warmup_steps_value=0
mixed_precision_value="bf16"
# Other
cpu_value=0  # Include to enable, empty to disable
num_workers_value=0
early_stop_threshold_value=3


## INFERENCE
checkpoints=(
    "checkpoints_targeted_epochs/epoch_1"
    # ""
)
split_value="test"
# Default
is_debug_value=0 
# Batch Size
batch_size_inference_value=8

if [[ -n "$DO_TRAINING" ]]; then
    for index in "${!project_dir_bases[@]}"; do         
        project_dir_base="${project_dir_bases[index]}"
        sapt_classname_value="${sapt_classnames[index]}"
        for seed_value in "${random_seeds[@]}"; do
            for dataset_portion_value in "${dataset_portion_values[@]}"; do
                # Create a new directory for the current seed
                project_dir_value="${project_dir_base}/seed_${seed_value}_${dataset_portion_value}"
                config_model_value="${project_dir_base}/model.yml"
                mkdir -p "$project_dir_value"
                echo "Running train.py"
                python ../train.py \
                    --project_dir "$project_dir_value" \
                    --config_model $config_model_value \
                    --config_generation "$config_generation_value" \
                    --project_name "$project_name_value" \
                    --sapt_classname "$sapt_classname_value" \
                    --seed $seed_value \
                    --is_debug $is_debug_value \
                    --dataset_path "$dataset_path_value" \
                    --max_seq_len $max_seq_len_value \
                    --dataset_portion $dataset_portion_value \
                    --save_total_limit $save_total_limit_value \
                    --checkpointing_must_save_steps "$checkpointing_must_save_steps_value" \
                    --checkpointing_must_save_epochs "$checkpointing_must_save_epochs_value" \
                    --resume_from_checkpoint "$resume_from_checkpoint_value" \
                    --validation_metric "$validation_metric_value" \
                    --validation_batch_size $validation_batch_size_value \
                    --training_batch_size $training_batch_size_value \
                    --with_tracking $with_tracking_value \
                    --resume_tracking "$resume_tracking_value" \
                    --notes "$notes_value" \
                    --run_name "$run_name_value" \
                    --tags "$tags_value" \
                    --lr $lr_value \
                    --num_epochs $num_epochs_value \
                    --max_grad_norm $max_grad_norm_value \
                    --gradient_accumulation_steps $gradient_accumulation_steps_value \
                    --num_warmup_steps $num_warmup_steps_value \
                    --mixed_precision "$mixed_precision_value" \
                    --cpu $cpu_value \
                    --num_workers $num_workers_value \
                    --early_stop_threshold $early_stop_threshold_value
            done
        done
    done
fi

if [[ -n "$DO_INFERENCE" ]]; then
    for index in "${!project_dir_bases[@]}"; do 
        project_dir_base="${project_dir_bases[index]}"
        sapt_classname_value="${sapt_classnames[index]}"
        config_model_value="${project_dir_base}/model.yml"
        
        for seed_value in "${random_seeds[@]}"; do    
            # Check if seed_value is not empty
            if [[ -n "$seed_value" ]]; then
                project_dir_value="${project_dir_base}/seed_${seed_value}"
            else
                project_dir_value="${project_dir_base}"
            fi
            
            for num_data in "${dataset_portion_values[@]}"; do
                project_dir_value_w_num_data="${project_dir_value}_${num_data}"

                for checkpoint_value in "${checkpoints[@]}"; do
                    
                    if [[ "$sapt_classname_value" != "no-finetuning" ]]; then
                        checkpoint_dir="${project_dir_value_w_num_data}/${checkpoint_value}"
                        echo "Running inference.py using checkpoint ${checkpoint_dir}"
                    else
                        project_dir_value="${project_dir_base}"
                        checkpoint_dir=""
                        echo "Running inference.py without loading any checkpoints"
                    fi
                    
                    python ../inference.py \
                        --project_dir "$project_dir_value_w_num_data" \
                        --config_model $config_model_value \
                        --config_generation "$config_generation_value" \
                        --project_name "$project_name_value" \
                        --sapt_classname "$sapt_classname_value" \
                        --seed $seed_value \
                        --save_output 1 \
                        --is_debug $is_debug_value \
                        --dataset_path "$dataset_path_value" \
                        --split "$split_value" \
                        --max_seq_len $max_seq_len_value \
                        --dataset_portion 1 \
                        --save_total_limit 1 \
                        --resume_from_checkpoint "$checkpoint_dir" \
                        --batch_size $batch_size_inference_value \
                        --with_tracking $with_tracking_value \
                        --resume_tracking "$resume_tracking_value" \
                        --notes "$notes_value" \
                        --run_name "$run_name_value" \
                        --tags "$tags_value" \
                        --gradient_accumulation_steps $gradient_accumulation_steps_value \
                        --mixed_precision "no" \
                        --cpu $cpu_value \
                        --num_workers $num_workers_value
                done
            done
        done
    done
fi
