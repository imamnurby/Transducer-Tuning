# adapted from https://github.com/huggingface/accelerate/tree/main/examples
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Third-party library imports
from torch.optim import AdamW
from transformers import (
    get_constant_schedule_with_warmup, 
    get_cosine_schedule_with_warmup,
    set_seed,
)
from tqdm.auto import tqdm
import pandas as pd
import wandb 
import time
# Local file imports
from config import ConfigTraining
from metrics import (
    METRIC_NAME_TO_LABEL_KEY_MAPPING,
    METRIC_TO_EVALUATOR,
    # MODEL_TO_POSTPROCESSING,
    METRIC_NAME_TO_LABEL_KEY_MAPPING,
    normalize_string,
    PostProcessor
)
from utils import (
    save_checkpoint,
    handle_checkpoint_resume, 
    load_model_tokenizer_datacollator,
    initialize_accelerator,
    do_inference_generative,
    maybe_delete_checkpoints,
    prepare_dataloader,
    do_inference_classification,
    get_gpu_utilization,
    compute_trainable_params
)
from modelling import VALID_BASELINES
import os
import json
import yaml
import argparse
import pdb
from typing import Dict

############################################################################
# Adapted from: https://github.com/huggingface/accelerate/tree/main/examples
############################################################################
def training_function(config_training: ConfigTraining, config_model: Dict):
    # Initialize accelerator
    accelerator = initialize_accelerator(config_training, config_model)
    
    # Set seed
    set_seed(config_training.seed)

    # Load model, tokenizer, data collator, and configuration
    tokenizer, model, data_collator, generation_config = load_model_tokenizer_datacollator(config_model, config_training)
    memory_consumption = {
        "initial": 0,
        "end": 0,
        "trainable_param": 0
    }
    
    memory_consumption["initial"] = get_gpu_utilization()
    memory_consumption["trainable_param"] = compute_trainable_params(config_training, model)

    # Load dataloader
    train_dataloader = prepare_dataloader(
        dataset_path=config_training.dataset_path,
        dataset_portion=config_training.dataset_portion,
        split="train",
        data_collator=data_collator,
        batch_size=config_training.training_batch_size,
        num_workers=config_training.num_workers,
        pin_memory=True,
        seed=config_training.seed,
        is_debug=config_training.is_debug,
    )

    evaluator = METRIC_TO_EVALUATOR.get(config_training.validation_metric, None)
    label_key = METRIC_NAME_TO_LABEL_KEY_MAPPING.get(config_training.validation_metric, None)
    if evaluator:
        label_key = METRIC_NAME_TO_LABEL_KEY_MAPPING[config_training.validation_metric]
        eval_dataloader = prepare_dataloader(
            dataset_path=config_training.dataset_path,
            dataset_portion=config_training.dataset_portion,
            split="eval",
            data_collator=data_collator,
            batch_size=config_training.validation_batch_size,
            num_workers=config_training.num_workers,
            pin_memory=True,
            seed=config_training.seed,
            is_debug=config_training.is_debug,
        )
            
    # Instantiate optimizer
    # to do: pass only the trainable params to make sure that the updated params are those that are not frozen
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.00},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(params=optimizer_grouped_parameters, lr=config_training.lr)
    # optimizer = AdamW(params=model.parameters(), lr=config_training.lr)


    # Instantiate scheduler
    lr_scheduler = get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config_training.num_warmup_steps,
        # num_training_steps=(len(train_dataloader) * config_training.num_epochs) // config_training.gradient_accumulation_steps,
    )

    # Prepare everything
    if evaluator:
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    else:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

    # Instantiate training loop variables
    starting_epoch = 0
    if config_training.resume_from_checkpoint:
        resume_step, starting_epoch = handle_checkpoint_resume(config_training, accelerator, train_dataloader)
    global_step = 0
    total_batched_samples = 0
    stop_counter = 0
    validation_best_score = None
    validation_current_score = None
    epochs_to_checkpoint = config_training.checkpointing_must_save_epochs.split(",")
    epochs_to_checkpoint = [int(x) for x in epochs_to_checkpoint if x != '']
    steps_to_checkpoint = config_training.checkpointing_must_save_steps.split(",")
    steps_to_checkpoint = [int(x) for x in steps_to_checkpoint if x != '']
    
    if config_training.with_tracking:
        wandb.watch(model, log='all', log_freq=10)

    # Now we train the model
    loss_per_steps = []
    for epoch in range(starting_epoch, config_training.num_epochs):
        accelerator.print(f"### Epoch {epoch+1}/{config_training.num_epochs} ###")
        model.train()
        start_time = time.time()  # Start time measurement

        if config_training.sapt_classname not in VALID_BASELINES:
            model.freeze_backbone_model_params()
            model.print_trainable_params()

        if config_training.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We need to skip steps until we reach the resumed step
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            global_step += resume_step
        else:
            # After the first iteration though, we need to go back to the original dataloader
            active_dataloader = train_dataloader

        progress_bar = tqdm(active_dataloader, disable=not accelerator.is_local_main_process)
        steps_in_epoch = len(train_dataloader)
        total_steps = config_training.num_epochs*steps_in_epoch
        total_loss = 0
        temp_loss = 0
        for step, batch in enumerate(progress_bar):
            total_batched_samples += 1
            with accelerator.accumulate(model):
                
                if config_training.sapt_classname not in VALID_BASELINES:
                    outputs = model(
                        input_ids=batch["input_ids"], 
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        node_features=batch["node_features"],
                        edge_indices=batch["edge_indices"],
                        node_features_shape=batch["node_features_shape"],
                        edge_indices_shape=batch["edge_indices_shape"],
                    )
                else:
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                loss = outputs.loss

                # if gpu is more than 1, then get the average
                if accelerator.num_processes > 1:
                    loss = loss.mean()

                accelerator.backward(loss)
                loss = (loss.detach() / config_training.gradient_accumulation_steps)
            
            loss_per_steps.append(loss)
            total_loss += loss
            temp_loss += loss
            is_last_step_and_steps_less_than_grad_acc = (
                steps_in_epoch <= config_training.gradient_accumulation_steps and (step + 1) == steps_in_epoch
            )

            # Gradient clipping
            if (
                total_batched_samples % config_training.gradient_accumulation_steps == 0
                or
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                is_last_step_and_steps_less_than_grad_acc
            ):
                if is_last_step_and_steps_less_than_grad_acc:
                    accelerator.gradient_state._set_sync_gradients(True)
            
                if config_training.max_grad_norm is not None and config_training.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), config_training.max_grad_norm)
            
                optimizer.step()
                optimizer_was_run = not accelerator.optimizer_step_was_skipped
                if optimizer_was_run:
                    lr_scheduler.step()
                optimizer.zero_grad()    
                
                # Log training loss and learning rate at each step
                current_lr = lr_scheduler.get_last_lr()[0]  # Assuming a single learning rate for simplicity
                temp_loss = temp_loss/config_training.gradient_accumulation_steps
                accelerator.print(f"Epoch {epoch+1}/{config_training.num_epochs} | Step {global_step+1}/{total_steps}: Train Loss: {temp_loss:.4f}, Learning Rate: {current_lr:.2e}")
                
                if config_training.with_tracking:
                    accelerator.log(
                        {"train_loss/batch": temp_loss,
                        "learning_rate": current_lr},
                        step=global_step+1
                    )
                temp_loss=0
            
            global_step += 1

            # is_last_step = (step + 1) == steps_in_epoch
            # if global_step in steps_to_checkpoint or is_last_step:
            #     save_checkpoint(
            #         args=config_training,
            #         accelerator=accelerator,
            #         model=model,
            #         tokenizer=tokenizer,
            #         is_step="targeted_steps",
            #         step_count=global_step,
            #         save_total_limit=len(steps_to_checkpoint)+1
            #     )
        
        if evaluator:
            accelerator.print("### Evaluating the performance on the validation set ###")
            output_filepath = os.path.join(config_training.project_dir, "result_on_eval_set.json")
            if config_model["architecture"] != "encoder":
                output_dict = do_inference_generative(
                    model=model,
                    tokenizer=tokenizer,
                    eval_dataloader=eval_dataloader,
                    accelerator=accelerator,
                    generation_config=generation_config,
                    output_filepath=output_filepath,
                    use_sapt=False if config_training.sapt_classname in VALID_BASELINES else True
                )
            else:
                output_dict = do_inference_classification(
                    model=model,
                    tokenizer=tokenizer,
                    eval_dataloader=eval_dataloader,
                    accelerator=accelerator,
                    output_filepath=output_filepath,
                    use_sapt=False if config_training.sapt_classname in VALID_BASELINES else True
                )

            post_processing_fn = PostProcessor(config_model["backbone_model_path"])

            df = pd.DataFrame(output_dict)

            if config_model["architecture"] != "encoder":
                df["preds"] = df["raw_preds"].apply(lambda x: post_processing_fn(x))
                df["preds"] = df["preds"].apply(lambda x: normalize_string(x))
                df["labels"] = df["labels"].apply(lambda x: normalize_string(x))
            else:
                df["preds"] = df["raw_preds"]

            validation_current_score = evaluator(
                predictions=df["preds"].to_list(),
                truths=df[label_key].to_list(),
                batch_size=config_training.validation_batch_size)
        
            accelerator.print(f"Epoch {epoch}, {config_training.validation_metric}: {validation_current_score:.4f}")
            
            if config_training.with_tracking:
                accelerator.log(
                    {f"eval_{config_training.validation_metric}/epoch": validation_current_score,
                    "train_loss/epoch": total_loss/len(train_dataloader)},
                    step=global_step
                )

            if validation_best_score is None or validation_current_score > validation_best_score:
                validation_best_score = validation_current_score
                stop_counter = 0
            else:
                stop_counter += 1
        
            if stop_counter >= config_training.early_stop_threshold:
                accelerator.print(f"Early stopping triggered at Epoch {epoch+1}")
                break
            elif stop_counter > 0:
                accelerator.print(f"Early stopping counter: {stop_counter}/{config_training.early_stop_threshold} (Epoch {epoch})")
        else:
            if config_training.with_tracking:
                accelerator.log(
                    {"train_loss/epoch": total_loss/len(train_dataloader)},
                    step=global_step
                )

        if epoch+1 in epochs_to_checkpoint:
            save_checkpoint(
                args=config_training,
                accelerator=accelerator,
                model=model,
                tokenizer=tokenizer,
                is_step="targeted_epochs",
                step_count=epoch+1,
                save_total_limit=len(epochs_to_checkpoint)
            )

    if config_training.with_tracking:
        accelerator.end_training()

    end_time = time.time()  # End time measurement
    total_elapsed_time = end_time - start_time

    # Now, saving the total elapsed time to a text file
    output_filepath = os.path.join(config_training.project_dir, 'training_time.txt')
    with open(output_filepath, 'w') as f:
        f.write(str(total_elapsed_time))

    output_filepath = os.path.join(config_training.project_dir, 'loss.txt')
    with open(output_filepath, 'w') as f:
        for loss in loss_per_steps:
            f.write(f"{loss}\n")

    memory_consumption["end"] = get_gpu_utilization()
    output_filepath = os.path.join(config_training.project_dir, "memory_consumption.json")
    with open(output_filepath, "w") as f:
        json.dump(memory_consumption, f)

    ckpt_directory_latest = f"{config_training.project_dir}/seed_{config_training.seed}/checkpoint_latest"
    if os.path.exists(ckpt_directory_latest):
        maybe_delete_checkpoints(checkpoint_directory=ckpt_directory_latest, save_total_limit=config_training.save_total_limit, accelerator=accelerator)   

def main():    
    # Parser setup and script arguments parsing
    parser = argparse.ArgumentParser(description="Load configuration from a file.")
    parser.add_argument("--project_dir", type=str, help="Experiment logs and output directory.")
    parser.add_argument("--config_model", type=str, help="Path to the model configuration file.")
    parser.add_argument("--config_generation", default="", type=str, help="Path to the generation configuration file.")
    parser.add_argument("--project_name", type=str, help="Project name.")
    parser.add_argument("--sapt_classname", type=str, help="Which model is used to perform the experiment.")
    parser.add_argument("--seed", type=int, help="Seed for reproducibility.")
    # Debug Mode
    parser.add_argument("--is_debug", default=0, type=int, help="Enable debug mode.")
    # Dataset
    parser.add_argument("--dataset_path", type=str, help="Training dataset path.")
    parser.add_argument("--max_seq_len", default=400, type=int, help="Max sequence length for source/target.")
    parser.add_argument("--dataset_portion", default=128.0, type=float, help="How much training data is included.")
    # Checkpointing
    parser.add_argument("--save_total_limit", default=1, type=int, help="Max number of checkpoints to retain.")
    parser.add_argument("--checkpointing_must_save_steps", default="-1", type=str, help="Steps to checkpoint. Must be less than or equal save_total_limit.")
    parser.add_argument("--checkpointing_must_save_epochs", default="-1", type=str, help="Epochs to checkpoint. Must be less than or equal save_total_limit.")
    parser.add_argument("--resume_from_checkpoint", default="", type=str, help="Checkpoint path to resume training.")
    # Validation Metric
    parser.add_argument("--validation_metric", default="no validation", type=str, help="The validation metric name used to evaluate the model on the validation dataset.")
    # Batch Size
    parser.add_argument("--validation_batch_size", default=1, type=int, help="Training batch size.")
    parser.add_argument("--training_batch_size", default=1, type=int, help="Validation batch size.")
    # Tracking
    parser.add_argument("--with_tracking", default=1, type=int, help="Enable experiment tracking.")
    parser.add_argument("--resume_tracking", default="", type=str, help="Specify the id of the wandb run to resume the logging.")
    parser.add_argument("--notes", default="", type=str, help="Notes explaining what the experiment does.")
    parser.add_argument("--run_name", default="", type=str, help="Run identifier.")
    parser.add_argument("--tags", default="", type=str, help="Tags for wandb run.")
    # Learning
    parser.add_argument("--lr", default=3e-4, type=float, help="Learning rate.")
    parser.add_argument("--num_epochs", default=1, type=int, help="Number of training epochs.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max grad norm for clipping.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Gradient accumulation steps.")
    parser.add_argument("--num_warmup_steps", default=0, type=int, help="Warmup steps for learning rate scheduler.")
    parser.add_argument("--mixed_precision", default="bf16", type=str, help="Mixed precision type ('fp16' or 'bf16').")
    # Other
    parser.add_argument("--cpu", default=1, type=int, help="Use CPU for training.")
    parser.add_argument("--num_workers", default=0, type=int, help="Number of dataloader workers.")
    parser.add_argument("--early_stop_threshold", default=5, type=int, help="Early stopping threshold level.")
    
    args = parser.parse_args()
    with open(args.config_model, "r") as f2:
        config_model = (yaml.safe_load(f2))
    config_training = parser.parse_args()
    
    # Set run name and tags in wandb
    config_training.run_name = f"train-{config_training.sapt_classname}-{config_model.get('backbone_model_path')}-seed_{config_training.seed}"
    config_training.tags = f"seed={config_training.seed},sapt={config_training.sapt_classname},gnn={config_model.get('gnn_type')},type=train,backbone={config_model.get('backbone_model_path')}"
    
    try:
        training_function(config_training, config_model)
        config_training_filepath = os.path.join(config_training.project_dir, "config_training.json")
        with open(config_training_filepath, "w") as f:
            json.dump(vars(config_training), f)
    except Exception as e:
        print(f"An error occurred: {e}")
        pdb.post_mortem()
        output_filepath = os.path.join(config_training.project_dir, "error.txt")
        with open(output_filepath, "w") as f:
            f.write(str(e))
if __name__ == "__main__":
    main()
