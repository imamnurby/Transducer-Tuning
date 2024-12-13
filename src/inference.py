import json
import os
import argparse
import yaml
import pdb
from pprint import pprint
from typing import Dict
import torch

from config import ConfigInference
from utils import (
    initialize_accelerator,
    load_model_tokenizer_datacollator,
    do_inference_generative,
    do_inference_classification,
    handle_checkpoint_resume,
    prepare_dataloader
)
from modelling import VALID_BASELINES
from transformers import set_seed


def inference(config_inference: ConfigInference, config_model: Dict):
    accelerator = initialize_accelerator(config_inference, config_model)
    
    # Set seed
    set_seed(config_inference.seed)

    # Load model, tokenizer, data collator, and configuration
    tokenizer, model, data_collator, generation_config = load_model_tokenizer_datacollator(config_model, config_inference)

    # Load dataset
    test_dataloader = prepare_dataloader(
        dataset_path=config_inference.dataset_path,
        dataset_portion=config_inference.dataset_portion,
        split=config_inference.split,
        data_collator=data_collator,
        batch_size=config_inference.batch_size,
        num_workers=config_inference.num_workers,
        pin_memory=True,
        seed=config_inference.seed,
        is_debug=config_inference.is_debug,
    )

    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )

    if config_inference.resume_from_checkpoint:
        _, _ = handle_checkpoint_resume(config_inference, accelerator, test_dataloader)
    
    accelerator.print(f"Performing inference on the test set {config_inference.dataset_path}")

    if config_inference.resume_from_checkpoint:
        output_directory = os.path.join(config_inference.resume_from_checkpoint, "results")
    else:
        output_directory = os.path.join(config_inference.project_dir, "results")
    os.makedirs(output_directory, exist_ok=True)
    output_filepath = os.path.join(output_directory, "result_on_test_set.csv") if config_inference.save_output else None

    if config_model["architecture"] != "encoder":
        do_inference_generative(
            model=model,
            tokenizer=tokenizer,
            eval_dataloader=test_dataloader,
            accelerator=accelerator,
            generation_config=generation_config,
            output_filepath=output_filepath,
            use_sapt=False if config_inference.sapt_classname in VALID_BASELINES else True,
            use_logit_processor=True
        )
        
    else:
        do_inference_classification(
            model=model,
            tokenizer=tokenizer,
            eval_dataloader=test_dataloader,
            accelerator=accelerator,
            output_filepath=output_filepath,
            use_sapt=False if config_inference.sapt_classname in VALID_BASELINES else True
        )
        

def main():
    parser = argparse.ArgumentParser(description="Load configuration from a file.")
    parser.add_argument("--project_dir", type=str, help="Experiment logs and output directory.")
    parser.add_argument("--config_model", default="experiments/codet5p-220m/run-5/model.yml", type=str, help="Path to the model configuration file.")
    parser.add_argument("--config_generation", type=str, help="Path to the generation configuration file.")
    parser.add_argument("--project_name", type=str, help="Project name.")
    parser.add_argument("--sapt_classname", type=str, help="Which model is used to perform the experiment.")
    parser.add_argument("--seed", type=int, help="Seed for reproducibility.")
    parser.add_argument("--save_output", type=int, help="Whether to save the csv output or not")
    parser.add_argument("--split", type=str, help="Which split to infer from")
    # Debug Mode
    parser.add_argument("--is_debug", default=0, type=int, help="Enable debug mode.")
    # Dataset
    parser.add_argument("--dataset_path", type=str, help="Training dataset path.")
    parser.add_argument("--max_seq_len", default=400, type=int, help="Max sequence length for source/target.")
    parser.add_argument("--dataset_portion", default=128.0, type=float, help="How much training data is included.")
    # Checkpointing
    parser.add_argument("--save_total_limit", default=1, type=int, help="Max number of checkpoints to retain.")
    parser.add_argument("--resume_from_checkpoint", default="", type=str, help="Checkpoint path to resume training.")
    # Validation Metric
    # Batch Size
    parser.add_argument("--batch_size", default=1, type=int, help="Training batch size.")
    # Tracking
    parser.add_argument("--with_tracking", default=1, type=int, help="Enable experiment tracking.")
    parser.add_argument("--resume_tracking", default="", type=str, help="Specify the id of the wandb run to resume the logging.")
    parser.add_argument("--notes", default="", type=str, help="Notes explaining what the experiment does.")
    parser.add_argument("--run_name", default="", type=str, help="Run identifier.")
    parser.add_argument("--tags", default="", type=str, help="Tags for wandb run.")
    # Learning
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Gradient accumulation steps.")
    parser.add_argument("--mixed_precision", default="bf16", type=str, help="Mixed precision type ('fp16' or 'bf16').")
    # Other
    parser.add_argument("--cpu", default=1, type=int, help="Use CPU for training.")
    parser.add_argument("--num_workers", default=0, type=int, help="Number of dataloader workers.")
    
    config_inference = parser.parse_args()
    with open(config_inference.config_model, "r") as f2:
        config_model = (yaml.safe_load(f2))

    if config_inference.sapt_classname in ("no_finetuning", "full_finetuning"):
        step = config_inference.sapt_classname
    else:
        step = config_inference.resume_from_checkpoint.split("/")[-1]
    config_inference.run_name = f"{config_inference.sapt_classname}-{config_model.get('gnn_type')}-seed_{config_inference.seed}-{step}"
    config_inference.tags = f"seed={config_inference.seed},sapt={config_inference.sapt_classname},gnn={config_model.get('gnn_type')},step={step}"

    pprint(vars(config_inference))

    try:
        inference(config_inference, config_model)
        config_inference_filepath = os.path.join(config_inference.project_dir, "config_inference.json")
        with open(config_inference_filepath, "w") as f:
            json.dump(vars(config_inference), f)
    except Exception as e:
        # Print the error message
        print(f"An error occurred: {e}")
        pdb.post_mortem()

if __name__ == "__main__":
    main()