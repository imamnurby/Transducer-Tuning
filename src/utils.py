from datetime import datetime
import os
import pandas as pd
from pprint import pprint
import shutil
import torch
from tqdm import tqdm
import wandb
import yaml
from peft import get_peft_model, LoraConfig, TaskType, PrefixTuningConfig, PromptTuningConfig, PromptTuningInit
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from config import ConfigModel, ConfigTraining, ConfigInference
from datasets import load_from_disk
from modelling import (
    SAPT_MODEL_MAPPING,
    VALID_BASELINES,
    SAPTConfig,
    DataCollatorForSeq2SeqWithGraph,
    DataCollatorForSeq2SeqWithoutGraph,
    DataCollatorForClassification
)
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GenerationConfig,
    HfArgumentParser,
    LogitsProcessor,
    LogitsProcessorList,
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers.generation import LogitsProcessor, LogitsProcessorList
from typing import Dict, Tuple, Union, Any
from pathlib import Path
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

def get_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used//1024**2

def maybe_delete_checkpoints(checkpoint_directory: str, save_total_limit: int, accelerator: Accelerator) -> None:
    """
    Deletes the directory with the smallest numbered name if the total number of directories
    exceeds the specified limit.

    Args:
        checkpoint_directory (str): The path to the directory containing checkpoint subdirectories.
        save_total_limit (int): The maximum number of checkpoint directories to retain.

    Notes:
        Assumes that the subdirectory names follow the format "step_{number}".
    """
    ckpt_dirs = [os.path.join(checkpoint_directory, subdir) for subdir in os.listdir(checkpoint_directory)]
    ckpt_dirs = [(x, int(x.split("_")[-1])) for x in ckpt_dirs]
    if len(ckpt_dirs) > save_total_limit:
        ckpt_dirs.sort(key=lambda x: x[1])
        accelerator.print(f"Deleting checkpoint: {ckpt_dirs[0][0]}")
        shutil.rmtree(ckpt_dirs[0][0])

def prepare_dataloader(
    dataset_path: str,
    split: str,
    data_collator: Union[DataCollatorForLanguageModeling, DataCollatorForSeq2SeqWithGraph, DataCollatorForSeq2SeqWithoutGraph],
    batch_size: int,
    dataset_portion: Union[float, int],
    num_workers: int,
    pin_memory: bool,
    is_debug: bool,
    seed: int
) -> DataLoader:
    """
    Loads and prepares a DataLoader for a specific split of a dataset based on the given configurations.

    Args:
        dataset_path (str): Path to the dataset to load.
        split (str): Dataset split to load (e.g., 'train', 'test').
        data_collator (Union[DataCollatorForLanguageModeling, DataCollatorForSeq2SeqWithGraph, DataCollatorForSeq2SeqWithoutGraph]):
        Function to collate data samples into batches.
        batch_size (int): Number of samples per batch.
        dataset_portion (Union[float, int]): Portion of the dataset to load, can be a percentage less than 1 or an absolute number greater than 1.
        num_workers (int): Number of subprocesses to use for data loading.
        pin_memory (bool): If True, the data loader will copy tensors into CUDA pinned memory before returning them.
        is_debug (bool): Flag to indicate whether to run in debug mode (limits data to the batch size).
        seed (int): Seed value for shuffling the dataset.

    Returns:
        DataLoader: DataLoader configured for the specified dataset split.
    """
    print("#"*50)
    print(f"Loading Dataset from {dataset_path}")
    assert split in ("train", "eval", "test")

    ds = load_from_disk(dataset_path)

    if is_debug:
        target_range = min(batch_size, len(ds[split]))
    elif 0 < dataset_portion < 1:
        target_range = int(dataset_portion * len(ds[split]))
    elif dataset_portion > 1:
        target_range = int(min(dataset_portion, len(ds[split])))
    else:
        target_range = len(ds[split])

    if target_range < len(ds[split]):
        print(f"Using a subset of {target_range} samples from the dataset.")
        dataset_subset = ds[split].shuffle(seed=seed).select(range(target_range))
    else:
        print("Using the full dataset.")
        dataset_subset = ds[split]

    dataloader = DataLoader(
        dataset_subset,
        shuffle=True if split=="train" else False,
        collate_fn=data_collator,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return dataloader

def save_checkpoint(
    args: HfArgumentParser,
    accelerator: Accelerator,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    is_step: str,
    step_count: int,
    save_total_limit: int,
)->None:
    """
    Save the model (transformers compatible), tokenizer, and accelerate state. The accelerate.save_state handles the case 
    where the checkpoint folder count is larger than checkpoint total limit
    """
    if accelerator.is_main_process:
        if is_step == "last":
            base_checkpoint_directory = f"{args.project_dir}/checkpoints_latest"
        elif is_step == "targeted_steps":
            base_checkpoint_directory = f"{args.project_dir}/checkpoints_targeted_steps"
        elif is_step == "targeted_epochs":
            base_checkpoint_directory = f"{args.project_dir}/checkpoints_targeted_epochs"
        else:
            raise ValueError("Invalid is_step value")
        os.makedirs(base_checkpoint_directory, exist_ok=True)
        maybe_delete_checkpoints(base_checkpoint_directory, save_total_limit, accelerator)

        checkpoint_directory = f"step_{str(step_count)}" if is_step=="targeted_steps" else f"epoch_{str(step_count)}"
        checkpoint_directory = os.path.join(base_checkpoint_directory, checkpoint_directory)
        
        accelerator.save_state(output_dir=checkpoint_directory, safe_serialization=False)

        accelerator.print(f"Saving accelerator state to {checkpoint_directory}")
        
        # save tokenizer
        tokenizer.save_pretrained(checkpoint_directory)
        # save hyperparam setting
        torch.save(args, os.path.join(checkpoint_directory, "hyperparam.config"))
    
        # Save a trained model and configuration using `save_pretrained()`.
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_directory=checkpoint_directory,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            safe_serialization=False
        )

        accelerator.print(f"Saving model checkpoint, tokenizer, and hyperparam config to {checkpoint_directory}")

def handle_checkpoint_resume(args: HfArgumentParser, accelerator: Accelerator, train_dataloader:DataLoader) -> Tuple[int, int]:
    if args.resume_from_checkpoint != "":
        accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        path = os.path.basename(args.resume_from_checkpoint)
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = 0
            return resume_step, starting_epoch
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
        return resume_step, starting_epoch
    
    else:
        raise ValueError("Loading from the checkpoint is error. Check the specifed path again.") 

def load_model_tokenizer_datacollator(config_model: ConfigModel, config_training: ConfigTraining) -> Tuple[PreTrainedTokenizer, PreTrainedModel, Union[DataCollatorForSeq2SeqWithGraph, DataCollatorForLanguageModeling]]:
    """
    Load the model, tokenizer, and data collator based on the provided arguments.
    """
    print("#"*50)
    if config_training.sapt_classname in SAPT_MODEL_MAPPING:
        del config_model["num_virtual_tokens"]
        config_model = SAPTConfig(**config_model)
        print(f"Loading the SAPT model. The backbone model checkpoint is {config_model.backbone_model_path} and the SAPT classname is {config_training.sapt_classname}.")
        assert(config_training.sapt_classname in SAPT_MODEL_MAPPING)
        model_class = SAPT_MODEL_MAPPING[config_training.sapt_classname]
        model = model_class(config_model)
    elif config_training.sapt_classname in VALID_BASELINES:
        num_virtual_tokens = config_model["num_virtual_tokens"]
        del config_model["num_virtual_tokens"]
        config_model = ConfigModel(**config_model)
        print(f"Loading the original model without SAPT. The model checkpoint is {config_model.backbone_model_path}")
        if config_model.architecture == "encoder-decoder":
            model_class = AutoModelForSeq2SeqLM
        elif config_model.architecture == "decoder":
            model_class = AutoModelForCausalLM
        elif config_model.architecture == "encoder":
            model_class = AutoModelForSequenceClassification
        else:
            raise ValueError("You specify sapt_classname in config_training to be None and config_model.architecture is invalid. The valid values for config_model.architecture are 'encoder-decoder', 'decoder'.")

        if config_model.architecture != "encoder":
            print("Loading model without the num_labels argument")
            model = model_class.from_pretrained(config_model.backbone_model_path)
        else:
            print("Loading model with the num_labels argument")
            model = model_class.from_pretrained(config_model.backbone_model_path, num_labels=config_model.num_labels, problem_type="single_label_classification")
        
        if config_training.sapt_classname == "lora":
            print("Loading with LORA!")
            peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=num_virtual_tokens, lora_alpha=32, lora_dropout=0.1)
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        elif config_training.sapt_classname == "prefix-tuning":
            print("Loading with Prefix-Tuning!")
            peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=num_virtual_tokens)
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        elif config_training.sapt_classname == "prompt-tuning":
            print("Loading with Prompt-Tuning!")
            peft_config = PromptTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=num_virtual_tokens)
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        
        if config_model.gradient_checkpointing and config_training.sapt_classname not in VALID_BASELINES:
            print("Gradient Checkpointing is enabled!")
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={})
    else:
        raise ValueError("Wrong sapt_classname!")
    
    tokenizer = AutoTokenizer.from_pretrained(config_model.backbone_model_path)
    assert hasattr(tokenizer, "pad_token_id")
    assert hasattr(tokenizer, "bos_token_id")
    assert hasattr(tokenizer, "eos_token_id")
    print(f"The pad_token (pad_token_id) is {tokenizer.pad_token} ({tokenizer.pad_token_id})")
    print(f"The bos_token (bos_token_id) is {tokenizer.bos_token} ({tokenizer.bos_token_id})")
    print(f"The eos_token (eos_token_id) is {tokenizer.eos_token} ({tokenizer.eos_token_id})")

    if config_training.config_generation:
        with open(config_training.config_generation, "r") as f:
            generation_config = yaml.safe_load(f)

        generation_config["pad_token_id"] = tokenizer.pad_token_id
        generation_config["bos_token_id"] = tokenizer.bos_token_id
        generation_config["eos_token_id"] = tokenizer.eos_token_id

        print(f"Initializing pad_token (pad_token_id) in the generation_config to: {tokenizer.pad_token} ({tokenizer.pad_token_id})")
        print(f"Initializing bos_token (bos_token_id) in the generation_config to: {tokenizer.bos_token} ({tokenizer.bos_token_id})")
        print(f"Initializing eos_token (eos_token_id) in the generation_config to: {tokenizer.eos_token} ({tokenizer.eos_token_id})")
        
    else:
        generation_config = None

    if config_model.architecture == "encoder-decoder":     
        if config_training.sapt_classname in SAPT_MODEL_MAPPING:
            data_collator_class = DataCollatorForSeq2SeqWithGraph
            config = model.backbone_model.config
        else:
            data_collator_class = DataCollatorForSeq2SeqWithoutGraph
            config = model.config
        data_collator = data_collator_class(tokenizer, model=model, max_length=config_training.max_seq_len)
        
        assert hasattr(config, "decoder_start_token_id")
        generation_config["decoder_start_token_id"] = config.decoder_start_token_id
        print(f"Initializing decoder_start_token_id to: {config.decoder_start_token_id}")

        model.config.is_encoder_decoder = True
    
    elif config_model.architecture == "encoder":
        print("Loading data collator for a classification task!")
        using_graph = True if config_training.sapt_classname in SAPT_MODEL_MAPPING else False
        data_collator = DataCollatorForClassification(
            tokenizer,
            max_length=config_training.max_seq_len,
            using_graph=using_graph
        )

    elif config_model.architecture == "decoder":
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    else:
        raise ValueError("Invalid Architecture")
    
    if config_training.config_generation:
        generation_config = GenerationConfig(**generation_config)
        print("Loading generation config")
        pprint(generation_config)
    print("#"*50)
    return tokenizer, model, data_collator, generation_config

def initialize_accelerator(config_training:ConfigTraining, config_model:ConfigModel)->Accelerator:
    """
    Initialize the Accelerator with or without tracking based on arguments.
    """
    def append_date_and_time(input_str):
        # Get the current date in yyyymmdd format
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Append '_time' as a placeholder for the current time
        # If you need an actual time, replace '_time' with datetime.now().strftime("_%H%M%S")
        return f"{input_str}-{current_time}"
    
    config = ProjectConfiguration()
    config.project_dir = config_training.project_dir
    config.total_limit = config_training.save_total_limit
    config.automatic_checkpoint_naming = False
    
    if config_training.is_debug:
        config_training.with_tracking = False 

    wandb_config = vars(config_training)
    wandb_config.update(config_model)
    print("#"*50)
    print("Model Configuration and Training Setting")
    pprint(wandb_config, sort_dicts=False)
    print("#"*50)
    
    if config_training.with_tracking:
        accelerator = Accelerator(
            cpu=True if config_training.cpu==1 else False, 
            mixed_precision=config_training.mixed_precision, 
            log_with="wandb",
            gradient_accumulation_steps=config_training.gradient_accumulation_steps,
            project_config=config
        )
        
        wandb_init_kwargs = {
            "id": config_training.resume_tracking if config_training.resume_tracking else wandb.util.generate_id(),
            "notes": config_training.notes if config_training.notes else "",
            "resume": "allow",
            "tags": [tag for tag in config_training.tags.split(",") if tag != ""] if config_training.tags else [],
        }

        accelerator.init_trackers(config_training.project_name, config=wandb_config, init_kwargs={"wandb": wandb_init_kwargs})
        accelerator.trackers[0].run.name = append_date_and_time(config_training.run_name)
    else:
        accelerator = Accelerator(
            cpu=True if config_training.cpu==1 else False, 
            mixed_precision=config_training.mixed_precision,
            gradient_accumulation_steps=config_training.gradient_accumulation_steps,
            project_config=config
        )
    accelerator.free_memory()
    return accelerator    
    
def do_inference_generative(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_dataloader: DataLoader,
    accelerator: Accelerator,
    generation_config: GenerationConfig,
    output_filepath: str = None,
    use_sapt: bool = True,
    use_stopping_criteria = True,
    use_logit_processor: bool = False
)->Dict:
    accelerator.print("#"*50)
    accelerator.print("Performing inference!")
    if use_stopping_criteria:
        class StoppingCriteriaSub(StoppingCriteria):
            def __init__(self, eos_token_id):
                super().__init__()
                self.eos_token_id = eos_token_id

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
                return (input_ids[:, -1] == self.eos_token_id).all().item()
        
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(tokenizer.eos_token_id)])
        accelerator.print(f"Use stopping criteria: {stopping_criteria[0]}")
    else:
        stopping_criteria = None
        accelerator.print(f"Not using stopping criteria")
    
    if use_logit_processor:
        class PaddingAfterEOSTokenLogitsProcessor(LogitsProcessor):
            def __init__(self, eos_token_id: int, pad_token_id: int):
                self.eos_token_id = eos_token_id
                self.pad_token_id = pad_token_id

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                # Check for sequences that have already generated an EOS token
                eos_generated_mask = input_ids == self.eos_token_id

                # Find sequences where EOS token has been generated
                eos_in_sequence = eos_generated_mask.any(dim=1)
                # For those sequences, set the score of the EOS token to a very high value
                for i, has_eos in enumerate(eos_in_sequence):
                    if has_eos:
                        scores[i, :] = -float("inf")
                        scores[i, self.eos_token_id] = float("inf")  # Ensure EOS token has the highest score
                return scores
        logits_processor = LogitsProcessorList()
        logits_processor.append(PaddingAfterEOSTokenLogitsProcessor(tokenizer.eos_token_id, tokenizer.pad_token_id))
        accelerator.print(f"Use logits_processor: {type(logits_processor[0])}")
    else:
        logits_processor = None
        accelerator.print(f"Not using logits processor")

    model.eval()
    progress_bar = tqdm(eval_dataloader, disable=not accelerator.is_local_main_process)
    output_dict = {
        "idx": [],
        "input_ids": [],
        "labels": [],
        "raw_preds": []
    }
    loss_per_steps = []
    for step, batch in enumerate(progress_bar):
        batch.to(accelerator.device)
        if use_sapt:
            with torch.no_grad():
                batch["encoder_outputs"] = model.get_encoder_output(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    node_features=batch["node_features"],
                    edge_indices=batch["edge_indices"],
                    node_features_shape=batch["node_features_shape"],
                    edge_indices_shape=batch["edge_indices_shape"]
                )
        # compute the loss on the validation
        if use_sapt:
            temp_outputs = model(
                input_ids=batch["input_ids"], 
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                node_features=batch["node_features"],
                edge_indices=batch["edge_indices"],
                node_features_shape=batch["node_features_shape"],
                edge_indices_shape=batch["edge_indices_shape"],
            )
        else:
            temp_outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )        
        loss = temp_outputs.loss
        # if gpu is more than 1, then get the average
        if accelerator.num_processes > 1:
            loss = loss.mean()
        loss = loss.detach()
        loss_per_steps.append(loss)

        decoded_input_ids = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
        output_dict["input_ids"] += decoded_input_ids

        batch["labels"][batch["labels"] == -100] = tokenizer.pad_token_id
        decoded_labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
        output_dict["labels"] += decoded_labels

        idx = [x[0].item() for x in batch["idx"]]
        output_dict["idx"] += idx

        cols_to_remove = ["split", "idx", "labels"]
        if use_sapt:
            cols_to_remove += ["input_ids", "attention_mask", "node_features", "edge_indices", "node_features_shape", "edge_indices_shape"]
        for key in cols_to_remove:
            del batch[key]

        with torch.no_grad():
            if ((use_sapt
                and hasattr(model.backbone_model.config, "architectures") 
                and model.backbone_model.config.architectures 
                and model.backbone_model.config.architectures[0]  == "PLBartForConditionalGeneration") 
                or ((hasattr(model.config, "architectures") 
                and model.config.architectures 
                and model.config.architectures[0]  == "PLBartForConditionalGeneration")
            )): # hardcoded for PLBART
                preds = model.generate(
                    **batch,
                    generation_config=generation_config,
                    stopping_criteria=stopping_criteria,
                    decoder_start_token_id=2,
                    logits_processor=logits_processor
                )
            else:        
                preds = model.generate(
                    **batch,
                    generation_config=generation_config,
                    stopping_criteria=stopping_criteria,
                    logits_processor=logits_processor
                )

        raw_preds = tokenizer.batch_decode(preds, skip_special_tokens=False)
        output_dict["raw_preds"] += raw_preds

    if output_filepath:
        df = pd.DataFrame(output_dict)
        df.to_csv(output_filepath, index_label="idx")
        accelerator.print(f"Saving prediction to: {output_filepath}")

        parent_path = Path(output_filepath).parent
        output_filepath = os.path.join(parent_path, "loss.txt")
        with open(output_filepath, "w") as f:
            for loss in loss_per_steps: 
                f.write(f"{loss}\n")

    accelerator.print("Finish!")
    accelerator.print("#"*50)
    return output_dict

def do_inference_classification(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_dataloader: DataLoader,
    accelerator: Accelerator,
    output_filepath: str = None,
    use_sapt: bool = True,
)->Dict:
    accelerator.print("#"*50)
    accelerator.print("Performing inference!")
    model.eval()
    progress_bar = tqdm(eval_dataloader, disable=not accelerator.is_local_main_process)
    output_dict = {
        "idx": [],
        "input_ids": [],
        "labels": [],
        "raw_preds": []
    }
    for step, batch in enumerate(progress_bar):
        batch.to(accelerator.device)
    
        decoded_input_ids = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
        output_dict["input_ids"] += decoded_input_ids
        output_dict["labels"] += batch["labels"].tolist()

        idx = [x[0].item() for x in batch["idx"]]
        output_dict["idx"] += idx

        cols_to_remove = ["split", "idx", "labels"]
        for key in cols_to_remove:
            del batch[key]

        with torch.no_grad():
            model_output = model(**batch)

        logits = model_output.logits
        raw_preds = logits.argmax(dim=1)
        raw_preds = raw_preds.tolist()  
        output_dict["raw_preds"] += raw_preds
    
    if output_filepath:
        df = pd.DataFrame(output_dict)
        df.to_csv(output_filepath, index_label="idx")
        accelerator.print(f"Saving prediction to: {output_filepath}")
    accelerator.print("Finish!")
    accelerator.print("#"*50)
    return output_dict

def compute_trainable_params(config_, model):
    if config_.sapt_classname in ("prefix-tuning", "prompt-tuning", "lora"):
        trainable_params, all_param = model.get_nb_trainable_parameters()
    elif config_.sapt_classname in SAPT_MODEL_MAPPING:
        all_param, backbone_param_count = model.get_trainable_params()
        trainable_params = all_param - backbone_param_count
    elif config_.sapt_classname == "full-finetuning":
        trainable_params = 0
        for name, param in model.named_parameters():
            trainable_params += param.numel()
    else:
        trainable_params = 0
    return trainable_params