from typing import Optional, Union
from dataclasses import dataclass, field

@dataclass
class ConfigTraining:
    """
    Dataclass for configuring script arguments with default values and descriptions.
    It includes execution settings, paths, and hyperparameters for a deep learning experiment.
    """

    # Execution Settings
    is_debug: bool = field(default=False, metadata={"help": "Enable debug mode."})
    project_name: str = field(default="codet5p", metadata={"help": "Project name."})

    # SAPT Model Choice
    sapt_classname: str = field(default="", metadata={"help": "Which model is used to perform the experiment."})

    # Dataset Config
    dataset_path: Optional[str] = field(default="dataset/summarization/processed_examples", metadata={"help": "Training dataset path."})
    max_seq_len: int = field(default=512, metadata={"help": "Max sequence length for source/target."})
    dataset_portion: Union[str, float] = field(default=1.0, metadata={"help": "how much training data is included"})

    # Output Dir
    project_dir: str = field(default="output_training/codet5p_trial1", metadata={"help": "Experiment logs and output directory."})

    # Checkpointing
    save_total_limit: int = field(default=3, metadata={"help": "Max number of checkpoints to retain."})
    checkpointing_must_save_steps: str = field(default="0,2,4", metadata={"help": "steps to checkpoint. Must be less than or equal save_total_limit"})
    checkpointing_must_save_epochs: str = field(default="0,2,4", metadata={"help": "Epochs to checkpoint. Must be less than or equal save_total_limit"})
    resume_from_checkpoint: Optional[str] = field(default=None, metadata={"help": "Checkpoint path to resume training."})
    
    # Validation Metric
    validation_metric: str = field(default="sidescore", metadata={"help": "The validation metric name used to evaluate the model on the validation dataset"})
    
    # Batch Size
    validation_batch_size: int = field(default=2, metadata={"help": "Training batch size."})
    training_batch_size: int = field(default=2, metadata={"help": "Validation batch size."})

    # Tracking
    with_tracking: bool = field(default=True, metadata={"help": "Enable experiment tracking."})
    resume_tracking: str = field(default=None, metadata={"help": "Specify the id of the wandb run to resume the logging"})
    notes: str = field(default="", metadata={"help": "notes explaining what the experiment does"})
    run_name: str = field(default="trial1", metadata={"help": "Run identifier."})
    tags: str = field(default=None, metadata={"help": "Tags for wandb run"})
    
    # Learning Hyperparameters
    lr: float = field(default=3e-5, metadata={"help": "Learning rate."})
    num_epochs: int = field(default=20, metadata={"help": "Number of training epochs."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max grad norm for clipping."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Gradient accumulation steps."})
    num_warmup_steps: int = field(default=100, metadata={"help": "Warmup steps for learning rate scheduler."})

    # Hardware Settings
    mixed_precision: str = field(default="bf16", metadata={"help": "Mixed precision type ('fp16' or 'bf16'). Requires PyTorch >= 1.10 and Nvidia Ampere GPU for 'bf16'."})
    cpu: bool = field(default=False, metadata={"help": "Use CPU for training."})
    num_workers: int = field(default=4, metadata={"help": "Number of dataloader workers."})
    seed: int = field(default=42, metadata={"help": "Seed for reproducibility."})

    # Early Stop
    early_stop_threshold: int = field(default=5, metadata={"help": "Early stopping threshold level."})