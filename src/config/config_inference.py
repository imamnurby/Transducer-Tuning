from dataclasses import field, dataclass
from typing import Optional

@dataclass
class ConfigInference:
    """
    Dataclass for configuring evaluation script arguments with default values and descriptions.
    """

    # Execution Settings
    is_debug: bool = field(default=False, metadata={"help": "Enable debug mode."})

    # SAPT Model Choice
    sapt_classname: str = field(default="", metadata={"help": "Which model is used to perform the experiment."})

    # Dataset Config
    dataset_path: Optional[str] = field(default="dataset/summarization/processed_examples", metadata={"help": "Training dataset path."})
    max_seq_len: int = field(default=512, metadata={"help": "Max sequence length for source/target."})

    # Output
    project_dir: str = field(default="output_training/codet5p_trial1", metadata={"help": "Experiment logs and output directory."})
    save_output: bool = field(default=False, metadata={"help": "Whether or not to save the inference result"})

    # Checkpointing
    save_total_limit: int = field(default=1, metadata={"help": "Max number of checkpoints to retain."})
    resume_from_checkpoint: Optional[str] = field(default=None, metadata={"help": "Checkpoint path to resume training."})
    
    # Batch Size
    batch_size: int = field(default=2, metadata={"help": "Training batch size."})

    # Tracking
    with_tracking: bool = field(default=True, metadata={"help": "Enable experiment tracking."})
    
    # Learning Hyperparameters
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Gradient accumulation steps."})

    # Hardware Settings
    mixed_precision: str = field(default="bf16", metadata={"help": "Mixed precision type ('fp16' or 'bf16'). Requires PyTorch >= 1.10 and Nvidia Ampere GPU for 'bf16'."})
    cpu: bool = field(default=False, metadata={"help": "Use CPU for training."})
    num_workers: int = field(default=0, metadata={"help": "Number of dataloader workers."})
    seed: int = field(default=43, metadata={"help": "Seed for reproducibility."})





