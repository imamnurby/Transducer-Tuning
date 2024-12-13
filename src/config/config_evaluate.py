from dataclasses import dataclass, field

@dataclass
class ConfigEvaluate:
    """
    Dataclass for configuring script arguments with default values and descriptions.
    """

    # Execution Settings
    project_name: str = field(default="codet5p", metadata={"help": "Project name."})

    # Output Dir
    project_dir: str = field(default="output_training/codet5p_trial1", metadata={"help": "Experiment logs and output directory."})
    
    # Validation Metric
    metrics: str = field(default="sidescore", metadata={"help": "The validation metric name used to evaluate the model on the validation dataset"})
    
    # Batch Size
    batch_size: int = field(default=2, metadata={"help": "Training batch size."})

    # Tracking
    with_tracking: bool = field(default=True, metadata={"help": "Enable experiment tracking."})
    resume_tracking: str = field(default=None, metadata={"help": "Specify the id of the wandb run to resume the logging"})
    notes: str = field(default="", metadata={"help": "notes explaining what the experiment does"})
    run_name: str = field(default="trial1", metadata={"help": "Run identifier."})
    tags: str = field(default=None, metadata={"help": "Tags for wandb run"})

    # Path to generation config
    generation_config: str = field(default="", metadata={"help": "Path to the generation config"})
    