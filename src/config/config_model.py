from dataclasses import dataclass, field

@dataclass
class ConfigModel:
    """
    Dataclass for configuring model parameters with default values.
    This includes settings for the model architecture, sizes of hidden layers, and other hyperparameters.
    """

    # Model Architecture and Hyperparameters
    backbone_hidden_size: int = field(default=768, metadata={"help": "Size of the hidden layers in the backbone model."})
    mlp_hidden_act: str = field(default="silu", metadata={"help": "Activation function for MLP hidden layers."})
    gnn_type: str = field(default="GAT", metadata={"help": "Type of Graph Neural Network."})
    gnn_input_hidden_size: int = field(default=1024, metadata={"help": "Size of the input hidden layer for GNN."})
    gnn_output_hidden_size: int = field(default=768, metadata={"help": "Size of the output hidden layer for GNN."})
    gnn_intermediate_hidden_size: int = field(default=768, metadata={"help": "Size of the intermediate hidden layer for GNN."})
    gnn_attn_heads: int = field(default=8, metadata={"help": "Number of attention heads in GNN."})
    backbone_model_path: str = field(default="Salesforce/codet5p-220m", metadata={"help": "Path to the pretrained backbone model."})
    architecture: str = field(default="encoder-decoder", metadata={"help": "Model architecture type."})
    generation_config: str = field(default="", metadata={"help": "The path to the generation config file."})
    gradient_checkpointing: bool = field(default=True, metadata={"help": "Whether or not to use gradient checkpointing to save gpu memory usage."})
    num_labels: int = field(default=2, metadata={"help": "The number of target classification labels"})
    problem_type: int = field(default="single_label_classification", metadata={"help": "Choose either regression, single_label_classification, or multi_label_classification"})
    layers_to_train: str = field(default="", metadata={"help": "Layer in the backbone model to train"})