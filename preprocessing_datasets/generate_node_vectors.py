import os
import json
import argparse
import re
from typing import Callable, Tuple, Optional, Any, Union, Dict
from dataclasses import dataclass

import torch
from torch.nn import AvgPool1d
from torch.utils.data import DataLoader

import pandas as pd
import networkx as nx
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PreTrainedTokenizer, DataCollatorWithPadding
from datasets import Dataset, load_from_disk


tqdm.pandas()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser(description="Process and extract features from CSV data using transformers.")
    
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--base_path_dot_files", type=str, required=True, help="Base path for .dot files.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the processed data.")
    parser.add_argument("--node_feature_path", type=str, required=True, help="Path to save the node features.")
    parser.add_argument("--edge_indice_path", type=str, required=True, help="Path to save the edge indices.")
    
    parser.add_argument("--model_path_for_node_feature", type=str, required=True, help="Path to the model used for extracting node features.")
    parser.add_argument("--model_path_for_backbone", type=str, required=True, help="Path to the model used for the backbone.")
    
    parser.add_argument("--max_length", type=int, default=400, help="Maximum token length for the tokenizer.")
    parser.add_argument("--truncation", type=bool, default=True, help="Enable/disable truncation for the tokenizer.")
    parser.add_argument("--start", type=int, default=0, help="Start index for slicing the DataFrame.")
    parser.add_argument("--end", type=int, default=None, help="End index for slicing the DataFrame.")
    parser.add_argument("--extract_node_features", type=bool, default=True, help="Flag to extract node features.")
    
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for processing.")
    parser.add_argument("--node_embedding_size", type=int, default=1024, help="Size of the node embeddings.")
    
    return parser.parse_args()



def normalize_path(path: str)->str:
    if path.startswith("./"):
        return path.lstrip("./")
    return path

def check_valid_instances(basepath: str, path: str)->bool:
    filepath = os.path.join(basepath, path)
    return os.path.exists(filepath)

def map_splits_to_ids(split: str)->int:
    if split == "train":
        return 0
    elif split == "test" or split == "eval":
        return 1
    else:
        return 2

def generate_node_features(
        G: nx.Graph, 
        model: Optional[PreTrainedModel] = None, 
        tokenizer: Optional[PreTrainedTokenizer] = None, 
        batch_size: int = 512, 
        num_workers: int = 0,
        node_embedding_size: Optional[int] = None,
        data_collator: Optional[Any] = None,
        ) -> torch.Tensor:
    """
    Generates embeddings for each node in the given graph. If a model is provided, it uses the model to generate embeddings.
    Otherwise, it generates random embeddings.

    Parameters:
    - G (nx.Graph): The graph for which node embeddings are to be generated.
    - node_embedding_size (Optional[int]): The size of each node's embedding vector. Required if no model is provided.
    - model (Optional[PreTrainedModel]): A pretrained model for generating embeddings.
    - tokenizer (Optional[PreTrainedTokenizer]): A tokenizer for preprocessing node labels.
    - batch_size (int): Batch size for model inference.
    - num_workers (int): Number of workers for DataLoader.

    Returns:
    - torch.Tensor: A tensor containing the embeddings for all nodes in the graph,
      where each row corresponds to a node's embedding.
    """

    if model is None:
        if node_embedding_size is None:
            raise ValueError("embedding_size must be provided if no model is specified.")
        number_of_nodes = len(G.nodes())
        node_features = torch.rand((number_of_nodes, node_embedding_size), dtype=torch.float32)
        return node_features

    # Extracting and preprocessing node labels
    node_contents = []
    for node in G.nodes():
        label = G.nodes[node].get('label', '')
        node_contents.append(label)

    # Tokenizing node labels
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided if a model is specified.")
    tokenized_contents = tokenizer(node_contents, truncation=True)

    # Creating a DataLoader
    ds = Dataset.from_dict(tokenized_contents)
    dataloader = DataLoader(ds, collate_fn=data_collator, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # Generating embeddings
    node_features = []
    for batch in dataloader:
        batch = batch.to(model.device)
        with torch.no_grad():
            out = model(**batch)
        out = out.last_hidden_state
        embeddings = out[:, 0].detach().cpu()

        # Keep only the first k elements in the hidden dimension
        embeddings = embeddings[:, :node_embedding_size]

        node_features.append(embeddings)

    # Concatenating all embeddings into a single tensor
    return torch.cat(node_features, dim=0)

def extract_graph_features_and_edges(
        dot_file_path: str, 
        feature_generator: Callable,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 512,
        node_embedding_size: int = 512,
        num_workers: int = 0,
        data_collator: Optional[Any] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    """
    Extracts node features and edge indices from a DOT file using a specified feature generation function.

    This function reads a DOT file to create a graph using NetworkX and PyDot. It then uses the 
    provided feature_generator, which utilizes a specified model and tokenizer, to generate features for 
    each node in the graph. The function also computes the edge indices based on the graph structure.

    Parameters:
    - dot_file_path (str): The path to the DOT file representing the graph.
    - feature_generator (Callable): A function that generates features for each node in the graph. 
      It takes the graph, a pre-trained model, and a tokenizer as inputs.
    - model (PreTrainedModel): A pre-trained model used in the feature generation process.
    - tokenizer (PreTrainedTokenizer): A tokenizer used in conjunction with the pre-trained model 
      for feature generation.
    - num_workers (int): The number of workers for the dataloader

    Returns:
    - Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], Tuple[int, int]]: A tuple containing:
        - A torch.Tensor of the node features.
        - A torch.Tensor of the edge indices.
        - A tuple representing the original shape of the node features tensor.
        - A tuple representing the original shape of the edge indices tensor.
    """

    # Read the DOT file and create a graph
    G = nx.drawing.nx_pydot.read_dot(dot_file_path)

    node_features = feature_generator(
        G=G,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        node_embedding_size=node_embedding_size,
        num_workers=num_workers,
        data_collator=data_collator
    )

    # Node mapping to ensure indices in edge_index are correct
    node_mapping = {node: i for i, node in enumerate(G.nodes())}

    # Iterate over the edges to create the edge_index list
    source_nodes = []
    target_nodes = []
    for edge in G.edges():
        source_nodes.append(node_mapping[edge[0]])
        target_nodes.append(node_mapping[edge[1]])
    
    # Convert to tensors
    source_nodes_tensor = torch.tensor(source_nodes, dtype=torch.long)
    target_nodes_tensor = torch.tensor(target_nodes, dtype=torch.long)

    edge_indices = torch.stack([source_nodes_tensor, target_nodes_tensor], dim=0)

    # Flatten the output tensor and keep the original dimension. This original
    # dimension is used to revert back the shape later
    original_features_shape = node_features.shape
    node_features = node_features.view(-1)

    original_edge_indices_shape = edge_indices.shape
    edge_indices = edge_indices.view(-1)

    return node_features, edge_indices, original_features_shape, original_edge_indices_shape

def main():
    args = parse_args()

    model_for_node_feature = AutoModel.from_pretrained(args.model_path_for_node_feature)
    tokenizer_for_node_feature = AutoTokenizer.from_pretrained(args.model_path_for_node_feature, truncation=True)
    data_collator = DataCollatorWithPadding(tokenizer_for_node_feature)
    tokenizer_backbone = AutoTokenizer.from_pretrained(args.model_path_for_backbone)

    model_for_node_feature.to("cuda")

    os.makedirs(args.node_feature_path, exist_ok=True)
    os.makedirs(args.edge_indice_path, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)
    
    # Save configuration as JSON
    config_filepath = os.path.join(args.output_path, "config_preprocessing.json")
    with open(config_filepath, "w") as f:
        json.dump(vars(args), f, indent=4)

    df = pd.read_csv(args.csv_path, index_col="idx", lineterminator="\n")
    df["source"] = df["source"].progress_apply(lambda x: x.strip("\n"))
    df["target"] = df["target"].progress_apply(lambda x: x.strip("\n"))
    df = df[args.start:args.end].copy() if args.end else df[args.start:].copy()

    df["dot"] = df["dot"].progress_apply(lambda x: normalize_path(x))
    df["is_exist"] = df["dot"].progress_apply(lambda x: check_valid_instances(args.base_path_dot_files, x))
    df = df[df["is_exist"] == True].copy()
    df["split"] = df["split"].progress_apply(lambda x: map_splits_to_ids(x))

    if args.extract_node_features:
        def extract_graph_features(idx, dot_path):
            dot_path = os.path.join(args.base_path_dot_files, dot_path)
            try:
                feature, edge, feature_shape, edge_shape = extract_graph_features_and_edges(
                    dot_path, generate_node_features, model_for_node_feature, tokenizer_for_node_feature, 
                    args.batch_size, args.node_embedding_size, args.num_workers, data_collator)
                print(dot_path, " success!")
            except Exception as e:
                feature = edge = feature_shape = edge_shape = torch.tensor([-99])
                print(dot_path, "fail", str(e))

            temp_output_filepath_node = os.path.join(args.node_feature_path, f"{idx}.pt")
            torch.save(feature, temp_output_filepath_node)

            temp_output_filepath_edge = os.path.join(args.edge_indice_path, f"{idx}.pt")
            torch.save(edge, temp_output_filepath_edge)
            return temp_output_filepath_node, temp_output_filepath_edge, feature_shape, edge_shape

        df["temp_output"] = df.progress_apply(lambda x: extract_graph_features(x.name, x["dot"]), axis=1)
        df["node_features"] = df["temp_output"].progress_apply(lambda x: [x[0]])
        df["edge_indices"] = df["temp_output"].progress_apply(lambda x: [x[1]])
        df["node_features_shape"] = df["temp_output"].progress_apply(lambda x: [x[2]])
        df["edge_indices_shape"] = df["temp_output"].progress_apply(lambda x: [x[3]])

        df.drop(columns="temp_output", inplace=True)

    def prepare_examples(examples: Dict) -> Dict:
        model_inputs = tokenizer_backbone(examples["source"], truncation=True)
        targets = tokenizer_backbone(examples["target"], truncation=True)
        model_inputs["labels"] = targets["input_ids"]
        model_inputs["split"] = [[x] for x in examples["split"]]
        model_inputs["idx"] = [[x] for x in examples["idx"]]
        return model_inputs

    ds = Dataset.from_pandas(df)
    ds = ds.map(prepare_examples, batched=True, batch_size=args.batch_size)
    ds = ds.remove_columns(column_names=["source", "target", "dot", "is_exist"])

    ds.save_to_disk(args.output_path)
    load_from_disk(args.output_path)

if __name__ == "__main__":
    main()
