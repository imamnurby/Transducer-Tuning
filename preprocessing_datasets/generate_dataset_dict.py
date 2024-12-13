import pandas as pd
import torch
import datasets
from datasets import load_from_disk
from tqdm import tqdm
import argparse

tqdm.pandas()

def get_tensor(path):
    return torch.load(path[0], weights_only=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Process and filter a dataset with splits and tensor loading.")
    
    # Define the command-line arguments
    parser.add_argument("--input_dir", type=str, required=True, help="Directory where the dataset is loaded from.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the final processed dataset will be saved.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file containing split indices.")
    parser.add_argument("--num_proc", type=int, default=4, help="Number of processes to use for mapping, filtering, and loading tensors.")
    parser.add_argument("--output_filename", type=str, default="final_codet5p", help="Name of the final output dataset directory.")

    return parser.parse_args()

def main():
    args = parse_args()

    # Load the dataset from disk
    ds = load_from_disk(args.input_dir)

    # Map the "idx" to the correct format
    ds = ds.map(lambda example: {"idx": example["idx"][0]}, num_proc=args.num_proc)

    # Load indices from the CSV file
    temp_df = pd.read_csv(args.csv_path)
    test_idx = temp_df[temp_df["split"] == "test"]["idx"]
    eval_idx = temp_df[temp_df["split"] == "eval"]["idx"]
    train_idx = temp_df[temp_df["split"] == "train"]["idx"]

    # Filter the dataset based on the split indices
    train = ds.filter(lambda example: example["idx"] in train_idx, num_proc=args.num_proc)
    eval = ds.filter(lambda example: example["idx"] in eval_idx, num_proc=args.num_proc)
    test = ds.filter(lambda example: example["idx"] in test_idx, num_proc=args.num_proc)

    # Map tensors to the train, eval, and test datasets
    train = train.map(lambda example: {"node_features": get_tensor(example["node_features"])}, num_proc=args.num_proc)
    train = train.map(lambda example: {"edge_indices": get_tensor(example["edge_indices"])}, num_proc=args.num_proc)

    eval = eval.map(lambda example: {"node_features": get_tensor(example["node_features"])}, num_proc=args.num_proc)
    eval = eval.map(lambda example: {"edge_indices": get_tensor(example["edge_indices"])}, num_proc=args.num_proc)

    test = test.map(lambda example: {"node_features": get_tensor(example["node_features"])}, num_proc=args.num_proc)
    test = test.map(lambda example: {"edge_indices": get_tensor(example["edge_indices"])}, num_proc=args.num_proc)

    # Ensure "idx" is wrapped in a list
    train = train.map(lambda x: {"idx": [x["idx"]]})
    eval = eval.map(lambda x: {"idx": [x["idx"]]})
    test = test.map(lambda x: {"idx": [x["idx"]]})

    # Create a final dataset dictionary
    final_ds = datasets.DatasetDict({
        "train": train,
        "eval": eval,
        "test": test
    })

    # Correct the shapes of node features and edge indices in the final dataset
    final_ds["train"] = final_ds["train"].map(lambda x: {"node_features_shape": x["node_features_shape"][0]})
    final_ds["train"] = final_ds["train"].map(lambda x: {"edge_indices_shape": x["edge_indices_shape"][0]})

    final_ds["eval"] = final_ds["eval"].map(lambda x: {"node_features_shape": x["node_features_shape"][0]})
    final_ds["eval"] = final_ds["eval"].map(lambda x: {"edge_indices_shape": x["edge_indices_shape"][0]})

    final_ds["test"] = final_ds["test"].map(lambda x: {"node_features_shape": x["node_features_shape"][0]})
    final_ds["test"] = final_ds["test"].map(lambda x: {"edge_indices_shape": x["edge_indices_shape"][0]})

    # Save the final dataset to disk
    final_ds.save_to_disk(args.output_dir)

if __name__ == "__main__":
    main()
