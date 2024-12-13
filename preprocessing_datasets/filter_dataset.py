import argparse
from datasets import load_from_disk

def parse_args():
    parser = argparse.ArgumentParser(description="Filter and save a dataset using the datasets library.")
    
    parser.add_argument("--input_dir", type=str, required=True, help="Directory where the dataset is loaded from.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the filtered dataset will be saved.")
    parser.add_argument("--max_input_length", type=int, default=400, help="Maximum length of input_ids for filtering.")
    parser.add_argument("--max_node_features", type=int, default=50, help="Maximum node_features_shape[0] for filtering.")
    parser.add_argument("--num_proc", type=int, default=10, help="Number of processes to use for filtering.")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Load the dataset from the specified directory
    print(f"Loading dataset from {args.input_dir}...")
    ds = load_from_disk(args.input_dir)
    
    # Filter the dataset based on input length
    print(f"Filtering dataset where 'input_ids' length is less than {args.max_input_length}...")
    ds = ds.filter(lambda x: len(x["input_ids"]) < args.max_input_length, num_proc=args.num_proc)
    
    # Display the filtered dataset
    print("Dataset after filtering on 'input_ids' length:")
    print(ds)
    
    # Filter the dataset based on node features shape
    print(f"Filtering dataset where 'node_features_shape[0]' is less than or equal to {args.max_node_features}...")
    ds = ds.filter(lambda x: x["node_features_shape"][0] <= args.max_node_features, num_proc=args.num_proc)
    
    # Display the final dataset
    print("Final filtered dataset:")
    print(ds)
    
    # Save the filtered dataset to the specified output directory
    print(f"Saving filtered dataset to {args.output_dir}...")
    ds.save_to_disk(args.output_dir)
    print("Dataset saved successfully.")

if __name__ == "__main__":
    main()
