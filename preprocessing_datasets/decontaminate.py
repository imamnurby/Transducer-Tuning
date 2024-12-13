import pandas as pd
import datasets
from transformers import AutoTokenizer, PreTrainedTokenizer
from datasketch import MinHash, MinHashLSH
from typing import Set, Tuple, Dict
import argparse

def load_and_process_dataset(dataset_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load dataset from disk and process it into pandas DataFrames."""
    print(f"Loading dataset from {dataset_path}...")
    ds = datasets.load_from_disk(dataset_path)
    
    print("Removing 'node_features' column...")
    ds = ds.remove_columns("node_features")

    print("Converting dataset splits to pandas DataFrames...")
    train = ds["train"].to_pandas()
    val = ds["eval"].to_pandas()
    test = ds["test"].to_pandas()

    return train, val, test

def decode_input_ids(df: pd.DataFrame, tokenizer: PreTrainedTokenizer) -> None:
    """Decode the input_ids back into text."""
    print("Decoding 'input_ids' to text...")
    df["input_ids"] = df["input_ids"].apply(lambda x: tokenizer.decode(x, remove_special_tokens=True))

def tokenize(text: str) -> Set[str]:
    """Simple tokenizer that splits by whitespace."""
    return set(text.lower().split())

def get_minhash(text: str, num_perm: int = 128) -> MinHash:
    """Create a MinHash object for a given text."""
    tokens = tokenize(text)
    m = MinHash(num_perm=num_perm)
    for token in tokens:
        m.update(token.encode('utf8'))
    return m

def build_lsh(train_df: pd.DataFrame, num_perm: int = 128, threshold: float = 0.8) -> Tuple[MinHashLSH, Dict[str, MinHash]]:
    """Build LSH for the training set."""
    print("Building MinHashLSH index...")
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    train_minhashes = {}

    for idx, row in train_df.iterrows():
        m = get_minhash(row['input_ids'])
        lsh.insert(f"train_{idx}", m)
        train_minhashes[f"train_{idx}"] = m

    return lsh, train_minhashes

def find_near_duplicates(df: pd.DataFrame, lsh_index: MinHashLSH, prefix: str) -> Set[int]:
    """Find near duplicates in the DataFrame using the LSH index."""
    print(f"Finding near duplicates with prefix '{prefix}'...")
    duplicates = set()
    for idx, row in df.iterrows():
        m = get_minhash(row['input_ids'])
        result = lsh_index.query(m)
        if any(res.startswith(prefix) for res in result):
            duplicates.add(idx)
    return duplicates

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process dataset with MinHash LSH to remove near duplicates.")
    
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--output_csv", type=str, default="included_ids.csv", help="Path to save the cleaned test indices.")
    parser.add_argument("--tokenizer_name", type=str, default="Salesforce/codet5p-220m", help="Name of the tokenizer to use for decoding input_ids.")
    parser.add_argument("--num_perm", type=int, default=128, help="Number of permutations for MinHash.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Threshold for LSH to consider two items as duplicates.")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Load and process the dataset
    train, val, test = load_and_process_dataset(args.dataset_path)
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    # Decode 'input_ids' to text for all splits
    decode_input_ids(train, tokenizer)
    decode_input_ids(val, tokenizer)
    decode_input_ids(test, tokenizer)
    
    # Build LSH for the training set
    lsh, train_minhashes = build_lsh(train, num_perm=args.num_perm, threshold=args.threshold)
    
    # Find and remove duplicates from validation and test sets
    val_duplicates = find_near_duplicates(val, lsh, "train_")
    val_clean = val.drop(index=val_duplicates)
    
    test_duplicates = find_near_duplicates(test, lsh, "train_")
    test_clean = test.drop(index=test_duplicates)

    # Convert index lists back to their original form
    test_clean["idx"] = test_clean["idx"].apply(lambda x: x[0])
    
    # Save the cleaned test indices to a CSV file
    print(f"Saving cleaned test indices to {args.output_csv}...")
    test_clean[["idx"]].to_csv(args.output_csv, index=False)
    print("Finished processing.")

if __name__ == "__main__":
    main()
