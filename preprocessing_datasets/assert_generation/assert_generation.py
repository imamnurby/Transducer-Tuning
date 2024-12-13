import os
import pandas as pd
from tqdm import tqdm
import argparse

tqdm.pandas()

def parse_args():
    parser = argparse.ArgumentParser(description="Process text files and prepare a DataFrame.")
    
    # File paths and options
    parser.add_argument("--base_directory", type=str, default=".", help="Base directory containing the input files.")
    parser.add_argument("--splits", type=str, nargs='+', default=["valid", "test", "train"], help="List of splits to process.")
    return parser.parse_args()

def read_txt_files(splits: list) -> pd.DataFrame:
    inputs_list = []
    outputs_list = []
    split_list = []
    for split in splits:
        inputs = f"{split}_methods.txt"
        outputs = f"{split}_assert.txt"
        
        with open(inputs, "r") as f:
            contents = f.readlines()
        inputs_list.extend(contents)

        with open(outputs, "r") as f:
            contents = f.readlines()
        outputs_list.extend(contents)
        split_list.extend([split] * len(contents))

    df = pd.DataFrame({
        "source": inputs_list,
        "target": outputs_list,
        "split": split_list
    })
    
    return df

def main():
    args = parse_args()

    # Read the text files and create a DataFrame
    df = read_txt_files(args.splits)

    df.drop_duplicates(inplace=True)


    # Prepare the 'temp' column with Java code wrapped in a dummy class
    df["temp"] = df["source"]
    df["temp"] = df.source.apply(lambda x: "public class dummyClass { " + x + " }")

    # Reset the index and prepare a temporary index column
    df.reset_index(inplace=True)
    df.drop(columns="index", inplace=True)
    df.index.name = "idx"
    df["temp_idx"] = df.index

    # Save the DataFrame to a CSV file
    df.to_csv("../../assert_generation.csv", index=True, escapechar='\\')
    
if __name__ == "__main__":
    main()
