import os
import pandas as pd
from tqdm import tqdm
import argparse
from typing import Tuple

tqdm.pandas()

def parse_args():
    parser = argparse.ArgumentParser(description="Process text files and prepare a DataFrame.")
    
    # File paths
    parser.add_argument("--train", type=str, default="train", help="Path to the training directory.")
    parser.add_argument("--eval", type=str, default="eval", help="Path to the evaluation directory.")
    parser.add_argument("--test", type=str, default="test", help="Path to the test directory.")

    return parser.parse_args()

def read_txt_files(args: argparse.ArgumentParser) -> pd.DataFrame:
    DIRECTORIES = (args.train, args.eval, args.test)
    temp_list = []
    for dirname in DIRECTORIES:
        temp_dict = {}
        for filename in os.listdir(dirname):
            filepath = os.path.join(dirname, filename)
            with open(filepath, "r") as f:
                contents = f.readlines()
            col_name, _ = os.path.splitext(filename)
            temp_dict[col_name] = contents
        temp_df = pd.DataFrame(temp_dict)
        temp_df["split"] = dirname
        temp_list.append(temp_df)
    
    df = pd.concat(temp_list)
    return df

def main():
    args = parse_args()
    
    # Read text files from the base directory
    df = read_txt_files(args)
    
    # Drop duplicates and display the shape again
    df.drop_duplicates(inplace=True)
    
    # Rename columns
    df.rename(columns={
        "buggy": "source",
        "fixed": "target"
    }, inplace=True)

    # Prepare the 'temp' column with Java code wrapped in a dummy class
    df["temp"] = df["source"]
    df["temp"] = df.source.apply(lambda x: "public class dummyClass { " + x + " }")

    # Reset index and prepare a temporary index column
    df.reset_index(inplace=True)
    df.drop(columns="index", inplace=True)
    df.index.name = "idx"
    df["temp_idx"] = df.index

    # Output the DataFrame
    df.to_csv(f"../../code_repair.csv", index=True)

if __name__ == "__main__":
    main()
