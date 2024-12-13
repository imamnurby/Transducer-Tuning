import os
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Process JSON files and prepare a DataFrame.")
    
    # File paths
    parser.add_argument("--train_file", type=str, default="train.json", help="Path to the training JSON file.")
    parser.add_argument("--eval_file", type=str, default="eval.json", help="Path to the evaluation JSON file.")
    return parser.parse_args()

def main():
    args = parse_args()

    df = []
    for filename, split in zip((args.train_file, args.eval_file), ("train", "eval")):
        temp_df = pd.read_json(filename)
        temp_df["split"] = split
        df.append(temp_df)

    df = pd.concat(df)

    # Select relevant columns and rename them
    df = df[["query", "pos", "split"]]
    df.rename(columns={
        "query": "source",
        "pos": "target"
    }, inplace=True)

    # Display the shape of the DataFrame before and after removing duplicates
    df.drop_duplicates(inplace=True)

    # Prepare the 'temp' column with Java code wrapped in a dummy class
    df["temp"] = df["source"]
    df["temp"] = df.source.apply(lambda x: "public class dummyClass { " + x + " }")

    # Reset index and prepare temporary index column
    df.reset_index(inplace=True)
    df.drop(columns="index", inplace=True)
    df.index.name = "idx"
    df["temp_idx"] = df.index

    # Output the DataFrame
    df.to_csv(f"../../code_summarization.csv", index=True)

if __name__ == "__main__":
    main()
