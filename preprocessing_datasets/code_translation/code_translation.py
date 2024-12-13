import os
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Process and prepare a DataFrame from source and target files.")
    
    # File paths
    parser.add_argument("--train_java", type=str, default="train.java-cs.txt.java", help="Path to the training Java file.")
    parser.add_argument("--train_c", type=str, default="train.java-cs.txt.cs", help="Path to the training C file.")
    parser.add_argument("--valid_java", type=str, default="valid.java-cs.txt.java", help="Path to the validation Java file.")
    parser.add_argument("--valid_c", type=str, default="valid.java-cs.txt.cs", help="Path to the validation C file.")
    parser.add_argument("--test_java", type=str, default="test.java-cs.txt.java", help="Path to the test Java file.")
    parser.add_argument("--test_c", type=str, default="test.java-cs.txt.cs", help="Path to the test C file.")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Define paths and splits
    PATHS = (
        (args.train_c, args.train_java), 
        (args.valid_c, args.valid_java), 
        (args.test_c, args.test_java)
    )

    SPLITS = ("train", "valid", "test")
    
    # Initialize lists for DataFrame
    sources = []
    targets = []
    splits = []

    # Read the content from files
    for (target, source), split in zip(PATHS, SPLITS):
        with open(source, "r") as f1, open(target, "r") as f2:
            source_lines = f1.readlines()
            target_lines = f2.readlines()
        sources += source_lines
        targets += target_lines
        splits += [split] * len(source_lines)

    # Create the DataFrame
    df = pd.DataFrame(
        {
            "source": sources,
            "target": targets,
            "split": splits
        }
    )

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Prepare the 'temp' column with Java code wrapped in a dummy class
    df["temp"] = df["source"].apply(lambda x: "public class dummyClass { " + x + " }")
    df.reset_index(inplace=True)
    df.drop(columns="index", inplace=True)
    df.index.name = "idx"
    df["temp_idx"] = df.index

    # Output the DataFrame
    df.to_csv(f"../../code_translation.csv", index=True)

if __name__ == "__main__":
    main()
