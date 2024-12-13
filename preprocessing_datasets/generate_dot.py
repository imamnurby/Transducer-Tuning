import os
import pandas as pd
import subprocess
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor
import argparse

def save_files(input_str: str, idx: int, target_dir: str) -> None:
    filepath = os.path.join(target_dir, str(idx))
    filepath += ".java"
    with open(filepath, "w", errors="ignore") as f:
        f.write(input_str)

def generate_dot(index: int, input_dir: str, output_dir: str) -> Tuple[str, str]:
    input_filename = str(index) + ".java"
    input_filepath = os.path.join(input_dir, input_filename)

    output_dir = os.path.join(output_dir, str(index))
    os.makedirs(output_dir, exist_ok=True)

    output_cpg_filename = str(index) + ".bin"
    output_cpg_filepath = os.path.join(output_dir, output_cpg_filename)

    output_dot_dir = os.path.join(output_dir, "dot")

    parse_command = f"/opt/joern/joern-cli/joern-parse {input_filepath} --output {output_cpg_filepath}"
    try:
        subprocess.run(parse_command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        return "fail"
    
    export_command = f"/opt/joern/joern-cli/joern-export {output_cpg_filepath} --repr cpg14 --out {output_dot_dir}"
    try:
        subprocess.run(export_command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        return "fail"

    return output_cpg_filepath, output_dot_dir

def process_csv(csv_filepath: str, start: int, end: str, task_name: str, save_java: bool, multiprocessing: bool, num_workers: int, output_filename: str):
    df = pd.read_csv(csv_filepath, index_col="idx")

    os.makedirs(f"{task_name}_dot_files", exist_ok=True)
    os.makedirs(f"{task_name}_java_files", exist_ok=True)

    if save_java:
        df.apply(lambda x: save_files(x.temp, x.temp_idx, "java_files"), axis=1)

    df = df[start:end].copy() if end else df[start:].copy()

    if not multiprocessing:
        df["cpg_dot"] = df.apply(lambda x: generate_dot(x.temp_idx, "java_files", "dot_files"), axis=1)
    else:
        def process_row(row):
            return generate_dot(row.temp_idx, "java_files", "dot_files")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_row, df.itertuples(index=False)))
        df["cpg_dot"] = results

    df = df[df.cpg_dot != "fail"]
    df["dot"] = df.cpg_dot.apply(lambda x: x[1] + "/0-cpg.dot")
    df.drop(columns=["cpg_dot", "temp_idx"], inplace=True)

    df.to_csv(f"{output_filename}.csv", index=True)
    preview = df[:3]
    preview.to_csv(f"{output_filename}_preview.csv")

def parse_args():
    parser = argparse.ArgumentParser(description="Process a CSV file to generate dot files using Joern.")
    
    parser.add_argument("csv_filepath", type=str, help="Path to the CSV file to process.")
    parser.add_argument("--start", type=int, default=0, help="Start index for slicing the DataFrame.")
    parser.add_argument("--end", type=str, default="", help="End index for slicing the DataFrame. Leave empty to process until the end.")
    parser.add_argument("--task_name", type=str, default="", help="Name of the target task.")
    parser.add_argument("--save_java", action="store_true", help="Flag to save the temporary Java files.")
    parser.add_argument("--multiprocessing", action="store_true", help="Flag to use multiprocessing for generating dot files.")
    parser.add_argument("--num_workers", type=int, default=25, help="Number of workers for multiprocessing.")
    parser.add_argument("--output_filename", type=str, default="processed", help="Prefix for the output filenames.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    process_csv(
        csv_filepath=args.csv_filepath,
        start=args.start,
        end=args.end,
        task_name=args.task_name,
        save_java=args.save_java,
        multiprocessing=args.multiprocessing,
        num_workers=args.num_workers,
        output_filename=args.output_filename
    )

if __name__ == "__main__":
    main()
