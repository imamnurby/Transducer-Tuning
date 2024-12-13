import os
import argparse

import pandas as pd
import evaluate
from transformers import AutoTokenizer
from tqdm import tqdm
tqdm.pandas()

from metrics import PostProcessor, bleus, calc_codebleu

EXPERIMENT_DIRECTORY = "../experiments"

MODEL_TO_CKPT = {
    "codet5p-220m": "Salesforce/codet5p-220m",
    "codet5p-770m": "Salesforce/codet5p-770m",
    "codet5-base": "Salesforce/codet5-base",
    "codet5-large": "Salesforce/codet5-large",
}
TASK_TO_MAX_LEN = {
    "assert_generation": 220,
    "summarization": 175,
    "code_repair_short": 110,
    "code_repair_long": 210,
    "code_translation": 200
}

def load_df(tasks: list, model_dirs: list, tuning_methods: list, seeds: list)->dict:
    output_dict = {}
    for task in tasks:
        output_dict[task] = {}

        for model_dir in model_dirs:
            output_dict[task][model_dir] = {}
            
            for tuning_method in tuning_methods:            
                temp_df_list = []
                for seed in seeds:
                    directory_path = os.path.join(EXPERIMENT_DIRECTORY, task, model_dir, tuning_method, seed)
                    for root, dirnames, filenames in os.walk(directory_path):
                        for filename in filenames:
                            if filename == "result_on_test_set.csv":
                                filepath = os.path.join(root, filename)
                                print(f"found {filepath}")
                                temp_df = pd.read_csv(filepath, index_col="idx")
                                temp_df["seed"] = seed
                                temp_df_list.append(temp_df)
                if temp_df_list:
                    assert len(temp_df_list)==3
                    df_combined = pd.concat(temp_df_list)
                    output_dict[task][model_dir][tuning_method] = df_combined
    
    return output_dict

def compute_bleus(pred: str, truth: str):
    return bleus(
        [[truth.split()]], [pred.split()]
    )

def get_codebleu_stat(truths: str, predictions: str, task: str)->float:
    if task == "code_translation":
        result = calc_codebleu([truths], [predictions], lang="c_sharp", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
    else:
        result = calc_codebleu([truths], [predictions], lang="java", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
    return result

def compute_codebleu(x):
    codebleu_dict = x["codebleu_stat"]
    weight_bleu = 0.6
    weight_dataflow_match = 0.2
    weight_syntax_match = 0.2
    return (weight_bleu*x["bleu-cn"] + weight_dataflow_match*codebleu_dict["dataflow_match_score"]*100 + weight_syntax_match*codebleu_dict["syntax_match_score"]*100)

em_scorer  = evaluate.load("exact_match")
def compute_exact_match(pred: str, truth: str):
    score = em_scorer.compute(predictions=[pred], references=[truth], regexes_to_ignore=["\n"], ignore_punctuation=True)["exact_match"]
    return score

chrf_scorer = evaluate.load("chrf")
def compute_chrf(pred: str, truth: str):
    score = chrf_scorer.compute(predictions=[pred], references=[[truth]], eps_smoothing=True)
    return score

bleu_scorer = evaluate.load("bleu")
def compute_bleu4(pred: str, truth: str):
    score = bleu_scorer.compute(predictions=[pred], references=[[truth]])
    return score

def compute_metrics_per_instance(df: pd.DataFrame, postprocessing_fn: PostProcessor, backbone_model_path: str, max_pred_len, task: str)->dict:
    # prepare data
    tok = AutoTokenizer.from_pretrained(backbone_model_path)
    df["raw_preds"] = df["raw_preds"].apply(lambda x: tok.encode(x, max_length=max_pred_len))
    df["raw_preds"] = df["raw_preds"].apply(lambda x: tok.decode(x))
    df["preds"] = df["raw_preds"].apply(lambda row: postprocessing_fn(row))
    df["labels"] = df["labels"].apply(lambda row: postprocessing_fn(row))
    
    # compute em
    df["em"] = df.progress_apply(lambda x: compute_exact_match(x["preds"], x["labels"]), axis=1)
    
    # compute chrf
    df["chrf_stat"] = df.progress_apply(lambda x: compute_chrf(x["preds"], x["labels"]), axis=1)
    df["chrf"] = df["chrf_stat"].progress_apply(lambda x: x.get("score"))
    
    # compute bleu
    df["bleu-cn_stat"] = df.progress_apply(lambda x: compute_bleus(x["preds"], x["labels"]), axis=1)
    df["bleu-cn"] = df["bleu-cn_stat"].progress_apply(lambda x: x.get("BLEU-CN"))
    
    df["bleu-4_stat"] = df.progress_apply(lambda x: compute_bleu4(x["preds"], x["labels"]), axis=1)
    df["bleu-4"] = df["bleu-4_stat"].progress_apply(lambda x: x.get("score"))
    
    # compute codebleu
    if task != "summarization":
        df["codebleu_stat"] = df.progress_apply(lambda x: get_codebleu_stat(x["preds"], x["labels"], task), axis=1)
        df["codebleu-cn"] = df.progress_apply(lambda x: compute_codebleu(x), axis=1)

    return df

def prepare_all_df(df_dict: dict, output_dir: str):
    for task in df_dict:
        for model_dir in df_dict[task]:
           for tuning_method in df_dict[task][model_dir]:
                print(f"computing: {task}/{model_dir}/{tuning_method}")           
                df = df_dict[task][model_dir][tuning_method]
                backbone_model_path = MODEL_TO_CKPT[model_dir]
                postprocessing_fn = PostProcessor(backbone_model_path)
                max_pred_len = TASK_TO_MAX_LEN[task]
                df = compute_metrics_per_instance(df, postprocessing_fn, backbone_model_path, max_pred_len, task)

                temp_output_dir = os.path.join(output_dir, task, "computed")
                os.makedirs(temp_output_dir, exist_ok=True)
                
                output_filepath = os.path.join(temp_output_dir, f"{model_dir}_{tuning_method}.csv")
                df.to_csv(output_filepath, index=False)

def main():
    parser = argparse.ArgumentParser(description="Load data frames and prepare them for analysis.")
    
    parser.add_argument("--tasks", nargs='+', required=True, help="List of tasks.")
    parser.add_argument("--model_dirs", nargs='+', required=True, help="List of model directories.")
    parser.add_argument("--tuning_methods", nargs='+', required=True, help="List of tuning methods.")
    parser.add_argument("--seeds", nargs='+', required=True, help="List of seeds.")
    parser.add_argument("--output_dir", required=True, help="Output directory for the prepared data frames.")
    
    args = parser.parse_args()
    
    df_dict = load_df(args.tasks, args.model_dirs, args.tuning_methods, args.seeds)
    print(df_dict)
    prepare_all_df(df_dict, args.output_dir)

if __name__ == "__main__":
    main()