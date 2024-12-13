import sys
sys.path.append("./")
import re
import evaluate
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from typing import List
from .bleu.bleu_main import bleus
from codebleu import calc_codebleu

def compute_codebleu_c_sharp(truths: List[str], predictions: List[str], batch_size: int)->float:
    result = calc_codebleu(truths, predictions, lang="c_sharp", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
    return result["codebleu"]

def compute_codebleu_java(truths: List[str], predictions: List[str], batch_size: int)->float:
    result = calc_codebleu(truths, predictions, lang="java", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
    return result

def compute_bleu(truths: List[str], predictions: List[str], batch_size: int)->float:
    truths = [[x.split()] for x in truths]
    predictions = [x.split() for x in predictions]
    score_dict = bleus(truths, predictions)
    return score_dict

def compute_exact_match(truths: List[str], predictions: List[str], batch_size: int)->float:
    scorer  = evaluate.load("exact_match")
    score = scorer.compute(predictions=predictions, references=truths, regexes_to_ignore=["\n"], ignore_punctuation=True)["exact_match"]
    return round(score, 3)

class PostProcessor():
    def __init__(self, model_path: str):            
        special_tokens = []
        self.eos_token = None
        decoder_start_token_id = None
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        for key, token in tokenizer.special_tokens_map.items():
            if isinstance(token, str):
                if key == "eos_token":
                    self.eos_token = token
                elif key in ("decoder_start_token_id", "bos_token", "pad_token"):
                    decoder_start_token_id = token
                    special_tokens.append(token)
            # elif isinstance(token, list):
                # for t in token:
                #     special_tokens.append(t)
            #     continue
            # else:
            #     raise TypeError("Invalid token type")

        if not decoder_start_token_id:
            config = AutoConfig.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            if hasattr(tokenizer, "decoder_start_token_id"):
                decoder_start_token_id = tokenizer.decoder_start_token_id
            elif hasattr(config, "decoder_start_token_id"):
                decoder_start_token_id = config.decoder_start_token_id
            elif hasattr(model.config, "decoder_start_token_id"):
                decoder_start_token_id = model.config.decoder_start_token_id
            
            decoder_start_token_id = tokenizer.decode(decoder_start_token_id)

            if decoder_start_token_id != None:          
                special_tokens.append(decoder_start_token_id)
            else:
                print("### no decoder_start_token_id found! ###")
        self.pattern = '|'.join(re.escape(token) for token in special_tokens)
                
    def _replace_special_tokens_with_empty_str(self, s: str):
        return re.sub(self.pattern, "", s)
    
    def _get_pred(self, s: str):
         pred = [segment for segment in s.split(self.eos_token) if segment]
         return pred[0] if pred else s
    
    def __call__(self, s: str):
        pred = self._get_pred(s)
        pred = self._replace_special_tokens_with_empty_str(pred)
        pred = pred.strip("\n").strip()
        return pred

def normalize_string(s: str):
    return s.strip().strip("\n")

def no_postprocessing(label: int)->int:
    return label

METRIC_TO_EVALUATOR = {
    "exact_match": compute_exact_match,
    "bleu": compute_bleu,
    "codebleu_c_sharp": compute_codebleu_c_sharp,
    "codebleu_java": compute_codebleu_java
}

METRIC_NAME_TO_LABEL_KEY_MAPPING = {
    "exact_match": "labels",
    "bleu": "labels",
    "codebleu_c_sharp": "labels",
    "codebleu_java": "labels"
}