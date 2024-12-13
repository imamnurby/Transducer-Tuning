from .sapt import *
from .data_collator import *

SAPT_MODEL_MAPPING = {
    # ConcatPerVector Models
    "SAPTModelConcatPerVectorLinear": SAPTModelConcatPerVectorLinear,
    # Special Models
    "SAPTModelNoGNN": SAPTModelNoGNN,
    "SAPTModelConcatPerVectorNoGVE": SAPTModelConcatPerVectorNoGVE,
    "SAPTModelConcatPerVectorNoABF": SAPTModelConcatPerVectorNoABF
}

VALID_BASELINES = (
    "full-finetuning",
    "lora",
    "no-finetuning",
    "prefix-tuning",
    "prompt-tuning"
)