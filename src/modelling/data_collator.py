from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from typing import Optional, Union, Any

# adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py
def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded

@dataclass
class DataCollatorForClassification:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    label_pad_token_id: int = -100
    using_graph: bool = True

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        
        padding_side = self.tokenizer.padding_side  
        
        for feature in features:
            if len(feature["input_ids"]) >= self.max_length:
                feature["input_ids"] = feature["input_ids"][0:self.max_length-2] + [self.tokenizer.eos_token_id]
                feature["attention_mask"] = [1]*len(feature["input_ids"])
        
        if self.using_graph:
            node_features = [feature["node_features"] for feature in features] if "node_features" in features[0].keys() else None
            max_node_features_length = max(len(f) for f in node_features)
                
            edge_indices = [feature["edge_indices"] for feature in features] if "edge_indices" in features[0].keys() else None
            max_edge_indices_length = max(len(f) for f in edge_indices)

            for feature in features:
                if self.using_graph:
                    remainder = [self.label_pad_token_id] * (max_node_features_length - len(feature["node_features"]))
                    feature["node_features"] = (
                        feature["node_features"] + remainder if padding_side == "right" else remainder + feature["node_features"]
                    )

                    remainder = [self.label_pad_token_id] * (max_edge_indices_length - len(feature["edge_indices"]))
                    feature["edge_indices"] = (
                        feature["edge_indices"] + remainder if padding_side == "right" else remainder + feature["edge_indices"]
                    )
        else:
            for feature in features:   
                for key in ("node_features", "edge_indices", "node_features_shape", "edge_indices_shape"):
                    if key in feature:
                        del feature[key]
        
        features = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors=return_tensors,
        )

        return features

@dataclass
class DataCollatorForSeq2SeqWithGraph:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`], *optional*):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizer
    model: Optional[Any] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        
        padding_side = self.tokenizer.padding_side
        
        label_exist = None if "labels" not in features[0].keys() else True        
        
        # Get max length for labels, node_features, edge_indices
        for feature in features:
            if label_exist:
                if len(feature["labels"]) >= self.max_length:
                    # Ensure the sequence ends with the EOS token
                    feature["labels"] = (feature["labels"][0:self.max_length-2]
                                        + [self.tokenizer.eos_token_id])

            if len(feature["input_ids"]) >= self.max_length:
                feature["input_ids"] = feature["input_ids"][0:self.max_length-2] + [self.tokenizer.eos_token_id]
                feature["attention_mask"] = [1]*len(feature["input_ids"])
        
        if label_exist:
            labels = [feature["labels"] for feature in features]
            max_label_length = max(len(l) for l in labels)
        
        node_features = [feature["node_features"] for feature in features] if "node_features" in features[0].keys() else None
        max_node_features_length = max(len(f) for f in node_features)
            
        edge_indices = [feature["edge_indices"] for feature in features] if "edge_indices" in features[0].keys() else None
        max_edge_indices_length = max(len(f) for f in edge_indices)

        for feature in features:
            if label_exist:
                remainder = [self.tokenizer.eos_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )
            
            remainder = [self.label_pad_token_id] * (max_node_features_length - len(feature["node_features"]))
            feature["node_features"] = (
                feature["node_features"] + remainder if padding_side == "right" else remainder + feature["node_features"]
            )

            remainder = [self.label_pad_token_id] * (max_edge_indices_length - len(feature["edge_indices"]))
            feature["edge_indices"] = (
                feature["edge_indices"] + remainder if padding_side == "right" else remainder + feature["edge_indices"]
            )

        features = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors=return_tensors,
        )
        
        # prepare decoder_input_ids
        if (
            label_exist
            and self.model.backbone_model is not None
            and hasattr(self.model.backbone_model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.backbone_model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features
    
@dataclass
class DataCollatorForSeq2SeqWithoutGraph:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`], *optional*):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizer
    model: Optional[Any] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        padding_side = self.tokenizer.padding_side

        label_exist = None if "labels" not in features[0].keys() else True  

        # Get max length for labels, node_features, edge_indices
        for feature in features:
            if label_exist:
                if len(feature["labels"]) >= self.max_length:
                    # Ensure the sequence ends with the EOS token
                    feature["labels"] = (feature["labels"][0:self.max_length-2]
                                        + [self.tokenizer.eos_token_id])

            if len(feature["input_ids"]) >= self.max_length:
                feature["input_ids"] = feature["input_ids"][0:self.max_length-2] + [self.tokenizer.eos_token_id]
                feature["attention_mask"] = [1]*len(feature["input_ids"])

        if label_exist:
            labels = [feature["labels"] for feature in features]
            max_label_length = max(len(l) for l in labels)

        for feature in features:
            if label_exist:
                remainder = [self.tokenizer.eos_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )

            for key in ("node_features", "edge_indices", "node_features_shape", "edge_indices_shape"):
                if key in feature:
                    del feature[key]
        
        features = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors=return_tensors,
        )
        # prepare decoder_input_ids
        if (
            label_exist is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids
        return features