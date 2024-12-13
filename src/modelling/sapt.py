import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import Seq2SeqLMOutput

from typing import Optional, Union, Tuple

from .norm import RMSNorm
from .gnn import GNN_MAPPING

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
BACKBONE_CLASS_MAPPING = {
    "encoder-decoder": AutoModelForSeq2SeqLM,
    "decoder": AutoModelForCausalLM,
    "encoder": AutoModelForSequenceClassification
}

# adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py
class SAPTConfig(PretrainedConfig):
    def __init__(
            self,
            backbone_hidden_size: int = 768,
            mlp_hidden_act: str = "silu",
            gnn_type: str = "GAT",
            gnn_input_hidden_size: int = 1024,
            gnn_output_hidden_size: int = 768,
            gnn_intermediate_hidden_size: int = 768,
            gnn_attn_heads: int = 8,
            backbone_model_path: Optional[str] = "Salesforce/codet5p-220m", 
            architecture: Optional[str] = "encoder-decoder",
            num_labels: int = 2,
            problem_type: str = "single_label_classification",
            layers_to_train: str = "classifier.bias,classifier.weight,pooler.dense.bias,pooler.dense.weight",
            **kwargs,
    ):
        self.mlp_hidden_size = backbone_hidden_size + gnn_output_hidden_size
        self.mlp_intermediate_hidden_size = self.mlp_hidden_size*2
        self.mlp_hidden_act = mlp_hidden_act
        self.gnn_type = gnn_type
        self.gnn_input_hidden_size = gnn_input_hidden_size
        self.gnn_output_hidden_size = gnn_output_hidden_size
        self.gnn_intermediate_hidden_size = gnn_intermediate_hidden_size
        self.gnn_attn_heads = gnn_attn_heads

        self.backbone_model_path = backbone_model_path
        self.backbone_hidden_size = backbone_hidden_size
        self.architecture = architecture
        self.num_labels = num_labels
        self.problem_type = problem_type
        self.layers_to_train = layers_to_train.split(",") if layers_to_train else []
        super().__init__(**kwargs)


class SAPTModelBase(PreTrainedModel):
    config_class = SAPTConfig

    def __init__(self, config):
        super().__init__(config)

    def freeze_backbone_model_params(self)->list:
        frozen_layers = []
        for name, param in self.backbone_model.named_parameters():
            if name not in self.config.layers_to_train:
                param.requires_grad = False
                frozen_layers.append(name)
                # print(f"Layer {name} is frozen")
        return frozen_layers
    
    def get_trainable_params(self)->float:
        backbone_param_count = 0
        for name, param in self.backbone_model.named_parameters():
            backbone_param_count += param.numel()
        
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()

        return all_param, backbone_param_count

    def print_trainable_params(self)->None:
        all_param, backbone_param_count = self.get_trainable_params()
        print(f"Number of params in the backbone model: {backbone_param_count}")
        print(f"Number of trainable parameters: {all_param-backbone_param_count}")
    
    def _reorder_cache(self, past_key_values, beam_idx):
        return self.backbone_model._reorder_cache(
            past_key_values,
            beam_idx
        )

class FeatureFusionModuleAttention(nn.Module):
    def __init__(self, backbone_hidden_size, gnn_ouput_hidden_size, intermediate_hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = intermediate_hidden_size // num_heads
        
        # Validate that the backbone size is divisible by the number of heads
        assert backbone_hidden_size % num_heads == 0, "backbone_size must be divisible by num_heads"
        
        self.query_proj = torch.nn.Linear(backbone_hidden_size, intermediate_hidden_size, bias=False)
        self.key_proj = torch.nn.Linear(gnn_ouput_hidden_size, intermediate_hidden_size, bias=False)
        self.value_proj = torch.nn.Linear(backbone_hidden_size, intermediate_hidden_size, bias=False)
        self.final_proj = torch.nn.Linear(intermediate_hidden_size, backbone_hidden_size, bias=False)

    def forward(self, inputs_embeds, structure_features):
        batch_size = inputs_embeds.size(0)
        seq_len = inputs_embeds.size(1)
        
        # Project inputs and structure features
        queries = self.query_proj(inputs_embeds).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # queries = inputs_embeds.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # To match the sequence length, we need to repeat structure_features for each position in the sequence
        structure_features = structure_features.unsqueeze(1).repeat(1, seq_len, 1)
        keys = self.key_proj(structure_features).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value_proj(inputs_embeds).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # values = inputs_embeds.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute scaled dot-product attention
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.head_dim ** 0.5
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
        attended_values = torch.matmul(attn_probs, values)
        
        # Concatenate heads and put through final linear layer
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        final_embeds = self.final_proj(attended_values)
        return final_embeds

class GraphFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(GraphFeatureExtractor, self).__init__()
        gnn_class = GNN_MAPPING[config.gnn_type]
        self.gnn = gnn_class(
            input_dim = config.gnn_input_hidden_size,
            hidden_dim = config.gnn_intermediate_hidden_size,
            out_dim = config.gnn_output_hidden_size,
            n_heads = config.gnn_attn_heads
        )
        
    def forward(
        self,
        node_features: Optional[torch.Tensor] = None, 
        edge_indices: Optional[torch.Tensor] = None, 
        node_features_shape: Optional[torch.Tensor] = None, 
        edge_indices_shape: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        # Find the shape with the maximum product for reshaping node_features
        max_product_idx = torch.argmax(node_features_shape[:, 0] * node_features_shape[:, 1])
        max_shape = node_features_shape[max_product_idx]
        node_features = node_features.view(-1, max_shape[0], max_shape[1])

        structure_features = []
        for idx, (node_feature, edge_indice) in enumerate(zip(node_features, edge_indices)):
            # Filter out padding (-100) from node features
            valid_node_mask = (node_feature != -100).all(dim=-1)
            node_feature = node_feature[valid_node_mask]

            # Filter out padding (-100) from edge indices
            valid_edge_mask = edge_indice != -100
            edge_indice = edge_indice[valid_edge_mask]

            # Reshape edge indices to their original shape
            original_shape = edge_indices_shape[idx]
            edge_indice = edge_indice.view(original_shape[0], original_shape[1])

            # Compute node representation, number of nodes x hidden dimension of the gnn
            structure_feature = self.gnn(node_feature, edge_indice)

            # Average pooling over nodes to get graph representation, the shape is (gnn_output_hidden_size)
            structure_feature = torch.mean(structure_feature, dim=0, keepdim=True)
            structure_features.append(structure_feature)

        # Combine all graph representations, the shape is (batch_size, gnn_output_hidden_size)
        structure_features = torch.cat(structure_features, dim=0)
        return structure_features
    
class GraphFeatureExtractorMean(nn.Module):
    def __init__(self, config):
        super(GraphFeatureExtractorMean, self).__init__()
        
    def forward(
        self,
        node_features: Optional[torch.Tensor] = None, 
        edge_indices: Optional[torch.Tensor] = None, 
        node_features_shape: Optional[torch.Tensor] = None, 
        edge_indices_shape: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        # Find the shape with the maximum product for reshaping node_features
        max_product_idx = torch.argmax(node_features_shape[:, 0] * node_features_shape[:, 1])
        max_shape = node_features_shape[max_product_idx]
        node_features = node_features.view(-1, max_shape[0], max_shape[1])

        structure_features = []
        for idx, (node_feature, edge_indice) in enumerate(zip(node_features, edge_indices)):
            # Filter out padding (-100) from node features
            valid_node_mask = (node_feature != -100).all(dim=-1)
            node_feature = node_feature[valid_node_mask]

            # Filter out padding (-100) from edge indices
            valid_edge_mask = edge_indice != -100
            edge_indice = edge_indice[valid_edge_mask]

            # Reshape edge indices to their original shape
            original_shape = edge_indices_shape[idx]
            edge_indice = edge_indice.view(original_shape[0], original_shape[1])

            # Average pooling over nodes to get graph representation, the shape is (gnn_output_hidden_size)
            structure_feature = torch.mean(node_feature, dim=0, keepdim=True)
            structure_features.append(structure_feature)

        # Combine all graph representations, the shape is (batch_size, gnn_output_hidden_size)
        structure_features = torch.cat(structure_features, dim=0)
        return structure_features

class SAPTModelConcatPerVectorLinear(SAPTModelBase):
    def __init__(self, config):
        super().__init__(config)

        # Initialize backbone class
        backbone_class = BACKBONE_CLASS_MAPPING[config.architecture]
        if config.architecture != "encoder":
            self.backbone_model = backbone_class.from_pretrained(config.backbone_model_path)
        else:
            self.backbone_model = backbone_class.from_pretrained(config.backbone_model_path, num_labels=config.num_labels, problem_type=config.problem_type)

        if config.gradient_checkpointing:
            self.backbone_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":True})

        # Freeze the backbone model
        frozen_layers = self.freeze_backbone_model_params()
        assert len(frozen_layers) > 0
        for layer in config.layers_to_train:
            assert layer not in frozen_layers

        # Initialize GNN
        self.feature_extractor = GraphFeatureExtractor(config)
        self.feature_fusion = FeatureFusionModuleAttention(backbone_hidden_size=config.backbone_hidden_size, intermediate_hidden_size=config.gnn_intermediate_hidden_size, gnn_ouput_hidden_size=config.gnn_output_hidden_size, num_heads=1)
        self.layer_norm_structure_feature = RMSNorm(config.gnn_output_hidden_size)
        self.layer_norm_inputs_embeds = RMSNorm(config.backbone_hidden_size)
        
        # # Initialize Projection Layer
        # self.proj_up = nn.Linear(in_features=config.backbone_hidden_size, out_features=config.mlp_intermediate_hidden_size, bias=False)
        # self.proj_down = nn.Linear(in_features=config.mlp_intermediate_hidden_size, out_features=config.backbone_hidden_size, bias=False)
        # self.layer_norm_mlp = RMSNorm(config.backbone_hidden_size)
        self.print_trainable_params()
    
    def get_feature_extractor(self)->torch.nn:
        return self.gnn

    def _get_final_inputs_embeds_from_input_ids(
        self,
        input_ids: Optional[torch.LongTensor] = None, 
        node_features: Optional[torch.Tensor] = None, 
        edge_indices: Optional[torch.Tensor] = None, 
        node_features_shape: Optional[torch.Tensor] = None, 
        edge_indices_shape: Optional[torch.Tensor] = None
    )->torch.Tensor:
        
        # Embed input_ids
        # The shape is (batch_size, num_tokens, backbone_hidden_size) (768)
        inputs_embeds = self.backbone_model.get_input_embeddings()(input_ids)

        # Compute structure feature
        # The shape of structure_features is (batch_size, gnn_output_hidden_size)
        structure_features = self.feature_extractor(
            node_features=node_features,
            edge_indices=edge_indices,
            node_features_shape=node_features_shape,
            edge_indices_shape=edge_indices_shape
        )

        # Combine inputs_embeds with the structure feature
        # The shape is (batch_size, num_tokens, backbone_hidden_size)
        inputs_embeds = self.layer_norm_inputs_embeds(inputs_embeds)
        structure_features = self.layer_norm_structure_feature(structure_features)
        inputs_embeds = self.feature_fusion(inputs_embeds, structure_features)
        return inputs_embeds

    def get_encoder_output(
        self,
        input_ids: Optional[torch.LongTensor] = None, 
        attention_mask: Optional[torch.FloatTensor] = None,
        node_features: Optional[torch.Tensor] = None, 
        edge_indices: Optional[torch.Tensor] = None, 
        node_features_shape: Optional[torch.Tensor] = None, 
        edge_indices_shape: Optional[torch.Tensor] = None
    )->torch.Tensor:
        
        inputs_embeds=self._get_final_inputs_embeds_from_input_ids(
            input_ids=input_ids,
            node_features=node_features,
            node_features_shape=node_features_shape,
            edge_indices=edge_indices,
            edge_indices_shape=edge_indices_shape
        )
        encoder_outputs = self.backbone_model.get_encoder()(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return encoder_outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        if (hasattr(self.backbone_model.config, "architectures") 
            and self.backbone_model.config.architectures 
            and self.backbone_model.config.architectures[0]  == "PLBartForConditionalGeneration"):
            return self.backbone_model.prepare_inputs_for_generation(
                input_ids,
                past_key_values,
                attention_mask,
                head_mask,
                decoder_head_mask,
                # decoder_attention_mask,
                cross_attn_head_mask,
                use_cache,
                encoder_outputs,
                **kwargs
            )
        else:
            return self.backbone_model.prepare_inputs_for_generation(
                input_ids,
                past_key_values,
                attention_mask,
                head_mask,
                decoder_head_mask,
                decoder_attention_mask,
                cross_attn_head_mask,
                use_cache,
                encoder_outputs,
                **kwargs
            )
        
    def forward(
        self, 
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        node_features: Optional[torch.Tensor] = None, 
        edge_indices: Optional[torch.Tensor] = None, 
        node_features_shape: Optional[torch.Tensor] = None, 
        edge_indices_shape: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        if encoder_outputs is None:
            # Embed input ids, the shape is (batch_size, num_tokens, backbone_hidden_size) (768)
            inputs_embeds=self._get_final_inputs_embeds_from_input_ids(
                input_ids=input_ids,
                node_features=node_features,
                node_features_shape=node_features_shape,
                edge_indices=edge_indices,
                edge_indices_shape=edge_indices_shape
            )
                    
            # Generate outputs from the backbone model
            if self.config.architecture != "encoder":
                outputs = self.backbone_model(
                    # input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    head_mask=head_mask,
                    decoder_head_mask=decoder_head_mask,
                    labels=labels,
                    cross_attn_head_mask=cross_attn_head_mask,
                    # encoder_outputs=encoder_outputs,
                    past_key_values=past_key_values,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            else:
                outputs = self.backbone_model(
                    # input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            
        else:
            outputs = self.backbone_model(
                # input_ids=input_ids,
                # inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                labels=labels,
                cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

        return outputs

class SAPTModelConcatPerVectorNoGVE(SAPTModelBase):
    def __init__(self, config):
        super().__init__(config)

        # Initialize backbone class
        backbone_class = BACKBONE_CLASS_MAPPING[config.architecture]
        if config.architecture != "encoder":
            self.backbone_model = backbone_class.from_pretrained(config.backbone_model_path)
        else:
            self.backbone_model = backbone_class.from_pretrained(config.backbone_model_path, num_labels=config.num_labels, problem_type=config.problem_type)

        if config.gradient_checkpointing:
            self.backbone_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":True})

        # Freeze the backbone model
        frozen_layers = self.freeze_backbone_model_params()
        assert len(frozen_layers) > 0
        for layer in config.layers_to_train:
            assert layer not in frozen_layers

        # Initialize GNN
        self.feature_extractor = GraphFeatureExtractorMean(config)
        self.feature_fusion = FeatureFusionModuleAttention(backbone_hidden_size=config.backbone_hidden_size, intermediate_hidden_size=config.gnn_intermediate_hidden_size, gnn_ouput_hidden_size=config.gnn_output_hidden_size, num_heads=1)
        self.layer_norm_structure_feature = RMSNorm(config.gnn_output_hidden_size)
        self.layer_norm_inputs_embeds = RMSNorm(config.backbone_hidden_size)
        
        # # Initialize Projection Layer
        # self.proj_up = nn.Linear(in_features=config.backbone_hidden_size, out_features=config.mlp_intermediate_hidden_size, bias=False)
        # self.proj_down = nn.Linear(in_features=config.mlp_intermediate_hidden_size, out_features=config.backbone_hidden_size, bias=False)
        # self.layer_norm_mlp = RMSNorm(config.backbone_hidden_size)
        self.print_trainable_params()
    
    def get_feature_extractor(self)->torch.nn:
        return self.gnn

    def _get_final_inputs_embeds_from_input_ids(
        self,
        input_ids: Optional[torch.LongTensor] = None, 
        node_features: Optional[torch.Tensor] = None, 
        edge_indices: Optional[torch.Tensor] = None, 
        node_features_shape: Optional[torch.Tensor] = None, 
        edge_indices_shape: Optional[torch.Tensor] = None
    )->torch.Tensor:
        
        # Embed input_ids
        # The shape is (batch_size, num_tokens, backbone_hidden_size) (768)
        inputs_embeds = self.backbone_model.get_input_embeddings()(input_ids)

        # Compute structure feature
        # The shape of structure_features is (batch_size, gnn_output_hidden_size)
        structure_features = self.feature_extractor(
            node_features=node_features,
            edge_indices=edge_indices,
            node_features_shape=node_features_shape,
            edge_indices_shape=edge_indices_shape
        )

        # Combine inputs_embeds with the structure feature
        # The shape is (batch_size, num_tokens, backbone_hidden_size)
        inputs_embeds = self.layer_norm_inputs_embeds(inputs_embeds)
        structure_features = self.layer_norm_structure_feature(structure_features)
        inputs_embeds = self.feature_fusion(inputs_embeds, structure_features)
        return inputs_embeds

    def get_encoder_output(
        self,
        input_ids: Optional[torch.LongTensor] = None, 
        attention_mask: Optional[torch.FloatTensor] = None,
        node_features: Optional[torch.Tensor] = None, 
        edge_indices: Optional[torch.Tensor] = None, 
        node_features_shape: Optional[torch.Tensor] = None, 
        edge_indices_shape: Optional[torch.Tensor] = None
    )->torch.Tensor:
        
        inputs_embeds=self._get_final_inputs_embeds_from_input_ids(
            input_ids=input_ids,
            node_features=node_features,
            node_features_shape=node_features_shape,
            edge_indices=edge_indices,
            edge_indices_shape=edge_indices_shape
        )
        encoder_outputs = self.backbone_model.get_encoder()(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return encoder_outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        if (hasattr(self.backbone_model.config, "architectures") 
            and self.backbone_model.config.architectures 
            and self.backbone_model.config.architectures[0]  == "PLBartForConditionalGeneration"):
            return self.backbone_model.prepare_inputs_for_generation(
                input_ids,
                past_key_values,
                attention_mask,
                head_mask,
                decoder_head_mask,
                # decoder_attention_mask,
                cross_attn_head_mask,
                use_cache,
                encoder_outputs,
                **kwargs
            )
        else:
            return self.backbone_model.prepare_inputs_for_generation(
                input_ids,
                past_key_values,
                attention_mask,
                head_mask,
                decoder_head_mask,
                decoder_attention_mask,
                cross_attn_head_mask,
                use_cache,
                encoder_outputs,
                **kwargs
            )
        
    def forward(
        self, 
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        node_features: Optional[torch.Tensor] = None, 
        edge_indices: Optional[torch.Tensor] = None, 
        node_features_shape: Optional[torch.Tensor] = None, 
        edge_indices_shape: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        if encoder_outputs is None:
            # Embed input ids, the shape is (batch_size, num_tokens, backbone_hidden_size) (768)
            inputs_embeds=self._get_final_inputs_embeds_from_input_ids(
                input_ids=input_ids,
                node_features=node_features,
                node_features_shape=node_features_shape,
                edge_indices=edge_indices,
                edge_indices_shape=edge_indices_shape
            )
                    
            # Generate outputs from the backbone model
            if self.config.architecture != "encoder":
                outputs = self.backbone_model(
                    # input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    head_mask=head_mask,
                    decoder_head_mask=decoder_head_mask,
                    labels=labels,
                    cross_attn_head_mask=cross_attn_head_mask,
                    # encoder_outputs=encoder_outputs,
                    past_key_values=past_key_values,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            else:
                outputs = self.backbone_model(
                    # input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            
        else:
            outputs = self.backbone_model(
                # input_ids=input_ids,
                # inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                labels=labels,
                cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

        return outputs


class SAPTModelConcatPerVectorNoABF(SAPTModelBase):
    def __init__(self, config):
        super().__init__(config)

        # Initialize backbone class
        backbone_class = BACKBONE_CLASS_MAPPING[config.architecture]
        if config.architecture != "encoder":
            self.backbone_model = backbone_class.from_pretrained(config.backbone_model_path)
        else:
            self.backbone_model = backbone_class.from_pretrained(config.backbone_model_path, num_labels=config.num_labels, problem_type=config.problem_type)

        if config.gradient_checkpointing:
            self.backbone_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":True})

        # Freeze the backbone model
        frozen_layers = self.freeze_backbone_model_params()
        assert len(frozen_layers) > 0
        for layer in config.layers_to_train:
            assert layer not in frozen_layers

        # Initialize GNN
        self.feature_extractor = GraphFeatureExtractor(config)
        # self.feature_fusion = FeatureFusionModuleAttention(backbone_hidden_size=config.backbone_hidden_size, intermediate_hidden_size=config.gnn_intermediate_hidden_size, gnn_ouput_hidden_size=config.gnn_output_hidden_size, num_heads=1)
        self.layer_norm_structure_feature = RMSNorm(config.gnn_output_hidden_size)
        self.layer_norm_inputs_embeds = RMSNorm(config.backbone_hidden_size)
        
        # # Initialize Projection Layer
        # self.proj_up = nn.Linear(in_features=config.backbone_hidden_size, out_features=config.mlp_intermediate_hidden_size, bias=False)
        # self.proj_down = nn.Linear(in_features=config.mlp_intermediate_hidden_size, out_features=config.backbone_hidden_size, bias=False)
        # self.layer_norm_mlp = RMSNorm(config.backbone_hidden_size)
        self.print_trainable_params()
    
    def get_feature_extractor(self)->torch.nn:
        return self.gnn

    def _get_final_inputs_embeds_from_input_ids(
        self,
        input_ids: Optional[torch.LongTensor] = None, 
        node_features: Optional[torch.Tensor] = None, 
        edge_indices: Optional[torch.Tensor] = None, 
        node_features_shape: Optional[torch.Tensor] = None, 
        edge_indices_shape: Optional[torch.Tensor] = None
    )->torch.Tensor:
        
        # Embed input_ids
        # The shape is (batch_size, num_tokens, backbone_hidden_size) (768)
        inputs_embeds = self.backbone_model.get_input_embeddings()(input_ids)

        # Compute structure feature
        # The shape of structure_features is (batch_size, gnn_output_hidden_size)
        structure_features = self.feature_extractor(
            node_features=node_features,
            edge_indices=edge_indices,
            node_features_shape=node_features_shape,
            edge_indices_shape=edge_indices_shape
        )

        # Combine inputs_embeds with the structure feature
        # The shape is (batch_size, num_tokens, backbone_hidden_size)
        inputs_embeds = self.layer_norm_inputs_embeds(inputs_embeds)
        structure_features = self.layer_norm_structure_feature(structure_features)
        structure_features = structure_features.unsqueeze(1)

        inputs_embeds += structure_features
        
        return inputs_embeds

    def get_encoder_output(
        self,
        input_ids: Optional[torch.LongTensor] = None, 
        attention_mask: Optional[torch.FloatTensor] = None,
        node_features: Optional[torch.Tensor] = None, 
        edge_indices: Optional[torch.Tensor] = None, 
        node_features_shape: Optional[torch.Tensor] = None, 
        edge_indices_shape: Optional[torch.Tensor] = None
    )->torch.Tensor:
        
        inputs_embeds=self._get_final_inputs_embeds_from_input_ids(
            input_ids=input_ids,
            node_features=node_features,
            node_features_shape=node_features_shape,
            edge_indices=edge_indices,
            edge_indices_shape=edge_indices_shape
        )
        encoder_outputs = self.backbone_model.get_encoder()(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return encoder_outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        if (hasattr(self.backbone_model.config, "architectures") 
            and self.backbone_model.config.architectures 
            and self.backbone_model.config.architectures[0]  == "PLBartForConditionalGeneration"):
            return self.backbone_model.prepare_inputs_for_generation(
                input_ids,
                past_key_values,
                attention_mask,
                head_mask,
                decoder_head_mask,
                # decoder_attention_mask,
                cross_attn_head_mask,
                use_cache,
                encoder_outputs,
                **kwargs
            )
        else:
            return self.backbone_model.prepare_inputs_for_generation(
                input_ids,
                past_key_values,
                attention_mask,
                head_mask,
                decoder_head_mask,
                decoder_attention_mask,
                cross_attn_head_mask,
                use_cache,
                encoder_outputs,
                **kwargs
            )
        
    def forward(
        self, 
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        node_features: Optional[torch.Tensor] = None, 
        edge_indices: Optional[torch.Tensor] = None, 
        node_features_shape: Optional[torch.Tensor] = None, 
        edge_indices_shape: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        if encoder_outputs is None:
            # Embed input ids, the shape is (batch_size, num_tokens, backbone_hidden_size) (768)
            inputs_embeds=self._get_final_inputs_embeds_from_input_ids(
                input_ids=input_ids,
                node_features=node_features,
                node_features_shape=node_features_shape,
                edge_indices=edge_indices,
                edge_indices_shape=edge_indices_shape
            )
                    
            # Generate outputs from the backbone model
            if self.config.architecture != "encoder":
                outputs = self.backbone_model(
                    # input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    head_mask=head_mask,
                    decoder_head_mask=decoder_head_mask,
                    labels=labels,
                    cross_attn_head_mask=cross_attn_head_mask,
                    # encoder_outputs=encoder_outputs,
                    past_key_values=past_key_values,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            else:
                outputs = self.backbone_model(
                    # input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            
        else:
            outputs = self.backbone_model(
                # input_ids=input_ids,
                # inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                labels=labels,
                cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

        return outputs

class SAPTModelNoGNN(SAPTModelBase):
    def __init__(self, config):
        super().__init__(config)

        assert((config.gnn_output_hidden_size==config.backbone_hidden_size)==True)

        backbone_class = BACKBONE_CLASS_MAPPING[config.architecture]
        if config.architecture != "encoder":
            self.backbone_model = backbone_class.from_pretrained(config.backbone_model_path)
        else:
            self.backbone_model = backbone_class.from_pretrained(config.backbone_model_path, num_labels=config.num_labels, problem_type=config.problem_type)

        if config.gradient_checkpointing:
            self.backbone_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={})

        # freeze the backbone model
        frozen_layers = self.freeze_backbone_model_params()
        assert len(frozen_layers) > 0
        for layer in config.layers_to_train:
            assert layer not in frozen_layers

        self.MLP = nn.Linear(in_features=config.backbone_hidden_size, out_features=config.backbone_hidden_size, bias=False)
        self.print_trainable_params()

    def get_feature_extractor(self)->torch.nn:
        return self.gnn
    
    def _get_final_inputs_embeds_from_input_ids(
        self,
        input_ids: Optional[torch.LongTensor] = None, 
        node_features: Optional[torch.Tensor] = None, 
        edge_indices: Optional[torch.Tensor] = None, 
        node_features_shape: Optional[torch.Tensor] = None, 
        edge_indices_shape: Optional[torch.Tensor] = None
    )->torch.Tensor:
        
        # Embed input_ids, the shape is (batch_size, num_tokens, backbone_hidden_size) (768)
        inputs_embeds = self.backbone_model.get_input_embeddings()(input_ids)
        final_embeds = self.MLP(inputs_embeds)
        return final_embeds

    def get_encoder_output(
        self,
        input_ids: Optional[torch.LongTensor] = None, 
        attention_mask: Optional[torch.FloatTensor] = None,
        node_features: Optional[torch.Tensor] = None, 
        edge_indices: Optional[torch.Tensor] = None, 
        node_features_shape: Optional[torch.Tensor] = None, 
        edge_indices_shape: Optional[torch.Tensor] = None
    )->torch.Tensor:
        
        inputs_embeds=self._get_final_inputs_embeds_from_input_ids(
            input_ids=input_ids,
            node_features=node_features,
            node_features_shape=node_features_shape,
            edge_indices=edge_indices,
            edge_indices_shape=edge_indices_shape
        )
    
        encoder_outputs = self.backbone_model.get_encoder()(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return encoder_outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        return self.backbone_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values,
            attention_mask,
            head_mask,
            decoder_head_mask,
            decoder_attention_mask,
            cross_attn_head_mask,
            use_cache,
            encoder_outputs,
            **kwargs
        )
        
    def forward(
        self, 
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        node_features: Optional[torch.Tensor] = None, 
        edge_indices: Optional[torch.Tensor] = None, 
        node_features_shape: Optional[torch.Tensor] = None, 
        edge_indices_shape: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        if encoder_outputs is None:
            # Embed input ids, the shape is (batch_size, num_tokens, backbone_hidden_size) (768)
            inputs_embeds=self._get_final_inputs_embeds_from_input_ids(
                input_ids=input_ids,
                node_features=node_features,
                node_features_shape=node_features_shape,
                edge_indices=edge_indices,
                edge_indices_shape=edge_indices_shape
            )
                    
            # Generate outputs from the backbone model
            if self.config.architecture != "encoder":
                outputs = self.backbone_model(
                    # input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    head_mask=head_mask,
                    decoder_head_mask=decoder_head_mask,
                    labels=labels,
                    cross_attn_head_mask=cross_attn_head_mask,
                    # encoder_outputs=encoder_outputs,
                    past_key_values=past_key_values,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            else:
                outputs = self.backbone_model(
                    # input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            
        else:

            outputs = self.backbone_model(
                # input_ids=input_ids,
                # inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                labels=labels,
                cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

        return outputs



