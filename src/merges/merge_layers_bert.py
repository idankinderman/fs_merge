import torch
import torch.nn as nn

import copy
import math
from typing import List, Dict

from transformers import BertForSequenceClassification
from merges.merge_layers_vit_full import MMatrix, UMatrix, _bmm_linear_output, bmm_folding_M_U_into_weight, _bmm_linear

def transpose_for_scores(x, num_attention_heads, attention_head_size):
    new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
    x = x.view(new_x_shape)
    return x.permute(0, 2, 1, 3)


def load_ber_model(path):
    model = BertForSequenceClassification.from_pretrained(path)
    model = model.bert
    model = torch.nn.DataParallel(model, device_ids=[torch.cuda.device(i) for i in range(torch.cuda.device_count())])
    model.eval()
    model.cuda()
    return model


def average_models(model1, model2):
    # Create a new model by copying the structure of the first model
    averaged_model = copy.deepcopy(model1)

    # Iterate over the parameters of the new model and the two input models
    for (param1, param2, averaged_param) in zip(model1.parameters(), model2.parameters(), averaged_model.parameters()):
        # Average the weights
        averaged_param.data.copy_((param1.data + param2.data) / 2)

    return averaged_model

def compare_bert_merger(merger, batch):
    # Get BERT model
    model_1 = load_ber_model("/home/edank/text-transformers/merging-text-transformers-main/models/trained/multiberts/sst2/seed_4")
    model_2 = load_ber_model("/home/edank/text-transformers/merging-text-transformers-main/models/trained/multiberts/rte/seed_1")
    model = average_models(model_1, model_2)
    model.eval()
    model.cuda()

    # Data
    input_ids = batch['input_ids'].cuda()
    targets = batch['targets'].cuda()
    attention_mask = batch['attention_mask'].cuda()
    token_type_ids = batch['token_type_ids'].cuda()
    ids = batch['ids'].cuda()
    mask = ids == 0

    input_ids = input_ids[mask]
    targets = targets[mask]
    attention_mask = attention_mask[mask]
    token_type_ids = token_type_ids[mask]


    # Forward pass - BERT
    bert_output = bert_forward_pass(model, input_ids, attention_mask, token_type_ids)
    merger_output = bert_forward_pass(merger, input_ids, attention_mask, token_type_ids)
    _, emb = bert_output['pooled_output'].shape
    #merger_output = merger.module.forward_features(input_ids, attention_mask, token_type_ids)
    #merger_output = merger_output[:, :emb]

    # Loss
    loss_fn = torch.nn.MSELoss()

    for key in bert_output:
        if key == 'pooled_output':
            continue
        #print(f"\n{key} : bert shape {bert_output[key].shape} | merger shape {merger_output[key].shape}")
        print(f"{key} : Loss of loaded bert and merger: {loss_fn(bert_output[key], merger_output[key]).item()}, l2 norm: {torch.norm(bert_output[key])}")

    print("Loss of loaded bert and target: ", loss_fn(bert_output['pooled_output'], targets).item())
    print("Loss of merger and target: ", loss_fn(merger_output['pooled_output'], targets).item())
    print("Loss of loaded bert and merger: ", loss_fn(bert_output['pooled_output'], merger_output['pooled_output']).item())
    raise Exception("Stop here")

def bert_forward_pass(model, input_ids, attention_mask, token_type_ids):
    out_dict = {}

    num_attention_heads = model.module.encoder.layer[0].attention.self.num_attention_heads
    attention_head_size = model.module.encoder.layer[0].attention.self.attention_head_size

    # 1. Embeddings, use only input_ids and token_type_ids
    # output is embedding_output, B,T,d
    """
    embedding_output = model.module.embeddings(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
    )
    """
    input_shape = input_ids.size() # B,T
    seq_length = input_shape[1] # T

    inputs_embeds = model.module.embeddings.word_embeddings(input_ids) # B,T,d
    out_dict['inputs_embeds'] = inputs_embeds.detach().clone()
    token_type_embeddings = model.module.embeddings.token_type_embeddings(token_type_ids) # B,T,d
    out_dict['token_type_embeddings'] = token_type_embeddings.detach().clone()
    position_ids = model.module.embeddings.position_ids[:, 0: seq_length] # 1,T (0, 1, 2.... T-1)
    #out_dict['position_ids'] = position_ids.detach().clone()
    position_embeddings = model.module.embeddings.position_embeddings(position_ids) # 1,T,d
    out_dict['position_embeddings'] = position_embeddings.detach().clone()

    embeddings = inputs_embeds + token_type_embeddings
    embeddings += position_embeddings
    embeddings = model.module.embeddings.LayerNorm(embeddings)
    embedding_output = model.module.embeddings.dropout(embeddings) # B,T,d

    out_dict['embedding_output'] = embedding_output.detach().clone()

    # 2. Bert
    # use_sdpa_attention_masks = False
    # attention_mask from B,T -> B,1,1,T
    # encoder_extended_attention_mask = None
    # head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers), gives list with len(num_heads) times None

    """
    encoder_outputs = model.module.encoder(
        embedding_output,
        attention_mask=extended_attention_mask,
        head_mask=[None] * 12,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    """

    hidden_states = embedding_output
    #extended_attention_mask = model.module.get_extended_attention_mask(attention_mask, input_shape)

    dtype = embedding_output.dtype
    extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min

    for i, layer_module in enumerate(model.module.encoder.layer):
        """
        layer_outputs = layer_module(
            hidden_states=hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
        )
        hidden_states = layer_outputs[0] # B,T,d
        """

        # 2.1. Self-Attention
        """
        self_outputs = layer_module.attention.self(hidden_states=hidden_states,
                                                        attention_mask=extended_attention_mask,
                                                        head_mask=None,
                                                        encoder_hidden_states=None,
                                                        encoder_attention_mask=None,
                                                        output_attentions=False,
                                                        past_key_value=None)
        """

        query_layer = transpose_for_scores(layer_module.attention.self.query(hidden_states), num_attention_heads, attention_head_size) # B,T,d -> B,H,T,d/h
        key_layer = transpose_for_scores(layer_module.attention.self.key(hidden_states), num_attention_heads, attention_head_size)
        value_layer = transpose_for_scores(layer_module.attention.self.value(hidden_states), num_attention_heads, attention_head_size)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(layer_module.attention.self.attention_head_size)
        attention_scores = attention_scores + extended_attention_mask # B,H,T,T
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1) # B,H,T,T
        attention_probs = layer_module.attention.self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer) # B,H,T,d/h
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # B,T,H,d/h
        new_context_layer_shape = context_layer.size()[:-2] + (layer_module.attention.self.all_head_size,) # B,T,d
        context_layer = context_layer.view(new_context_layer_shape) # B,T,d

        # Attention output
        attention_output = layer_module.attention.output.dense(context_layer)
        attention_output = layer_module.attention.output.dropout(attention_output)
        attention_output = layer_module.attention.output.LayerNorm(attention_output + hidden_states)

        out_dict[f'attention_output_{i}'] = attention_output.detach().clone()

        # 2.2. MLP
        intermediate_output = layer_module.intermediate.dense(attention_output)
        intermediate_output = layer_module.intermediate.intermediate_act_fn(intermediate_output)
        layer_output = layer_module.output.dense(intermediate_output)
        layer_output = layer_module.output.dropout(layer_output)
        hidden_states = layer_module.output.LayerNorm(layer_output + attention_output)

        out_dict[f'mlp_output_{i}'] = hidden_states.detach().clone()

    encoder_outputs = hidden_states

    # 3. Pooler
    #sequence_output = encoder_outputs[0]  # B,T,d
    #pooled_output = model.pooler(sequence_output) # B,d
    pooled_output = model.module.pooler.dense(encoder_outputs[:, 0])  # B,d
    pooled_output = model.module.pooler.activation(pooled_output)  # B,d

    out_dict[f'pooled_output'] = pooled_output.detach().clone()

    return out_dict


#############################################################

class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, y = inputs
        for module in self._modules.values():
            x = module(x, y)
        return x

class BERTMergerFull(nn.Module):
    """
    Used for merging BERTs.
    """
    def __init__(self, embeddings_merger, transformer_merger_layers, pooler_merger):

        super(BERTMergerFull, self).__init__()
        self.embeddings_merger = embeddings_merger
        self.transformer_merger_layers = SequentialEncoder(*transformer_merger_layers)
        self.pooler_merger = pooler_merger

    def forward(self, input_ids, attention_mask, token_type_ids):
        x = self.embeddings_merger(input_ids, token_type_ids)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, dtype=x.dtype)
        x = self.transformer_merger_layers(x, extended_attention_mask)
        x = self.pooler_merger(x)
        return x

    def forward_features(self, input_ids, attention_mask, token_type_ids):
        out_dict = self.embeddings_merger.forward_features(input_ids, token_type_ids)
        x = out_dict['embedding_output']
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, dtype=x.dtype)

        for i, layer in enumerate(self.transformer_merger_layers):
            curr_block_num = math.floor(i / 2)
            layer_name = 'attention' if i % 2 == 0 else 'mlp'
            x = layer(x, extended_attention_mask)
            out_dict[f'{layer_name}_output_{curr_block_num}'] = x.detach().clone()

        x = self.pooler_merger(x)
        out_dict['pooled_output'] = x.detach().clone()
        return out_dict

    def get_extended_attention_mask(self, attention_mask, dtype):
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask


    # Folding M,U matrices into the weights, and return the new weights
    def fold(self, device) -> Dict[str, torch.Tensor]:
        merged_weights_dict = self.embeddings_merger.fold(device)

        for i, merger in enumerate(self.transformer_merger_layers):
            curr_layer_num = math.floor(i / 2)
            curr_merged_weights_dict = merger.fold(device)
            for key in curr_merged_weights_dict.keys():
                merged_weights_dict["{}_{}".format(key, curr_layer_num)] = curr_merged_weights_dict[key]

        merged_weights_dict.update(self.pooler_merger.fold(device))

        return merged_weights_dict


    # Getting the U output
    def get_U(self, device, scales=None, norm_U_scale=True):
        return self.pooler_merger.get_U(device, scales)


#############################################################

class BERTEmbeddingsMerger(nn.Module):
    """
    Used for merging the embeddings of the BERT model.
    In the case where we all the merger blocks together.

    word_embeddings: List of the word embeddings. Shape of each element: [dict_size, emb]
    position_embeddings: List of the position embeddings. Shape of each element: [T, emb]
    token_type_embeddings: List of the token type embeddings. Shape of each element: [2, emb]
    LN_state_dict: The weights of the layer normalization of the input. For initializing.
    number_of_models: The number of models that will be merged.
    MU_type: The type of the M,U matrices.
    comp_factor: The merged model new hidden dimension will be emb * comp_factor.
    rank: The rank of the learned M,U matrix. -1 means full rank (emb).
    last_features_scale: Scaling the features of the last model, and fix it later in the folding.
    """

    def __init__(self,
                 word_embeddings: List[torch.Tensor],
                 position_embeddings: List[torch.Tensor],
                 token_type_embeddings: List[torch.Tensor],
                 LN_state_dict: Dict[str, torch.Tensor],
                 number_of_models: int,
                 MU_type: str,
                 config,
                 num_heads: int,
                 freeze_LN: bool = False,
                 use_dropout: bool = False,
                 comp_factor: float = 1,
                 rank: int = -1,
                 MU_init_method: str = 'first'):

        super(BERTEmbeddingsMerger, self).__init__()
        self.number_of_models = number_of_models
        self.comp_factor = comp_factor
        self.rank = rank
        self.num_heads = num_heads
        self.MU_type = MU_type.lower()
        self.MU_init_method = MU_init_method.lower()
        self.freeze_LN = freeze_LN
        self.use_dropout = use_dropout
        self.emb = word_embeddings[0].size(1)
        self.scale = self.emb ** -0.5
        self.new_emb = int(self.emb * self.comp_factor)

        # Verify the rank
        if self.rank != -1 and (self.rank > self.emb or self.rank <= 0):
            raise ValueError('The rank must be between 1 and the embedding dimension.')

        self.word_embeddings = self.create_embeddings(word_embeddings)  # [dict_size, num_models*emb]
        self.position_embeddings = self.create_embeddings(position_embeddings) # [T_max, num_models*emb]
        self.token_type_embeddings = self.create_embeddings(token_type_embeddings) # [2, num_models*emb]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

        # Learned parameters
        self.LN = torch.nn.LayerNorm(self.new_emb, eps=config.layer_norm_eps)
        if LN_state_dict is not None:
            self.LN.load_state_dict(LN_state_dict)

        if freeze_LN:
            for param in self.LN.parameters():
                param.requires_grad = False

        # [emb * num_models, new_emb]
        self.M = MMatrix(emb=self.emb, new_emb=self.new_emb, num_models=self.number_of_models, rank=self.rank,
                         MU_type=self.MU_type, MU_init_method=self.MU_init_method, scale=self.scale,
                         comp_factor=self.comp_factor, num_heads=self.num_heads)


    def create_embeddings(self, embeddings_list):
        # [num_emb, num_models*emb]
        new_embeddings = nn.Embedding(embeddings_list[0].size(0), embeddings_list[0].size(1)*self.number_of_models)

        for embedding in embeddings_list:
            embedding.requires_grad = False

        new_embeddings.weight = torch.nn.Parameter(torch.cat(embeddings_list, dim=1), requires_grad=False)
        return new_embeddings


    # input_ids, token_type_ids in shape [B, T]
    # output is embeddings [B,T,emb]
    def forward(self, input_ids, token_type_ids):
        input_shape = input_ids.size()  # B,T
        seq_length = input_shape[1]  # T

        inputs_embeds = self.word_embeddings(input_ids)  # B,T,num_models*d
        token_type_embeddings = self.token_type_embeddings(token_type_ids)  # B,T,num_models*d
        position_ids = self.position_ids[:, 0: seq_length]  # 1,T (0, 1, 2.... T-1)
        position_embeddings = self.position_embeddings(position_ids)  # 1,T,num_models*d

        embeddings = inputs_embeds + token_type_embeddings
        embeddings += position_embeddings
        embeddings = self.M(embeddings)  # [B , T, num_models*d] @ [num_models*d , d_new] = [B , T, d_new]
        embeddings = self.LN(embeddings)
        if self.use_dropout:
            embeddings = self.dropout(embeddings)  # B,T,d_new

        return embeddings


    def forward_features(self, input_ids, token_type_ids):
        out_dict = {}

        input_shape = input_ids.size()  # B,T
        seq_length = input_shape[1]  # T

        inputs_embeds = self.word_embeddings(input_ids)  # B,T,num_models*d
        out_dict['inputs_embeds'] = inputs_embeds.detach().clone()[:,:,:self.emb]
        token_type_embeddings = self.token_type_embeddings(token_type_ids)  # B,T,num_models*d
        out_dict['token_type_embeddings'] = token_type_embeddings.detach().clone()[:,:,:self.emb]
        position_ids = self.position_ids[:, 0: seq_length]  # 1,T (0, 1, 2.... T-1)
        #out_dict['position_ids'] = position_ids.detach().clone()
        position_embeddings = self.position_embeddings(position_ids)  # 1,T,num_models*d
        out_dict['position_embeddings'] = position_embeddings.detach().clone()[:,:,:self.emb]

        embeddings = inputs_embeds + token_type_embeddings
        embeddings += position_embeddings
        embeddings = self.M(embeddings)  # [B , T, num_models*d] @ [num_models*d , d_new] = [B , T, d_new]
        embeddings = self.LN(embeddings)
        if self.use_dropout:
            embeddings = self.dropout(embeddings)  # B,T,d_new
        out_dict['embedding_output'] = embeddings.detach().clone()

        return out_dict


    # Folding M,U matrices into the weights, and return the new weights
    def fold(self, device) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            M = self.M.create_M_func()

            word_embeddings = self.word_embeddings.weight @ M  # [dict_size, emb]
            position_embeddings = self.position_embeddings.weight @ M  # [T_max, emb]
            token_type_embeddings = self.token_type_embeddings.weight @ M  # [2, emb]

            merged_weights_dict = {}
            merged_weights_dict['word_embeddings'] = word_embeddings.clone().to(device)
            merged_weights_dict['position_embeddings'] = position_embeddings.clone().to(device)
            merged_weights_dict['token_type_embeddings'] = token_type_embeddings.clone().to(device)
            merged_weights_dict['LN_emb_weight'] = self.LN.weight.clone().to(device) # [new_emb]
            merged_weights_dict['LN_emb_bias'] = self.LN.bias.clone().to(device) # [new_emb]

            return merged_weights_dict


    def end_of_training(self):
        pass

#############################################################
class BERTAttentionMerger(nn.Module):
    """
    Used for merging the attention sub-block inside the transformer.

    key_weight_list: Tuple of the weights that project to the key dim. shape: [emb, emb]
    key_bias_list: Tuple of the bias that project to the key dim. shape: [emb]
    query_weight_list: Tuple of the weights that project to the query dim. shape: [emb, emb]
    query_bias_list: Tuple of the bias that project to the query dim. shape: [emb]
    value_weight_list: Tuple of the weights that project to the value dim. shape: [emb, emb]
    value_bias_list: Tuple of the bias that project to the value dim. shape: [emb]
    linear_weight_list: Tuple of the weights that project to the output dim. shape: [emb, emb]
    linear_bias_list: Tuple of the bias that project to the output dim. shape: [emb]
    LN_state_dict: The weights of the layer normalization of the attention. For initializing.
    number_of_models: The number of models that will be merged.
    MU_type: The type of the M,U matrices.
    num_heads: The number of heads in the multi-head attention.
    comp_factor: The merged model new hidden dimension will be emb * comp_factor.
    rank: The rank of the learned M,U matrix. -1 means full rank (emb).
    LN_att_state_dict: The weights of the layer normalization of the attention sub-block. For initializing.
    freeze_LN: Whether to freeze the layer normalization weights.
    init_method: The method for initializing the learned parameters M and U.
    """

    def __init__(self,
                 key_weight_list: List[torch.Tensor],
                 key_bias_list: List[torch.Tensor],
                 query_weight_list: List[torch.Tensor],
                 query_bias_list: List[torch.Tensor],
                 value_weight_list: List[torch.Tensor],
                 value_bias_list: List[torch.Tensor],
                 linear_weight_list: List[torch.Tensor],
                 linear_bias_list: List[torch.Tensor],
                 LN_att_state_dict: Dict[str, torch.Tensor],
                 number_of_models: int,
                 MU_type: str,
                 config,
                 num_heads: int,
                 use_dropout: bool = False,
                 comp_factor: float = 1,
                 rank: int = -1,
                 MU_init_method: str = 'random',
                 freeze_LN: bool = False,):

        super(BERTAttentionMerger, self).__init__()
        _, self.emb = key_weight_list[0].shape
        self.num_models = number_of_models
        self.num_heads = num_heads
        self.scale = self.emb ** -0.5
        self.comp_factor = comp_factor
        self.new_emb = int(self.emb * self.comp_factor)
        self.head_dim = self.new_emb // self.num_heads

        self.rank = rank
        self.MU_type = MU_type.lower()
        self.MU_init_method = MU_init_method.lower()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        assert self.num_heads * self.head_dim == self.new_emb, "The new embedding dimension must be divisible by the number of heads."
        if self.rank != -1 and (self.rank > self.emb or self.rank <= 0):
            raise ValueError('The rank must be between 1 and the embedding dimension.')

        # Saves and arrange the models weights
        for parameter_list in [key_weight_list, key_bias_list, query_weight_list, query_bias_list, value_weight_list,
                               value_bias_list, linear_weight_list, linear_bias_list]:
            for parameter in parameter_list:
                parameter.requires_grad = False

        self.key_weight = nn.Parameter(torch.stack(key_weight_list, dim=0), requires_grad=False)  # [num_models, emb, emb]
        self.key_bias = nn.Parameter(torch.cat(key_bias_list, dim=0), requires_grad=False)  # [num_models * emb]
        self.query_weight = nn.Parameter(torch.stack(query_weight_list, dim=0), requires_grad=False)  # [num_models, emb, emb]
        self.query_bias = nn.Parameter(torch.cat(query_bias_list, dim=0), requires_grad=False)  # [num_models * emb]
        self.value_weight = nn.Parameter(torch.stack(value_weight_list, dim=0), requires_grad=False)  # [num_models, emb, emb]
        self.value_bias = nn.Parameter(torch.cat(value_bias_list, dim=0), requires_grad=False)  # [num_models * emb]
        self.linear_weight = nn.Parameter(torch.stack(linear_weight_list, dim=0), requires_grad=False)  # [num_models, emb, emb]
        self.linear_bias = nn.Parameter(torch.cat(linear_bias_list, dim=0), requires_grad=False)  # [num_models * emb]

        # Learned parameters
        self.LN_att = torch.nn.LayerNorm(self.new_emb)
        if LN_att_state_dict is not None:
            if self.comp_factor != 1:
                raise Exception('Cannot initialize the attention the layer according the pre-trained model,'
                                ' when the comp_factor is not 1.')
            self.LN_att.load_state_dict(LN_att_state_dict)

        if freeze_LN:
            for param in self.LN_att.parameters():
                param.requires_grad = False

        # M, U matrices
        # U_att in shape [new_emb, emb * num_models]
        # M_q, M_k, M_v in shape [emb * num_models, new_emb]
        # U_o_1 in shape [new_emb, emb * num_models]
        # M_o_2 in shape [emb * num_models, new_emb]
        self.U_att = UMatrix(emb=self.emb, new_emb=self.new_emb, num_models=self.num_models, rank=self.rank,
                             MU_type=self.MU_type, MU_init_method=self.MU_init_method, scale=self.scale,
                             comp_factor=self.comp_factor, num_heads=self.num_heads)

        self.M_q = MMatrix(emb=self.emb, new_emb=self.new_emb, num_models=self.num_models,
                           rank=self.rank, MU_type=self.MU_type, MU_init_method=self.MU_init_method,
                           scale=self.scale, comp_factor=self.comp_factor, num_heads=self.num_heads)

        self.M_k = MMatrix(emb=self.emb, new_emb=self.new_emb, num_models=self.num_models,
                           rank=self.rank, MU_type=self.MU_type, MU_init_method=self.MU_init_method,
                           scale=self.scale, comp_factor=self.comp_factor, num_heads=self.num_heads)

        self.M_v = MMatrix(emb=self.emb, new_emb=self.new_emb, num_models=self.num_models,
                           rank=self.rank, MU_type=self.MU_type, MU_init_method=self.MU_init_method,
                           scale=self.scale, comp_factor=self.comp_factor, num_heads=self.num_heads)

        self.U_o_1 = UMatrix(emb=self.emb, new_emb=self.new_emb, num_models=self.num_models, rank=self.rank,
                             MU_type=self.MU_type, MU_init_method=self.MU_init_method, scale=self.scale,
                             comp_factor=self.comp_factor, num_heads=self.num_heads)

        self.M_o_2 = MMatrix(emb=self.emb, new_emb=self.new_emb, num_models=self.num_models,
                             rank=self.rank, MU_type=self.MU_type, MU_init_method=self.MU_init_method,
                             scale=self.scale, comp_factor=self.comp_factor, num_heads=self.num_heads)


    # inputs in shape [B, T, new_emb]
    # outputs in shape [B, T, new_emb]
    def forward(self, inputs, attention_mask):

        B, seq_length, _ = inputs.shape
        x = self.U_att(inputs) # [B, T, emb * num_models]

        ######## Self attention ########
        q = _bmm_linear(x, self.query_weight, self.query_bias)  # [B, T, num_models * emb]
        k = _bmm_linear(x, self.key_weight, self.key_bias)  # [B, T, num_models * emb]
        v = _bmm_linear(x, self.value_weight, self.value_bias)  # [B, T, num_models * emb]

        q = self.M_q(q)  # [B, T, num_models * emb] @ [num_models * emb, emb] = [B, T, new_emb]
        k = self.M_k(k)  # [B, T, num_models * emb] @ [num_models * emb, emb] = [B, T, new_emb]
        v = self.M_k(v)  # [B, T, num_models * emb] @ [num_models * emb, emb] = [B, T, new_emb]

        q = transpose_for_scores(q, self.num_heads, self.head_dim)  # [B,T,d] -> [B,H,T,d/h]
        k = transpose_for_scores(k, self.num_heads, self.head_dim)
        v = transpose_for_scores(v, self.num_heads, self.head_dim)

        attention_scores = torch.matmul(q, k.transpose(-1, -2)) # [B, H, T, T]
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        attention_scores = attention_scores + attention_mask  # [B,H,T,T]
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)  # B,H,T,T

        attn_output = torch.matmul(attention_probs, v)  # B,H,T,d/h
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # B,T,H,d/h
        new_attn_output_shape = attn_output.size()[:-2] + (self.new_emb,)  # B,T,d
        attn_output = attn_output.view(new_attn_output_shape)  # B,T,d

        ######## OutPut projection ########
        # [B,T,d] @ [d, num_models * emb] = [B,T, num_models * emb]
        attn_output = self.U_o_1(attn_output)
        attn_output = _bmm_linear(attn_output, self.linear_weight, self.linear_bias)  # [B, T, num_models * emb]
        attn_output = self.M_o_2(attn_output)  # [B, T, num_models * emb] @ [num_models * emb, new_emb] = [B, T, new_emb]

        if self.use_dropout:
            attn_output = self.dropout(attn_output)

        attn_output = self.LN_att(attn_output + inputs)  # [B, T, new_emb]

        return attn_output


    # Folding M,U matrices into the weights, and return the new weights
    def fold(self, device) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            U_att = self.U_att.create_U_func()
            M_q = self.M_q.create_M_func()
            M_k = self.M_k.create_M_func()
            M_v = self.M_v.create_M_func()
            U_o_1 = self.U_o_1.create_U_func()
            M_o_2 = self.M_o_2.create_M_func()

            # Create the new weights
            new_query_weight = bmm_folding_M_U_into_weight(W=self.query_weight, M=M_q, U=U_att)
            new_key_weight = bmm_folding_M_U_into_weight(W=self.key_weight, M=M_k, U=U_att)
            new_value_weight = bmm_folding_M_U_into_weight(W=self.value_weight, M=M_v, U=U_att)

            new_query_bias = self.query_bias @ M_q
            new_key_bias = self.key_bias @ M_k
            new_value_bias = self.value_bias @ M_v

            new_linear_weight = bmm_folding_M_U_into_weight(W=self.linear_weight, M=M_o_2, U=U_o_1)
            new_linear_bias = self.linear_bias @ M_o_2

            merged_weights_dict = {}
            merged_weights_dict['query_weight'] = new_query_weight.clone().to(device)
            merged_weights_dict['key_weight'] = new_key_weight.clone().to(device)
            merged_weights_dict['value_weight'] = new_value_weight.clone().to(device)
            merged_weights_dict['query_bias'] = new_query_bias.clone().to(device)
            merged_weights_dict['key_bias'] = new_key_bias.clone().to(device)
            merged_weights_dict['value_bias'] = new_value_bias.clone().to(device)
            merged_weights_dict['att_linear_weight'] = new_linear_weight.clone().to(device)
            merged_weights_dict['att_linear_bias'] = new_linear_bias.clone().to(device)
            merged_weights_dict['LN_att_weight'] = self.LN_att.weight.clone().to(device)
            merged_weights_dict['LN_att_bias'] = self.LN_att.bias.clone().to(device)

            return merged_weights_dict


#############################################################

class BERTMLPMerger(nn.Module):
    """
    Used for merging the MLP sub-block inside the transformer.
    Including layer norm; linear layer; activation; and residual connection.

    linear1_weight_list: Tuple of the weights that project to the hidden dim. shape: [hidden_dim, emb]
    linear1_bias_list: Tuple of the bias that project to the hidden dim. shape: [hidden_dim]
    linear2_weight_list: Tuple of the weights that project to the emb dim. shape: [emb, hidden_dim]
    linear2_bias_list: Tuple of the bias that project to the emb dim. shape: [emb]
    number_of_models: The number of models that will be merged.
    mlp_activation: The activation function of the MLP sub-block.
    number_of_models: The number of models that will be merged.
    comp_factor: The merged model new hidden dimension will be emb * comp_factor.
    rank: The rank of the learned M,U matrix. -1 means full rank (emb).
    MU_type: The type of the M,U matrices.
    LN_mlp_state_dict: The weights of the layer normalization of the mlp sub-block. For initializing.
    freeze_LN: Whether to freeze the layer normalization weights.
    init_method: The method for initializing the learned parameters M and U.
    last_layer: Whether this is the last layer of the transformer.
    last_features_scale: Scaling the features of the last model, and fix it later in the folding.
    """

    def __init__(self,
                 linear1_weight_list: List[torch.Tensor],
                 linear1_bias_list: List[torch.Tensor],
                 linear2_weight_list: List[torch.Tensor],
                 linear2_bias_list: List[torch.Tensor],
                 LN_mlp_state_dict: Dict[str, torch.Tensor],
                 config,
                 mlp_activation,
                 number_of_models: int,
                 num_heads: int,
                 MU_type: str,
                 comp_factor: float = 1,
                 rank: int = -1,
                 freeze_LN: bool = False,
                 use_dropout: bool = False,
                 MU_init_method: str = 'random'):

        super(BERTMLPMerger, self).__init__()
        self.hidden_dim, self.emb = linear1_weight_list[0].shape
        self.num_models = number_of_models
        self.num_heads = num_heads
        self.scale = self.emb ** -0.5
        self.scale_hidden = self.hidden_dim ** -0.5
        self.comp_factor = comp_factor
        self.new_emb = int(self.emb * self.comp_factor)
        self.new_hid_dim = int(self.hidden_dim * self.comp_factor)
        self.rank = rank
        self.use_dropout = use_dropout
        self.MU_type = MU_type.lower()
        self.hidden_rank = int(self.rank * self.hidden_dim / self.emb)
        self.MU_init_method = MU_init_method.lower()
        self.config = config

        self.activation = mlp_activation
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Saves and arrange the models weights
        for parameter_list in [linear1_weight_list, linear1_bias_list, linear2_weight_list, linear2_bias_list]:
            for parameter in parameter_list:
                parameter.requires_grad = False

        linear1_weight = torch.stack(linear1_weight_list, dim=0)  # [num_models, hidden_dim, emb]
        linear1_bias = torch.cat(linear1_bias_list, dim=0)  # [num_models * hidden_dim]
        linear2_weight = torch.stack(linear2_weight_list, dim=0)  # [num_models, emb, hidden_dim]
        linear2_bias = torch.cat(linear2_bias_list, dim=0)  # [num_models * emb]
        self.linear1_weight = nn.Parameter(linear1_weight, requires_grad=False)
        self.linear1_bias = nn.Parameter(linear1_bias, requires_grad=False)
        self.linear2_weight = nn.Parameter(linear2_weight, requires_grad=False)
        self.linear2_bias = nn.Parameter(linear2_bias, requires_grad=False)

        # Learned parameters
        self.LN_mlp = torch.nn.LayerNorm(self.new_emb)
        if LN_mlp_state_dict is not None:
            if self.comp_factor != 1:
                raise Exception('Cannot initialize the attention the layer according the pre-trained model,'
                                ' when the comp_factor is not 1.')
            self.LN_mlp.load_state_dict(LN_mlp_state_dict)

        if freeze_LN:
            for param in self.LN_mlp.parameters():
                param.requires_grad = False

        # [new_emb, emb * num_models]
        self.U_0 = UMatrix(emb=self.emb, new_emb=self.new_emb, num_models=self.num_models, rank=self.rank,
                           MU_type=self.MU_type, MU_init_method=self.MU_init_method, scale=self.scale,
                           comp_factor=self.comp_factor, num_heads=self.num_heads)

        # [hidden_dim * num_models, new_hid_dim]
        self.M_1 = MMatrix(emb=self.hidden_dim, new_emb=self.new_hid_dim, num_models=self.num_models,
                           rank=self.hidden_rank, MU_type=self.MU_type, MU_init_method=self.MU_init_method,
                           scale=self.scale_hidden, comp_factor=self.comp_factor, num_heads=self.num_heads)

        # [new_hid_dim, hidden_dim * num_models]
        self.U_1 = UMatrix(emb=self.hidden_dim, new_emb=self.new_hid_dim, num_models=self.num_models,
                           rank=self.hidden_rank, MU_type=self.MU_type, MU_init_method=self.MU_init_method,
                           scale=self.scale_hidden, comp_factor=self.comp_factor, num_heads=self.num_heads)

        # [emb * num_models, new_emb]
        self.M_2 = MMatrix(emb=self.emb, new_emb=self.new_emb, num_models=self.num_models,
                           rank=self.rank, MU_type=self.MU_type, MU_init_method=self.MU_init_method,
                           scale=self.scale, comp_factor=self.comp_factor, num_heads=self.num_heads)


    # inputs in shape [B, T, new_emb]
    # outputs in shape [B, T, new_emb]
    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        y = self.U_0(inputs)
        # [B, T, new_emb] @ [new_emb, emb * num_models] = [B, T, emb * num_models]

        ######## First linear layer ########
        y = _bmm_linear(y, self.linear1_weight, self.linear1_bias)
        # [B, T, num_models * emb] @ [emb * num_models, num_models * hid_dim] = [B, T, num_models * hid_dim]

        y = self.M_1(y)
        # [B, T, num_models * hid_dim] @ [num_models * hid_dim, new_hid_dim] = [B, T, new_hid_dim]
        y = self.activation(y)
        y = self.U_1(y)
        # [B, T, new_hid_dim] @ [new_hid_dim, num_models * hid_dim] = [B, T, num_models * hid_dim]

        ######## Second linear layer ########
        y = _bmm_linear(y, self.linear2_weight, self.linear2_bias)
        # [B, T, num_models * hid_dim] @ [num_models * hid_dim, num_models * emb] = [B, T, num_models * emb]

        y = self.M_2(y)  # [B, T, num_models * emb] @ [num_models * emb, new_emb] = [B, T, new_emb]
        if self.use_dropout:
            y = self.dropout(y)

        output = self.LN_mlp(y + inputs)  # [B, T, new_emb]
        return output


    # Folding M,U matrices into the weights, and return the new weights
    def fold(self, device) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            # Create the M and U matrices
            U_0 = self.U_0.create_U_func()
            M_1 = self.M_1.create_M_func()
            U_1 = self.U_1.create_U_func()
            M_2 = self.M_2.create_M_func()

            # Create the new weights
            new_linear1_weight = bmm_folding_M_U_into_weight(W=self.linear1_weight, M=M_1, U=U_0)
            # [new_hid_dim, new_emb]

            new_linear1_bias = self.linear1_bias @ M_1
            # [num_models * hid_dim] @ [num_models * hid_dim, new_hid_dim] = [new_hid_dim]

            new_linear2_weight = bmm_folding_M_U_into_weight(W=self.linear2_weight, M=M_2, U=U_1)
            # [new_emb, new_hid_dim]

            new_linear2_bias = self.linear2_bias @ M_2
            # [num_models * emb] @ [num_models * emb, new_hid_dim] = [new_emb]

            merged_weights_dict = {}
            merged_weights_dict['linear1_weight'] = new_linear1_weight.clone().to(device)
            merged_weights_dict['linear1_bias'] = new_linear1_bias.clone().to(device)
            merged_weights_dict['linear2_weight'] = new_linear2_weight.clone().to(device)
            merged_weights_dict['linear2_bias'] = new_linear2_bias.clone().to(device)
            merged_weights_dict['LN_mlp_weight'] = self.LN_mlp.weight.clone().to(device)
            merged_weights_dict['LN_mlp_bias'] = self.LN_mlp.bias.clone().to(device)

            return merged_weights_dict



#############################################################

class BERTPoolerMerger(nn.Module):
    """
    Used for merging the pooler of the BERT model.
    In the case where we all the merger blocks together.

    linear_weight_list: Tuple of the weights that project to the output dim. shape: [output_dim, emb]
    linear_bias_list: Tuple of the bias that project to the hidden dim. shape: [output_dim]
    number_of_models: The number of models that will be merged.
    MU_type: The type of the M,U matrices.
    comp_factor: The merged model new hidden dimension will be emb * comp_factor.
    rank: The rank of the learned M,U matrix. -1 means full rank (emb).
    last_features_scale: Scaling the features of the last model, and fix it later in the folding.
    """

    def __init__(self,
                 linear_weight_list: List[torch.Tensor],
                 linear_bias_list: List[torch.Tensor],
                 activation,
                 number_of_models: int,
                 num_heads: int,
                 MU_type: str,
                 comp_factor: float = 1,
                 rank: int = -1,
                 MU_init_method: str = 'first'):

        super(BERTPoolerMerger, self).__init__()
        self.number_of_models = number_of_models
        self.comp_factor = comp_factor
        self.rank = rank
        self.num_heads = num_heads
        self.MU_type = MU_type.lower()
        self.MU_init_method = MU_init_method.lower()
        self.out_dim, self.emb = linear_weight_list[0].shape
        self.scale = self.emb ** -0.5
        self.scale_out = self.out_dim ** -0.5
        self.new_emb = int(self.emb * self.comp_factor)
        self.activation = activation

        # Verify the rank
        if self.rank != -1 and (self.rank > self.emb or self.rank <= 0):
            raise ValueError('The rank must be between 1 and the embedding dimension.')

        # Saves and arrange the models weights
        for parameter_list in [linear_weight_list, linear_bias_list]:
            for parameter in parameter_list:
                parameter.requires_grad = False

        linear_weight = torch.stack(linear_weight_list, dim=0)  # [num_models, out_dim, emb]
        linear_bias = torch.cat(linear_bias_list, dim=0)  # [num_models * out_dim]
        self.linear_weight = nn.Parameter(linear_weight, requires_grad=False)
        self.linear_bias = nn.Parameter(linear_bias, requires_grad=False)

        # [new_emb, emb * num_models]
        self.U = UMatrix(emb=self.emb, new_emb=self.new_emb, num_models=self.number_of_models, rank=self.rank,
                         MU_type=self.MU_type, MU_init_method=self.MU_init_method, scale=self.scale,
                         comp_factor=self.comp_factor, num_heads=self.num_heads)

        # [num_models * out_dim, out_dim]
        self.M = MMatrix(emb=self.out_dim, new_emb=self.out_dim, num_models=self.number_of_models, rank=self.rank,
                         MU_type=self.MU_type, MU_init_method=self.MU_init_method, scale=self.scale_out,
                         comp_factor=self.comp_factor, num_heads=self.num_heads)

        # [out_dim, out_dim * num_models]
        self.U_output = UMatrix(emb=self.out_dim, new_emb=self.out_dim, num_models=self.number_of_models, rank=self.rank,
                                MU_type=self.MU_type, MU_init_method=self.MU_init_method, scale=self.scale_out,
                                comp_factor=self.comp_factor, num_heads=self.num_heads)


    # inputs in shape [B, T, new_emb]
    # outputs in shape [B, num_models * out_dim]
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.U(inputs[:, 0, :])
        # [B, new_emb] @ [new_emb, emb * num_models] = [B, emb * num_models]

        x = _bmm_linear_output(x, self.linear_weight, self.linear_bias)
        # [B, emb * num_models] @ [emb * num_models, out_dim * num_models] = [B, out_dim * num_models]

        x = self.M(x)  # [B, num_models * out_dim] @ [num_models * out_dim, out_dim] = [B, out_dim]
        x = self.activation(x)
        x = self.U_output(x)  # [B, out_dim] @ [out_dim, num_models * out_dim] = [B, num_models * out_dim]
        return x


    def create_MU(self):
        U = self.U.create_U_func()
        M = self.M.create_M_func()
        U_output = self.U_output.create_U_func()
        return U, M, U_output


    # Folding M,U matrices into the weights, and return the new weights
    def fold(self, device) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            # Create the M and U matrices
            U, M, U_output = self.create_MU()

            # Create the new weights
            new_linear_weight = bmm_folding_M_U_into_weight(W=self.linear_weight, M=M, U=U) # [new_emb, out_dim]
            new_linear_bias = self.linear_bias @ M # [out_dim]

            merged_weights_dict = {}
            merged_weights_dict['pooler_linear_weight'] = new_linear_weight.clone().to(device)
            merged_weights_dict['pooler_linear_bias'] = new_linear_bias.clone().to(device)
            merged_weights_dict['U_output'] = U_output.clone().to(device)

            return merged_weights_dict


    # Getting the U output of this layer
    def get_U(self, device, scales=None, norm_U_scale=True):
        with torch.no_grad():
            _, _, U_output = self.create_MU()

            # Normalize the U_output
            if scales != None and norm_U_scale:
                print("Use scales at the end of training: ", scales)
                for i, scale in enumerate(scales):
                    U_output[:, i * self.out_dim: (i + 1) * self.out_dim] /= scale
            else:
                print("Won't change the scales of the U_output at the end of training")

        return U_output.detach().clone().to(device)

