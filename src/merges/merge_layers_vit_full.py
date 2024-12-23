from typing import List, Dict

import torch
import torch.nn as nn

import math

# input: [batch_size, input_dim]
# weight: [output_dim, input_dim]
# bias: [output_dim]
def _linear(input, weight, bias=0):
    return input @ weight.T + bias


"""
# input: [seq_length, B, emb * num_models]
# weight: [num_models, output_dim, emb]
# bias: [output_dim * num_models]
def _bmm_linear(input, weight, bias=0):
    seq_length, B, = input.shape[0], input.shape[1]
    num_models, output_dim, emb = weight.shape
    input = input.view(num_models, seq_length * B, emb) # [num_models, seq_length * B, emb]
    weight = weight.permute(0, 2, 1) # [num_models, output_dim, emb] -> [num_models, emb, output_dim]
    output = torch.bmm(input, weight) # [num_models, seq_length * B, output_dim]
    return output.view(seq_length, B, num_models * output_dim) + bias # [seq_length, B, num_models * output_dim]

# input: [seq_length, B, emb * num_models]
# weight: [num_models, output_dim, emb]
# bias: [output_dim * num_models]
def _bmm_linear(input, weight, bias=0):
    num_models, output_dim, emb = weight.shape
    output = torch.zeros(0, device=input.device)
    for i in range(num_models):
        output = torch.cat((output, _linear(input[:, :, i * emb:(i + 1) * emb], weight[i])), dim=2)
    return output + bias  # [seq_length, B, num_models * output_dim]
"""

def _bmm_linear(input, weight, bias=0):
    num_models, output_dim, emb = weight.shape
    input_reshaped = input.view(-1, num_models, emb)
    output = torch.einsum('bme,moe->bmo', input_reshaped, weight)
    output = output.reshape(-1, input.shape[1], num_models * output_dim)
    output += bias
    return output


# input: [B, emb * num_models]
# weight: [num_models, output_dim, emb]
# bias: [output_dim * num_models]
def _bmm_linear_output(input, weight, bias=0):
    num_models, output_dim, emb = weight.shape
    output = torch.zeros(0, device=input.device)
    for i in range(num_models):
        output = torch.cat((output, _linear(input[:, i * emb:(i + 1) * emb], weight[i])), dim=-1)
    return output + bias  # [seq_length, B, num_models * output_dim]


# Folding M,U matrices into a linear layer weights.
# M: [output_dim * num_models, output_dim]
# U: [input_dim, input_dim * num_models]
# weight_list: List of the weights of the linear layers. shape: [output_dim, input_dim]
# bias_list: List of the biases of the linear layers. shape: [output_dim]
# Reminder: layer = input @ weight.T + bias
def fold_M_U_into_linear(M, U, weight_list, bias_list=None):
    input_dim, _ = U.shape
    _, output_dim = M.shape
    assert input_dim == weight_list[0].shape[1], 'The input dimension of the linear layers must be the same.'
    assert output_dim == weight_list[0].shape[0], 'The output dimension of the linear layers must be the same.'

    W_merged = torch.zeros((output_dim, input_dim))
    for i, weight in enumerate(weight_list):
        W_merged += M[i * output_dim: (i + 1) * output_dim, :].T @ weight @ U[:, i * input_dim: (i + 1) * input_dim].T
        # [output_dim, output_dim] @ [output_dim, input_dim] @ [input_dim, input_dim] = [output_dim, input_dim]

    if bias_list is not None:
        bias_merged = torch.zeros(output_dim)
        for i, bias in enumerate(bias_list):
            bias_merged += bias @ M[i * output_dim: (i + 1) * output_dim, :]
            # [output_dim] @ [output_dim, output_dim] = [output_dim]
        return W_merged, bias_merged

    return W_merged


def custom_block_diag(tensors):
    """
    Create a block diagonal matrix from the provided tensors.
    """
    # Calculate the total shape of the resulting matrix
    total_rows = sum([t.size(0) for t in tensors])
    total_cols = sum([t.size(1) for t in tensors])
    # Initialize an empty matrix of the required shape
    block_matrix = torch.zeros(total_rows, total_cols, dtype=tensors[0].dtype, device=tensors[0].device)
    # Fill the matrix
    row_start = 0
    col_start = 0
    for t in tensors:
        row_end = row_start + t.size(0)
        col_end = col_start + t.size(1)
        block_matrix[row_start:row_end, col_start:col_end] = t
        row_start = row_end
        col_start = col_end

    return block_matrix


def folding_M_U_into_weight(W, M, U):
    W_new = U @ W.T @ M
    # [new_emb, num_models * emb] @ [num_models * emb, num_models * emb] @ [num_models * emb, new_emb]
    # = [new_emb, new_emb]
    return W_new.T


# M: [num_models * emb, new_emb]
# U: [new_emb, num_models * emb]
# weight: [num_models, emb, emb]
def bmm_folding_M_U_into_weight(W, M, U):
    W = torch.chunk(W, W.shape[0], dim=0)
    W = [w.squeeze() for w in W]
    W = custom_block_diag(W)  # [num_models * emb, num_models * emb]
    W_new = U @ W.T @ M
    # [new_emb, num_models * emb] @ [num_models * emb, num_models * emb] @ [num_models * emb, new_emb]
    # = [new_emb, new_emb]
    return W_new.T


def create_concatenated_diagonal_matrices(vec, emb, num_models, dim_cat):
    """
    Transforms a vector into a concatenated diagonal matrix.

    Args:
    vec (torch.Tensor): A PyTorch vector of length emb*num_models.
    emb (int): The size of each segment in the vector, and one dimension of the square diagonal matrices.
    num_models (int): The number of segments to split the vector into.
    dim_cat (int): The dimension to concatenate the diagonal matrices on.

    Returns:
    torch.Tensor: A matrix of shape (emb*num_models, emb) formed by concatenating diagonal matrices.
    """

    # Reshape the vector into num_models segments, each of length emb
    split_vec = vec.view(num_models, emb)

    # Create diagonal matrices for each segment and concatenate them
    diagonal_matrices = [torch.diag(v) for v in split_vec]
    result = torch.cat(diagonal_matrices, dim=dim_cat)

    return result



LOW_RANK_FACTOR = 0.1


#############################################################

class VITMergerFull(nn.Module):
    """
    Used for merging VITs.
    """

    def __init__(self, pre_processing_merger, transformer_merger_layers, output_merger,
                 loss_type, loss_layer_num):

        super(VITMergerFull, self).__init__()
        self.pre_processing_merger = pre_processing_merger
        self.transformer_merger_layers = nn.Sequential(*transformer_merger_layers)
        self.output_merger = output_merger

        self.loss_type = loss_type
        self.loss_layer_num = loss_layer_num
        self.with_u = True  # Use the U matrix in the output layer

    def forward(self, x):
        if self.loss_type not in ['rec_with_inner_att', 'rec_with_inner_mlp', 'rec_with_inner_att_ids',
                                  'rec_with_inner_mlp_ids']:
            return self.regular_forward(x)
        else:
            return self.forward_inner_features(x)

    def regular_forward(self, x):
        x = self.pre_processing_merger(x)
        x = self.transformer_merger_layers(x)
        x = self.output_merger(x, with_u=self.with_u)
        return x

    def forward_inner_features(self, x):
        inner_features_dict = {}
        x = self.pre_processing_merger(x)

        for i, layer in enumerate(self.transformer_merger_layers):
            layer_num = math.floor(i / 2)
            layer_name = 'att' if i % 2 == 0 else 'mlp'

            if layer_num in self.loss_layer_num and layer_name == 'att' and self.loss_type in ['rec_with_inner_att',
                                                                                               'rec_with_inner_att_ids']:
                x, inner = layer(x, for_features=True)
                # Adding attention features
                inner_features_dict[f'inner_{layer_num}_{layer_name}'] = inner.permute(1, 0,
                                                                                       2)  # [batch, seq_len, num_models*emb]

            elif layer_num in self.loss_layer_num and layer_name == 'mlp' and self.loss_type in ['rec_with_inner_mlp',
                                                                                                 'rec_with_inner_mlp_ids']:
                x, inner = layer(x, for_features=True)
                # Adding mlp features
                inner_features_dict[f'inner_{layer_num}_{layer_name}'] = inner.permute(1, 0,
                                                                                       2)  # [batch, seq_len, num_models*emb]

            else:
                x = layer(x, for_features=False)

        x = self.output_merger(x)
        return x, inner_features_dict

    def forward_features(self, x):
        dict = {}
        x = self.pre_processing_merger(x)
        dict['ln_pre'] = x.clone().detach().cpu().permute(1, 0, 2)

        for i, layer in enumerate(self.transformer_merger_layers):
            layer_num = math.floor(i / 2)
            layer_name = 'attn_out' if i % 2 == 0 else 'mlp_out'
            x, dict_mlp = layer(x, features=True)
            for key in dict_mlp.keys():
                dict['{}-{}'.format(key, layer_num)] = dict_mlp[key]

        x = self.output_merger(x)
        dict['output'] = x.clone().detach().cpu()
        return dict

    # Folding M,U matrices into the weights, and return the new weights
    def fold(self, device) -> Dict[str, torch.Tensor]:
        merged_weights_dict = self.pre_processing_merger.fold(device)

        for i, merger in enumerate(self.transformer_merger_layers):
            curr_layer_num = math.floor(i / 2)
            curr_merged_weights_dict = merger.fold(device)
            for key in curr_merged_weights_dict.keys():
                merged_weights_dict["{}_{}".format(key, curr_layer_num)] = curr_merged_weights_dict[key]

        curr_merged_weights_dict = self.output_merger.fold(device)
        merged_weights_dict.update(curr_merged_weights_dict)

        return merged_weights_dict

    # Getting the U output
    def get_U(self, device, scales=None, norm_U_scale=True):
        return self.output_merger.get_U(device, scales)


#############################################################

class VITMergerBlock(nn.Module):
    """
    Used to merge a VIT block. This can include the pre-processing layer and a few transformer layers.
    At the end, there will always be a special merge layer: output merger or LayerNormUBlock merger.
    merger_layers: list of merger layers.
    start_layer_id: the layer id of the first layer in the block.
    end_layer_id: the layer id of the last layer in the block.
    num_layers: the total number of layers in the VIT.
    """

    def __init__(self, merger_layers, start_layer_id, end_layer_id, num_layers):

        super(VITMergerBlock, self).__init__()
        self.start_layer_id = start_layer_id
        self.end_layer_id = end_layer_id
        self.num_layers = num_layers
        self.merger_layers = nn.Sequential(*merger_layers)

    # If self.start_layer_id != 0, it isn't a pre-processing layer,
    # so x should be in shape [B, seq_length, emb*num_models]
    def forward(self, x, for_features=False, two_outputs=False):
        if self.start_layer_id != 0:
            x = x.permute(1, 0, 2)  # [B, seq_length, emb*num_models] -> [seq_length, B, emb*num_models]

        # x = self.merger_layers(x)
        for i, layer in enumerate(self.merger_layers):
            if i == len(self.merger_layers) - 1 and two_outputs:
                y = x.clone().permute(1, 0, 2)  # [seq_length, B, emb] -> [B, seq_length, num_models]
            x = layer(x, for_features=for_features)

        if not two_outputs:
            return x
        else:
            return x, y

    # Folding M,U matrices into the weights, and return the new weights
    def fold(self, device) -> Dict[str, torch.Tensor]:
        merged_weights_dict = {}

        # The last layer is always special. It is either the output layer or a LayerNormUBlock.
        transformer_merger_layers = self.merger_layers[:-1]

        if self.start_layer_id == 0:
            merged_weights_dict = self.merger_layers[0].fold(device)  # pre-processing layer
            transformer_merger_layers = transformer_merger_layers[1:]

        # Folding the weights of the transformers layers
        for i, merger in enumerate(transformer_merger_layers):
            counter = math.floor(i / 2)  # Because a transformer layer has two sub-layers, attention and MLP.
            curr_layer_num = self.start_layer_id + counter
            curr_merged_weights_dict = merger.fold(device)
            for key in curr_merged_weights_dict.keys():
                merged_weights_dict["{}_{}".format(key, curr_layer_num)] = curr_merged_weights_dict[key]

        if self.end_layer_id == self.num_layers:
            # Folding the last layer it is an output layer.
            curr_merged_weights_dict = self.merger_layers[-1].fold(device)
            merged_weights_dict.update(curr_merged_weights_dict)

        return merged_weights_dict

    # Getting the last U
    def get_U(self, device):
        return self.merger_layers[-1].get_U(device)

    # Getting the last U, LayerNorm
    def get_U_LN(self):
        return self.merger_layers[-1].get_U_LN()

    def end_of_training(self):
        return

#############################################################

class MMatrix(nn.Module):

    def __init__(self,
                 emb: int,
                 new_emb: int,
                 num_models: int,
                 rank: int,
                 MU_type: str,
                 MU_init_method: str,
                 num_heads: int = 1,
                 scale: float = 1,
                 comp_factor: float = 1,):

        super(MMatrix, self).__init__()
        self.emb = emb
        self.new_emb = new_emb
        self.num_models = num_models
        self.rank = rank
        self.num_heads = num_heads
        self.head_dim = self.emb // self.num_heads
        self.MU_type = MU_type
        self.MU_init_method = MU_init_method
        self.scale = scale
        self.comp_factor = comp_factor

        self.initialize()

    def initialize(self):
        if self.MU_init_method not in ['random', 'average', 'first']:
            raise ValueError('Unknown MU_init_method: {}'.format(self.MU_init_method))

        #######################################
        if self.MU_type == 'diagonal':
            if self.MU_init_method == 'random':
                self.M_diag = nn.Parameter(self.scale * torch.randn(self.emb * self.num_models) / math.sqrt(self.num_models))
            elif self.MU_init_method == 'average':
                self.M_diag = nn.Parameter(torch.ones(self.emb * self.num_models) / self.num_models)
            elif self.MU_init_method == 'first':
                self.M_diag = nn.Parameter(torch.cat([torch.ones(self.emb), torch.zeros(self.emb * (self.num_models - 1))]))

        #######################################
        elif self.MU_type == 'low_rank':
            if self.MU_init_method in ['average', 'first']:
                raise Exception(f"MU_init_method = {self.MU_init_method} is unsupported for low_rank")
            self.M_a = nn.Parameter(self.scale * torch.randn((self.emb * self.num_models, self.rank)) / math.sqrt(self.rank))
            self.M_b = nn.Parameter(self.scale * torch.randn((self.rank, self.new_emb)) / math.sqrt(self.rank))

        #######################################
        elif self.MU_type == 'diagonal_and_low_rank':
            if self.MU_init_method == 'random':
                self.M_diag = nn.Parameter(torch.zeros(self.emb * self.num_models))
                self.M_a = nn.Parameter(self.scale * torch.randn((self.emb * self.num_models, self.rank)) / math.sqrt(self.rank))
                self.M_b = nn.Parameter(self.scale * torch.randn((self.rank, self.new_emb)) / math.sqrt(self.rank))

            elif self.MU_init_method == 'average':
                self.M_diag = nn.Parameter(torch.ones(self.emb * self.num_models) / self.num_models)
                self.M_a = nn.Parameter(LOW_RANK_FACTOR * self.scale * torch.zeros((self.emb * self.num_models, self.rank)) / self.rank)
                self.M_b = nn.Parameter(LOW_RANK_FACTOR * self.scale * torch.randn((self.rank, self.new_emb)) / self.rank)

            elif self.MU_init_method == 'first':
                self.M_diag = nn.Parameter(torch.cat([torch.ones(self.emb), torch.zeros(self.emb * (self.num_models - 1))]))
                self.M_a = nn.Parameter(LOW_RANK_FACTOR * self.scale * torch.zeros((self.emb * self.num_models, self.rank)) / self.rank)
                self.M_b = nn.Parameter(LOW_RANK_FACTOR * self.scale * torch.randn((self.rank, self.new_emb)) / self.rank)

        #######################################
        elif self.MU_type == 'full':
            if self.MU_init_method == 'random':
                self.M = nn.Parameter(self.scale * torch.randn((self.emb * self.num_models, self.new_emb)) / self.num_models)
            elif self.MU_init_method == 'average':
                self.M = nn.Parameter(torch.eye(self.emb).repeat(self.num_models, 1) / self.num_models)

        #######################################
        elif self.MU_type == 'att_heads':
            if self.MU_init_method == 'first':
                # M1
                M_1 = torch.zeros(self.num_models, self.num_heads, self.head_dim, self.head_dim)
                for i in range(self.num_heads):
                    M_1[0, i] = torch.eye(self.head_dim)
                self.M_1 = nn.Parameter(M_1)

                # M2
                identity_matrix = torch.eye(self.num_heads)
                M_2 = identity_matrix.unsqueeze(0).expand(self.num_models, -1, -1)
                self.M_2 = nn.Parameter(M_2)

            else:
                raise Exception(f"MU_init_method = {self.MU_init_method} is unsupported for att_heads")
        #######################################

        else:
            raise Exception("Unsupported MU_type")


    def create_M_func(self):
        if self.MU_type == 'diagonal':
            return create_concatenated_diagonal_matrices(vec=self.M_diag, emb=self.emb, num_models=self.num_models, dim_cat=0)

        elif self.MU_type == 'low_rank':
            return self.M_a @ self.M_b

        elif self.MU_type == 'diagonal_and_low_rank':
            M = create_concatenated_diagonal_matrices(vec=self.M_diag, emb=self.emb, num_models=self.num_models, dim_cat=0)
            return M + self.M_a @ self.M_b

        elif self.MU_type == 'full':
            return self.M

        elif self.MU_type == 'att_heads':
            m1_mat = torch.zeros(self.num_models * self.num_heads * self.head_dim, self.num_models * self.num_heads * self.head_dim)
            for model in range(self.num_models):
                for head in range(self.num_heads):
                    start_idx = (model * self.num_heads + head) * self.head_dim
                    end_idx = start_idx + self.head_dim
                    m1_mat[start_idx:end_idx, start_idx:end_idx] = self.M_1[model, head]

            m2_mat = torch.zeros(self.num_models * self.num_heads * self.head_dim, self.num_heads * self.head_dim)
            for model in range(self.num_models):
                for head_in in range(self.num_heads):
                    for head_out in range(self.num_heads):
                        idx1 = (model * self.num_heads + head_in) * self.head_dim
                        idx2 = head_out * self.head_dim
                        diagonal_values = torch.full((self.head_dim,), self.M_2[model, head_in, head_out].item())
                        m2_mat[idx1:idx1 + self.head_dim, idx2:idx2 + self.head_dim] = torch.diag(diagonal_values)

            M = (m1_mat @ m2_mat).to('cuda')
            return M


    # X in shape [B, T, emb * models]
    # Output in shape [B, T, new_emb]
    def forward(self, x):
        if self.MU_type != 'att_heads':
            M = self.create_M_func()
            return x @ M

        else:
            B, T  = x.shape[0], x.shape[1]
            if len(x.shape) == 3:
                x = x.reshape(B, T, self.num_models, self.emb).reshape(B, T, self.num_models, self.num_heads, self.head_dim)
                x = torch.einsum('btmnh,mnhk->btmnk', x, self.M_1)
                x = torch.einsum('btmnh,mnk->btkh', x, self.M_2)
                x = x.reshape(B, T, self.emb)
                return x
            elif len(x.shape) != 3:  # x is [B, emb]
                x = x.reshape(B, self.num_models, self.emb).reshape(B, self.num_models, self.num_heads, self.head_dim)
                x = torch.einsum('bmnh,mnhk->bmnk', x, self.M_1)
                x = torch.einsum('bmnh,mnk->bkh', x, self.M_2)
                x = x.reshape(B, self.emb)
                return x




class UMatrix(nn.Module):

    def __init__(self,
                 emb: int,
                 new_emb: int,
                 num_models: int,
                 rank: int,
                 MU_type: str,
                 MU_init_method: str,
                 num_heads: int = 1,
                 scale: float = 1,
                 comp_factor: float = 1,):

        super(UMatrix, self).__init__()
        self.emb = emb
        self.new_emb = new_emb
        self.num_models = num_models
        self.rank = rank
        self.num_heads = num_heads
        self.head_dim = self.emb // self.num_heads
        self.MU_type = MU_type
        self.MU_init_method = MU_init_method
        self.scale = scale
        self.comp_factor = comp_factor

        self.initialize()

    def initialize(self):
        if self.MU_init_method not in ['random', 'average', 'first']:
            raise ValueError('Unknown MU_init_method: {}'.format(self.MU_init_method))

        #######################################
        if self.MU_type == 'diagonal':
            if self.MU_init_method == 'random':
                self.U_diag = nn.Parameter(self.scale * torch.randn(self.emb * self.num_models))
            elif self.MU_init_method in ['average', 'first']:
                self.U_diag = nn.Parameter(torch.ones(self.emb * self.num_models))

        #######################################
        elif self.MU_type == 'low_rank':
            if self.MU_init_method in ['average', 'first']:
                raise Exception(f"MU_init_method = {self.MU_init_method} is unsupported for low_rank")
            self.U_a = nn.Parameter(self.scale * torch.randn((self.new_emb, self.rank)) / math.sqrt(self.rank))
            self.U_b = nn.Parameter(self.scale * torch.randn((self.rank, self.emb * self.num_models)) / math.sqrt(self.rank))

        #######################################
        elif self.MU_type == 'diagonal_and_low_rank':
            if self.MU_init_method == 'random':
                self.U_diag = nn.Parameter(torch.zeros(self.emb * self.num_models))
                self.U_a = nn.Parameter(self.scale * torch.randn((self.new_emb, self.rank)) / math.sqrt(self.rank))
                self.U_b = nn.Parameter(self.scale * torch.randn((self.rank, self.emb * self.num_models)) / math.sqrt(self.rank))

            elif self.MU_init_method in ['average', 'first']:
                self.U_diag = nn.Parameter(torch.ones(self.emb * self.num_models))
                self.U_a = nn.Parameter(LOW_RANK_FACTOR * self.scale * torch.zeros((self.new_emb, self.rank)) / self.rank)
                self.U_b = nn.Parameter(LOW_RANK_FACTOR * self.scale * torch.randn((self.rank, self.emb * self.num_models)) / self.rank)

        #######################################
        elif self.MU_type == 'full':
            if self.MU_init_method == 'random':
                self.U = nn.Parameter(self.scale * torch.randn((self.new_emb, self.emb * self.num_models)))

            elif self.MU_init_method == 'average':
                self.U = nn.Parameter(torch.eye(self.emb).repeat(1, self.num_models))

        #######################################
        elif self.MU_type == 'att_heads':
            if self.MU_init_method == 'first':
                # U1
                identity_matrix = torch.eye(self.head_dim)
                U_1 = identity_matrix.unsqueeze(0).expand(self.num_heads, -1, -1)
                self.U_1 = nn.Parameter(U_1)

                # M2
                identity_matrix = torch.eye(self.num_heads)
                U_2 = identity_matrix.unsqueeze(0).expand(self.num_models, -1, -1)
                self.U_2 = nn.Parameter(U_2)

            else:
                raise Exception(f"MU_init_method = {self.MU_init_method} is unsupported for att_heads")
            #######################################

        #######################################
        else:
            raise Exception("Unsupported MU_type")


    def initialized_U_with_prev_model(self, prev_merged_model_U):
        # U in size [new_emb, emb * num_models]
        # prev_merged_model_U in size [new_emb, emb * (num_models-1)]

        if self.MU_type == 'diagonal':
            U_diag = self.U_diag.data
            U_diag[:self.emb * (self.num_models-1)] = prev_merged_model_U.U_diag
            self.U_diag = nn.Parameter(U_diag)

        #######################################
        elif self.MU_type == 'low_rank':
            self.U_a = nn.Parameter(prev_merged_model_U.U_a.data)
            U_b = self.U_b.data
            U_b[:, :self.emb * (self.num_models-1)] = prev_merged_model_U.U_b
            self.U_b = nn.Parameter(U_b)

        #######################################
        elif self.MU_type == 'diagonal_and_low_rank':
            U_diag = self.U_diag.data
            U_diag[:self.emb * (self.num_models - 1)] = prev_merged_model_U.U_diag
            self.U_diag = nn.Parameter(U_diag)

            self.U_a = nn.Parameter(prev_merged_model_U.U_a.data)
            U_b = self.U_b.data
            U_b[:, :self.emb * (self.num_models - 1)] = prev_merged_model_U.U_b
            self.U_b = nn.Parameter(U_b)

        #######################################
        elif self.MU_type == 'full':
            U = self.U.data
            U[:self.emb * (self.num_models - 1)] = prev_merged_model_U.U
            self.U = nn.Parameter(U)


        #######################################
        else:
            raise Exception("Unsupported MU_type")

    def create_U_func(self):
        if self.MU_type == 'diagonal':
            return create_concatenated_diagonal_matrices(vec=self.U_diag, emb=self.emb, num_models=self.num_models, dim_cat=1)

        elif self.MU_type == 'low_rank':
            return self.U_a @ self.U_b

        elif self.MU_type == 'diagonal_and_low_rank':
            U = create_concatenated_diagonal_matrices(vec=self.U_diag, emb=self.emb, num_models=self.num_models, dim_cat=1)
            return U + self.U_a @ self.U_b

        elif self.MU_type == 'full':
            return self.U

        elif self.MU_type == 'att_heads':
            u1_mat = torch.zeros(self.num_heads * self.head_dim, self.num_heads * self.head_dim)
            for head in range(self.num_heads):
                start_idx = head * self.head_dim
                end_idx = start_idx + self.head_dim
                u1_mat[start_idx:end_idx, start_idx:end_idx] = self.U_1[head]

            u2_mat = torch.zeros(self.num_heads * self.head_dim, self.num_models * self.num_heads * self.head_dim)
            for model in range(self.num_models):
                for head_in in range(self.num_heads):
                    for head_out in range(self.num_heads):
                        idx1 = head_in * self.head_dim
                        idx2 = (model * self.num_heads + head_out) * self.head_dim
                        diagonal_values = torch.full((self.head_dim,), self.U_2[model, head_in, head_out].item())
                        u2_mat[idx1:idx1 + self.head_dim, idx2:idx2 + self.head_dim] = torch.diag(diagonal_values)

            U = (u1_mat @ u2_mat).to('cuda')
            return U

    # X in shape [B, T, new_emb]
    # Output in shape [B, T, emb * models]
    def forward(self, x):
        if self.MU_type != 'att_heads':
            U = self.create_U_func()
            return x @ U

        else:
            B, T = x.shape[0], x.shape[1]
            if len(x.shape) == 3:
                x = x.reshape(B, T, self.num_heads, self.head_dim)
                x = torch.einsum('btnh,nhk->btnk', x, self.U_1)
                x = torch.einsum('btnh,mnk->btmkh', x, self.U_2)
                x = x.reshape(B, T, self.num_models, self.emb).reshape(B, T, self.emb * self.num_models)
                return x
            elif len(x.shape) != 3: # x is [B, emb]
                x = x.reshape(B, self.num_heads, self.head_dim)
                x = torch.einsum('bnh,nhk->bnk', x, self.U_1)
                x = torch.einsum('bnh,mnk->bmkh', x, self.U_2)
                x = x.reshape(B, self.num_models, self.emb).reshape(B, self.emb * self.num_models)
                return x


#############################################################


class VITPreProcessingMergerFull(nn.Module):
    """
    Used for merging the VIT layers that comes before the transformer.
    In the case where we all the merger blocks together.
    Including the projection of the patches, embeddings, and layer normalization.

    conv_weights: List of the weights of the convolutional layer that projects the patches. shape: [emb, C, patch_size, patch_size]
    class_embeddings: List of the class embeddings. shape: [emb]
    positional_embeddings: List of the positional embeddings. shape: [seq_length, emb]
    number_of_models: The number of models that will be merged.
    MU_type: The type of the M,U matrices.
    comp_factor: The merged model new hidden dimension will be emb * comp_factor.
    rank: The rank of the learned M,U matrix. -1 means full rank (emb).
    LN_in_state_dict: The weights of the layer normalization of the input. For initializing.
    last_features_scale: Scaling the features of the last model, and fix it later in the folding.
    """

    def __init__(self,
                 conv_weights: List[torch.Tensor],
                 class_embeddings: List[torch.Tensor],
                 positional_embeddings: List[torch.Tensor],
                 LN_in_state_dict: Dict[str, torch.Tensor],
                 number_of_models: int,
                 MU_type: str,
                 freeze_LN: bool = False,
                 comp_factor: float = 1,
                 rank: int = -1,
                 MU_init_method: str = 'random'):

        super(VITPreProcessingMergerFull, self).__init__()
        self.emb, _, self.patch_size, _ = conv_weights[0].shape
        self.seq_length, _ = positional_embeddings[0].shape
        # seq_length = 3 * patch_size * patch_size + 1
        self.num_models = number_of_models
        self.comp_factor = comp_factor
        self.new_emb = int(self.emb * self.comp_factor)
        self.rank = rank
        self.MU_type = MU_type.lower()
        self.MU_init_method = MU_init_method.lower()
        self.scale = self.emb ** -0.5

        # Verify the rank
        if self.rank != -1 and (self.rank > self.emb or self.rank <= 0):
            raise ValueError('The rank must be between 1 and the embedding dimension.')

        # Saves and arrange the models weights from the models that will be merged.
        # Those won't be learned so requires_grad=False.
        for i, conv_weight in enumerate(conv_weights):
            conv_weight.requires_grad = False
            conv_weights[i] = conv_weight.view(self.emb, 3 * self.patch_size * self.patch_size)

        for embedding_list in [class_embeddings, positional_embeddings]:
            for embedding in embedding_list:
                embedding.requires_grad = False

        self.conv_weights = nn.Parameter(torch.cat(conv_weights, dim=0),
                                         requires_grad=False)  # [num_models * emb, 3 * patch_size**2]
        self.class_embeddings = nn.Parameter(torch.cat(class_embeddings, dim=0),
                                             requires_grad=False)  # [emb * num_models]
        self.positional_embeddings = nn.Parameter(torch.cat(positional_embeddings, dim=1),
                                                  requires_grad=False)  # [seq_length - 1, num_models * emb]

        # Learned parameters
        self.LN_in = torch.nn.LayerNorm(self.new_emb)
        if LN_in_state_dict is not None:
            self.LN_in.load_state_dict(LN_in_state_dict)

        if freeze_LN:
            for param in self.LN_in.parameters():
                param.requires_grad = False

        # [emb * num_models, new_emb]
        self.M = MMatrix(emb=self.emb, new_emb=self.new_emb, num_models=self.num_models, rank=self.rank,
                         MU_type=self.MU_type, MU_init_method=self.MU_init_method, scale=self.scale,
                         comp_factor=self.comp_factor)

    # inputs in shape [B, C, H, W]
    # outputs in shape [seq_length, B, new_emb]
    def forward(self, inputs: torch.Tensor, for_features=False) -> torch.Tensor:
        ######## The conv part, projecting chunks of the image to vectors ########
        B, _, N, _ = inputs.shape

        # Calculate the number of chunks along the last two dimensions
        chunks_y = N // self.patch_size
        chunks_x = N // self.patch_size

        # Split the tensor along the last two dimensions

        results_list = []
        for i in range(chunks_y):
            for j in range(chunks_x):
                sub_tensor = inputs[:, :,
                             i * self.patch_size:(i + 1) * self.patch_size,
                             j * self.patch_size:(j + 1) * self.patch_size]

                sub_tensor_reshaped = sub_tensor.reshape(B, 3 * self.patch_size * self.patch_size)
                # [B, 3 * patch * patch] @ [3 * patch * patch , num_models * emb] = [B , num_models * emb]
                result = sub_tensor_reshaped @ self.conv_weights.T
                results_list.append(result)

        x = torch.stack(results_list, dim=2)  # [B , num_models * emb, seq_length-1]
        x = x.permute(0, 2, 1)  # [B , seq_length-1, num_models * emb]

        """
        ######################################################
        x = x @ M  # [B , seq_length-1, num_models * emb] @ [num_models * emb , new_emb] = [B , seq_length-1, new_emb]

        ######## The embedding part ########
        cls = self.class_embeddings @ M  # [num_models * emb] @ [num_models * emb, new_emb] = [new_emb]

        # [seq_length, num_models * emb] @ [num_models * emb, new_emb] = [seq_length, new_emb]
        pos_emb = self.positional_embeddings @ M

        x = torch.cat(
            [cls.to(x.dtype) +
             torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [B, seq_length, new_emb]

        x = x + pos_emb.to(x.dtype)
        """
        ######################################################
        x = torch.cat(
            [self.class_embeddings.to(x.dtype) +
             torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [B , seq_length, num_models * emb]

        x = x + self.positional_embeddings.to(x.dtype)

        x = self.M(x)  # [B , seq_length, num_models * emb] @ [num_models * emb , new_emb] = [B , seq_length, new_emb]
        ######################################################

        x = self.LN_in(x)
        x = x.permute(1, 0, 2)  # [B, seq_length, new_emb] -> [seq_length, B, new_emb]
        return x

    # Folding M,U matrices into the weights, and return the new weights
    def fold(self, device) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            M = self.M.create_M_func()

            # [seq_length-1 , num_models * emb] @ [num_models * emb , new_emb] = [seq_length-1 , new_emb]
            merged_conv_weights = self.conv_weights.T @ M
            merged_conv_weights = merged_conv_weights.T  # [new_emb, seq_length-1]
            merged_conv_weights = merged_conv_weights.view(self.new_emb, 3, self.patch_size, self.patch_size)

            # [emb * num_models] @ [emb * num_models, new_emb] = [new_emb]
            merged_cls_emb = self.class_embeddings @ M

            # [seq_length, num_models * emb] @ [num_models * emb, new_emb] = [seq_length, new_emb]
            merged_pos_emb = self.positional_embeddings @ M

            merged_weights_dict = {}
            merged_weights_dict['conv_weights'] = merged_conv_weights.clone().to(device)
            merged_weights_dict['cls'] = merged_cls_emb.clone().to(device)
            merged_weights_dict['pos_emb'] = merged_pos_emb.clone().to(device)
            merged_weights_dict['LN_in_weight'] = self.LN_in.weight.clone().to(device)
            merged_weights_dict['LN_in_bias'] = self.LN_in.bias.clone().to(device)

            return merged_weights_dict

    def end_of_training(self):
        pass


#############################################################


class MergeAttentionSubBlockFull(nn.Module):
    """
    Used for merging the attention sub-block inside the transformer.
    Including layer norm; the projection for q,k,v; attention; and residual connection.

    in_proj_weight_list: Tuple with length num_models, of the weights that project to q,k,v. shape: [3*emb, emb]
    in_proj_bias_list: Tuple with length num_models, of the bias that project to q,k,v. shape: [3*emb]
    out_proj_weight_list: Tuple of the weights that project from the attention output to the emb space. shape: [emb, emb]
    out_proj_bias_list: Tuple of the bias that project from the attention output to the emb space. shape: [emb]
    number_of_models: The number of models that will be merged.
    MU_type: The type of the M,U matrices.
    num_heads: The number of heads in the multi-head attention.
    comp_factor: The merged model new hidden dimension will be emb * comp_factor.
    rank: The rank of the learned M,U matrix. -1 means full rank (emb).
    LN_att_state_dict: The weights of the layer normalization of the attention sub-block. For initializing.
    freeze_LN: Whether to freeze the layer normalization weights.
    init_method: The method for initializing the learned parameters M and U.
    last_features_scale: Scaling the features of the last model, and fix it later in the folding.
    learn_LN_U: Whether to learn the layer normalization weights and the U matrix. If false, it means that a previous block
    already learned those, so we will use those we get in U_prev, LN_dict_prev.
    """

    def __init__(self,
                 in_proj_weight_list: List[torch.Tensor],
                 in_proj_bias_list: List[torch.Tensor],
                 out_proj_weight_list: List[torch.Tensor],
                 out_proj_bias_list: List[torch.Tensor],
                 LN_att_state_dict: Dict[str, torch.Tensor],
                 number_of_models: int,
                 MU_type: str,
                 num_heads: int,
                 comp_factor: float = 1,
                 rank: int = -1,
                 MU_init_method: str = 'random',
                 learn_LN_U: bool = True,
                 freeze_LN: bool = False,
                 U_prev: torch.Tensor = None,
                 LN_dict_prev: Dict[str, torch.Tensor] = None):

        super(MergeAttentionSubBlockFull, self).__init__()
        _, self.emb = in_proj_weight_list[0].shape
        self.num_models = number_of_models
        self.num_heads = num_heads
        self.scale = self.emb ** -0.5
        self.comp_factor = comp_factor
        self.new_emb = int(self.emb * self.comp_factor)
        self.head_dim = self.new_emb // self.num_heads
        self.rank = rank
        self.MU_type = MU_type.lower()
        self.MU_init_method = MU_init_method.lower()
        self.learn_LN_U = learn_LN_U

        assert self.num_heads * self.head_dim == self.new_emb, "The new embedding dimension must be divisible by the number of heads."
        if self.rank != -1 and (self.rank > self.emb or self.rank <= 0):
            raise ValueError('The rank must be between 1 and the embedding dimension.')

        # Saves and arrange the models weights
        for parameter_list in [in_proj_weight_list, in_proj_bias_list, out_proj_weight_list, out_proj_bias_list]:
            for parameter in parameter_list:
                parameter.requires_grad = False

        in_proj_weight_q, in_proj_weight_k, in_proj_weight_v = \
            self.split_to_q_k_v(in_proj_weight_list, is_bias=False)
        self.in_proj_weight_q = nn.Parameter(in_proj_weight_q, requires_grad=False)
        self.in_proj_weight_k = nn.Parameter(in_proj_weight_k, requires_grad=False)
        self.in_proj_weight_v = nn.Parameter(in_proj_weight_v, requires_grad=False)
        # each in shape [num_models, emb, emb] # new
        # each in shape [num_models * emb, num_models * emb] # old

        in_proj_bias_q, in_proj_bias_k, in_proj_bias_v = \
            self.split_to_q_k_v(in_proj_bias_list, is_bias=True)
        self.in_proj_bias_q = nn.Parameter(in_proj_bias_q, requires_grad=False)
        self.in_proj_bias_k = nn.Parameter(in_proj_bias_k, requires_grad=False)
        self.in_proj_bias_v = nn.Parameter(in_proj_bias_v, requires_grad=False)
        # each in shape [num_models * emb]

        # out_proj_weight = custom_block_diag(out_proj_weight_list)  # shape: [num_models * emb, num_models * emb] #old
        out_proj_weight = torch.stack(out_proj_weight_list, dim=0)  # shape: [num_models, emb, emb] # new
        self.out_proj_weight = nn.Parameter(out_proj_weight, requires_grad=False)
        out_proj_bias = torch.cat(out_proj_bias_list, dim=0)  # shape: [num_models * emb]
        self.out_proj_bias = nn.Parameter(out_proj_bias, requires_grad=False)

        # Learned parameters
        self.LN_att = torch.nn.LayerNorm(self.new_emb)
        if LN_att_state_dict is not None:
            if self.comp_factor != 1:
                raise Exception('Cannot initialize the attention the layer according the pre-trained model,'
                                ' when the comp_factor is not 1.')
            self.LN_att.load_state_dict(LN_att_state_dict)

        if not self.learn_LN_U:  # So will use the previous layer norm, and won't learn it.
            self.LN_att.load_state_dict(LN_dict_prev)
            self.LN_att.weight.requires_grad = False
            self.LN_att.bias.requires_grad = False

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
                         comp_factor=self.comp_factor)

        self.M_q = MMatrix(emb=self.emb, new_emb=self.new_emb, num_models=self.num_models,
                           rank=self.rank, MU_type=self.MU_type, MU_init_method=self.MU_init_method,
                           scale=self.scale, comp_factor=self.comp_factor)

        self.M_k = MMatrix(emb=self.emb, new_emb=self.new_emb, num_models=self.num_models,
                           rank=self.rank, MU_type=self.MU_type, MU_init_method=self.MU_init_method,
                           scale=self.scale, comp_factor=self.comp_factor)

        self.M_v = MMatrix(emb=self.emb, new_emb=self.new_emb, num_models=self.num_models,
                           rank=self.rank, MU_type=self.MU_type, MU_init_method=self.MU_init_method,
                           scale=self.scale, comp_factor=self.comp_factor)

        self.U_o_1 = UMatrix(emb=self.emb, new_emb=self.new_emb, num_models=self.num_models, rank=self.rank,
                         MU_type=self.MU_type, MU_init_method=self.MU_init_method, scale=self.scale,
                         comp_factor=self.comp_factor)

        self.M_o_2 = MMatrix(emb=self.emb, new_emb=self.new_emb, num_models=self.num_models,
                           rank=self.rank, MU_type=self.MU_type, MU_init_method=self.MU_init_method,
                           scale=self.scale, comp_factor=self.comp_factor)

    # Split the tensor_list into num_models tensors.
    # Biases should be concatenated, weights should be formed into a diagonal block matrix.
    def split_to_q_k_v(self, tensor_list, is_bias):
        # Lists to store the split tensors
        w_q_list, w_k_list, w_v_list = [], [], []
        i = 0
        for tensor in tensor_list:
            w_q, w_k, w_v = torch.split(tensor, [self.emb, self.emb, self.emb], dim=0)
            w_q_list.append(w_q)
            w_k_list.append(w_k)
            w_v_list.append(w_v)
            i += 1

        if is_bias:
            # Concatenate the tensors
            w_q_cat = torch.cat(w_q_list, dim=0)
            w_k_cat = torch.cat(w_k_list, dim=0)
            w_v_cat = torch.cat(w_v_list, dim=0)

        else:
            # new
            w_q_cat = torch.stack(w_q_list, dim=0)  # shape: [num_models, emb, emb]
            w_k_cat = torch.stack(w_k_list, dim=0)  # shape: [num_models, emb, emb]
            w_v_cat = torch.stack(w_v_list, dim=0)  # shape: [num_models, emb, emb]
            """
            #old
            w_q_cat = custom_block_diag(w_q_list)
            w_k_cat = custom_block_diag(w_k_list)
            w_v_cat = custom_block_diag(w_v_list)
            """

        w_q_cat.requires_grad = False
        w_k_cat.requires_grad = False
        w_v_cat.requires_grad = False

        return w_q_cat, w_k_cat, w_v_cat

    # inputs in shape [seq_length, B, new_emb]
    # outputs in shape [seq_length, B, new_emb]
    def forward(self, inputs: torch.Tensor, for_features=False) -> torch.Tensor:
        # Forward
        seq_length, B, _ = inputs.shape
        x = self.U_att(self.LN_att(inputs))
        # [seq_length, B, new_emb] @ [new_emb, emb * num_models] -> [seq_length, B, emb * num_models]

        ######## Multi-head Attention ########
        # new

        q = _bmm_linear(x, self.in_proj_weight_q, self.in_proj_bias_q)  # [seq_length, B, num_models * emb]
        k = _bmm_linear(x, self.in_proj_weight_k, self.in_proj_bias_k)  # [seq_length, B, num_models * emb]
        v = _bmm_linear(x, self.in_proj_weight_v, self.in_proj_bias_v)  # [seq_length, B, num_models * emb]

        """ #old
        q = _linear(x, self.in_proj_weight_q, self.in_proj_bias_q)  # [seq_length, B, num_models * emb]
        k = _linear(x, self.in_proj_weight_k, self.in_proj_bias_k)  # [seq_length, B, num_models * emb]
        v = _linear(x, self.in_proj_weight_v, self.in_proj_bias_v)  # [seq_length, B, num_models * emb]
        """

        q = self.M_q(q)  # [seq_length, B, num_models * emb] @ [num_models * emb, emb] = [seq_length, B, new_emb]
        k = self.M_k(k)  # [seq_length, B, num_models * emb] @ [num_models * emb, emb] = [seq_length, B, new_emb]
        v = self.M_v(v)  # [seq_length, B, num_models * emb] @ [num_models * emb, emb] = [seq_length, B, new_emb]

        q = q.reshape(seq_length, B * self.num_heads, self.head_dim).transpose(0, 1)  # [B * num_heads, seq_length, head_dim]
        k = k.reshape(k.shape[0], B * self.num_heads, self.head_dim).transpose(0, 1)  # [B * num_heads, seq_length, head_dim]
        v = v.reshape(v.shape[0], B * self.num_heads, self.head_dim).transpose(0, 1)  # [B * num_heads, seq_length, head_dim]

        _, Nt, E = q.shape
        q_scaled = q / math.sqrt(E)
        attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = attn_output_weights.softmax(dim=-1)  # [B * num_heads, seq_length, seq_length]
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(seq_length * B,
                                                                    self.new_emb)  # [seq_length * B, new_emb]

        ######## OutPut projection ########
        # [seq_length * B, new_emb] @ [new_emb, num_models * emb] = [seq_length * B, num_models * emb]
        attn_output = self.U_o_1(attn_output)

        # [seq_length * B, num_models * emb] @ [num_models * emb, num_models * emb] = [seq_length * B, num_models * emb]
        """ old
        attn_output = _linear(attn_output,
                              self.out_proj_weight,
                              self.out_proj_bias)
        """
        # new
        attn_output = attn_output.view(seq_length, B, self.num_models * self.emb)  # [seq_length, B, num_models * emb]
        attn_output = _bmm_linear(attn_output,
                                  self.out_proj_weight,
                                  self.out_proj_bias)

        attn_output_viewd = attn_output.view(seq_length * B, self.num_models * self.emb)

        # [seq_length * B, num_models * emb] @ [num_models * emb, new_emb] = [seq_length * B, new_emb]
        attn_output_viewd = self.M_o_2(attn_output_viewd)

        attn_output_viewd = attn_output_viewd.view(seq_length, B, attn_output_viewd.size(1))  # [seq_length, B, new_emb]

        attn_output_viewd = attn_output_viewd + inputs

        if not for_features:
            return attn_output_viewd  # [seq_length, B, new_emb]
        else:
            return attn_output_viewd, attn_output  # [seq_length, B, num_models * emb]

    # inputs in shape [seq_length, B, new_emb]
    # outputs in shape [seq_length, B, new_emb]
    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        # Forward
        dict = {}
        seq_length, B, _ = inputs.shape
        x = self.U_att(self.LN_att(inputs))
        # [seq_length, B, new_emb] @ [new_emb, emb * num_models] -> [seq_length, B, emb * num_models]

        dict['LN-att'] = x.clone().detach().cpu().permute(1, 0, 2)[:, :, :self.emb]

        ######## Multi-head Attention ########
        # new

        q = _bmm_linear(x, self.in_proj_weight_q, self.in_proj_bias_q)  # [seq_length, B, num_models * emb]
        k = _bmm_linear(x, self.in_proj_weight_k, self.in_proj_bias_k)  # [seq_length, B, num_models * emb]
        v = _bmm_linear(x, self.in_proj_weight_v, self.in_proj_bias_v)  # [seq_length, B, num_models * emb]

        """ #old
        q = _linear(x, self.in_proj_weight_q, self.in_proj_bias_q)  # [seq_length, B, num_models * emb]
        k = _linear(x, self.in_proj_weight_k, self.in_proj_bias_k)  # [seq_length, B, num_models * emb]
        v = _linear(x, self.in_proj_weight_v, self.in_proj_bias_v)  # [seq_length, B, num_models * emb]
        """

        q = self.M_q(q)  # [seq_length, B, num_models * emb] @ [num_models * emb, emb] = [seq_length, B, new_emb]
        k = self.M_k(k)  # [seq_length, B, num_models * emb] @ [num_models * emb, emb] = [seq_length, B, new_emb]
        v = self.M_v(v)  # [seq_length, B, num_models * emb] @ [num_models * emb, emb] = [seq_length, B, new_emb]

        dict['q'] = q.clone().detach().cpu().permute(1, 0, 2)
        dict['k'] = k.clone().detach().cpu().permute(1, 0, 2)
        dict['v'] = v.clone().detach().cpu().permute(1, 0, 2)

        q = q.reshape(seq_length, B * self.num_heads, self.head_dim).transpose(0,
                                                                               1)  # [B * num_heads, seq_length, head_dim]
        k = k.reshape(k.shape[0], B * self.num_heads, self.head_dim).transpose(0,
                                                                               1)  # [B * num_heads, seq_length, head_dim]
        v = v.reshape(v.shape[0], B * self.num_heads, self.head_dim).transpose(0,
                                                                               1)  # [B * num_heads, seq_length, head_dim]

        _, Nt, E = q.shape
        q_scaled = q / math.sqrt(E)
        attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = attn_output_weights.softmax(dim=-1)  # [B * num_heads, seq_length, seq_length]
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(seq_length * B,
                                                                    self.new_emb)  # [seq_length * B, new_emb]

        ######## OutPut projection ########
        # [seq_length * B, new_emb] @ [new_emb, num_models * emb] = [seq_length * B, num_models * emb]
        attn_output = self.U_o_1(attn_output)

        # [seq_length * B, num_models * emb] @ [num_models * emb, num_models * emb] = [seq_length * B, num_models * emb]
        """ old
        attn_output = _linear(attn_output,
                              self.out_proj_weight,
                              self.out_proj_bias)
        """
        # new
        attn_output = attn_output.view(seq_length, B,
                                       self.num_models * self.emb)  # [seq_length, B, num_models * emb]
        attn_output = _bmm_linear(attn_output,
                                  self.out_proj_weight,
                                  self.out_proj_bias)
        attn_output = attn_output.view(seq_length * B, self.num_models * self.emb)

        # [seq_length * B, num_models * emb] @ [num_models * emb, new_emb] = [seq_length * B, new_emb]
        attn_output = self.M_o_2(attn_output)

        attn_output = attn_output.view(seq_length, B, attn_output.size(1))  # [seq_length, B, new_emb]

        dict['attn'] = attn_output.clone().detach().cpu().permute(1, 0, 2)

        y = inputs + attn_output  # [seq_length, B, new_emb]

        return y, dict

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
            """ #old
            new_weight_q = folding_M_U_into_weight(W=self.in_proj_weight_q, M=M_q, U=U_att)
            new_weight_k = folding_M_U_into_weight(W=self.in_proj_weight_k, M=M_k, U=U_att)
            new_weight_v = folding_M_U_into_weight(W=self.in_proj_weight_v, M=M_v, U=U_att)
            """

            # new
            new_weight_q = bmm_folding_M_U_into_weight(W=self.in_proj_weight_q, M=M_q, U=U_att)
            new_weight_k = bmm_folding_M_U_into_weight(W=self.in_proj_weight_k, M=M_k, U=U_att)
            new_weight_v = bmm_folding_M_U_into_weight(W=self.in_proj_weight_v, M=M_v, U=U_att)

            new_in_proj_weight = torch.cat([new_weight_q, new_weight_k, new_weight_v], dim=0)

            # [num_models * emb] @ [num_models * emb, new_emb] = [new_emb]
            new_b_q = self.in_proj_bias_q @ M_q
            new_b_k = self.in_proj_bias_k @ M_k
            new_b_v = self.in_proj_bias_v @ M_v
            new_in_proj_b = torch.cat([new_b_q, new_b_k, new_b_v], dim=0)

            # new_out_proj_weight = folding_M_U_into_weight(W=self.out_proj_weight, M=M_o_2, U=U_o_1) #old
            new_out_proj_weight = bmm_folding_M_U_into_weight(W=self.out_proj_weight, M=M_o_2, U=U_o_1)  # new
            new_out_proj_b = self.out_proj_bias @ M_o_2

            merged_weights_dict = {}
            merged_weights_dict['LN_att_weight'] = self.LN_att.weight.clone().to(device)
            merged_weights_dict['LN_att_bias'] = self.LN_att.bias.clone().to(device)
            merged_weights_dict['in_proj_weight'] = new_in_proj_weight.clone().to(device)
            merged_weights_dict['in_proj_b'] = new_in_proj_b.clone().to(device)
            merged_weights_dict['out_proj_weight'] = new_out_proj_weight.clone().to(device)
            merged_weights_dict['out_proj_b'] = new_out_proj_b.clone().to(device)

            return merged_weights_dict

    def end_of_training(self):
        pass


#############################################################

class MergeMLPSubBlockFull(nn.Module):
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
                 mlp_activation,
                 number_of_models: int,
                 MU_type: str,
                 comp_factor: float = 1,
                 rank: int = -1,
                 freeze_LN: bool = False,
                 MU_init_method: str = 'random'):

        super(MergeMLPSubBlockFull, self).__init__()
        self.hidden_dim, self.emb = linear1_weight_list[0].shape
        self.num_models = number_of_models
        self.scale = self.emb ** -0.5
        self.scale_hidden = self.hidden_dim ** -0.5
        self.comp_factor = comp_factor
        self.new_emb = int(self.emb * self.comp_factor)
        self.new_hid_dim = int(self.hidden_dim * self.comp_factor)
        self.rank = rank
        self.MU_type = MU_type.lower()
        self.hidden_rank = int(self.rank * self.hidden_dim / self.emb)
        self.MU_init_method = MU_init_method.lower()

        # Saves and arrange the models weights
        for parameter_list in [linear1_weight_list, linear1_bias_list, linear2_weight_list, linear2_bias_list]:
            for parameter in parameter_list:
                parameter.requires_grad = False

        # linear1_weight = custom_block_diag(linear1_weight_list)  # [num_models * hidden_dim, num_models * emb] #old
        linear1_weight = torch.stack(linear1_weight_list, dim=0)  # [num_models, hidden_dim, emb] #new
        linear1_bias = torch.cat(linear1_bias_list, dim=0)  # [num_models * hidden_dim]
        # linear2_weight = custom_block_diag(linear2_weight_list)  # [num_models * emb, num_models * hidden_dim] # old
        linear2_weight = torch.stack(linear2_weight_list, dim=0)  # [num_models, emb, hidden_dim] # new
        linear2_bias = torch.cat(linear2_bias_list, dim=0)  # [num_models * emb]
        self.linear1_weight = nn.Parameter(linear1_weight, requires_grad=False)
        self.linear1_bias = nn.Parameter(linear1_bias, requires_grad=False)
        self.linear2_weight = nn.Parameter(linear2_weight, requires_grad=False)
        self.linear2_bias = nn.Parameter(linear2_bias, requires_grad=False)

        self.activation = mlp_activation

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
                         comp_factor=self.comp_factor)

        # [hidden_dim * num_models, new_hid_dim]
        self.M_1 = MMatrix(emb=self.hidden_dim, new_emb=self.new_hid_dim, num_models=self.num_models,
                           rank=self.hidden_rank, MU_type=self.MU_type, MU_init_method=self.MU_init_method,
                           scale=self.scale_hidden, comp_factor=self.comp_factor)

        # [new_hid_dim, hidden_dim * num_models]
        self.U_1 = UMatrix(emb=self.hidden_dim, new_emb=self.new_hid_dim, num_models=self.num_models,
                           rank=self.hidden_rank, MU_type=self.MU_type, MU_init_method=self.MU_init_method,
                           scale=self.scale_hidden, comp_factor=self.comp_factor)

        # [emb * num_models, new_emb]
        self.M_2 = MMatrix(emb=self.emb, new_emb=self.new_emb, num_models=self.num_models,
                           rank=self.rank, MU_type=self.MU_type, MU_init_method=self.MU_init_method,
                           scale=self.scale, comp_factor=self.comp_factor)

    # inputs in shape [seq_length, B, new_emb]
    # outputs in shape [seq_length, B, new_emb]
    def forward(self, inputs: torch.Tensor, for_features=False) -> torch.Tensor:
        # Forward
        y = self.U_0(self.LN_mlp(inputs))
        # [seq_length, B, new_emb] @ [new_emb, emb * num_models] = [seq_length, B, emb * num_models]

        ######## First linear layer ########
        # y = _linear(y, self.linear1_weight, self.linear1_bias) #old
        y = _bmm_linear(y, self.linear1_weight, self.linear1_bias)  # new
        # [seq_length, B, num_models * emb] @ [emb * num_models, num_models * hid_dim] = [seq_length, B, num_models * hid_dim]

        y = self.M_1(y)
        # [seq_length, B, num_models * hid_dim] @ [num_models * hid_dim, new_hid_dim] = [seq_length, B, new_hid_dim]
        y = self.activation(y)
        y = self.U_1(y)
        # [seq_length, B, new_hid_dim] @ [new_hid_dim, num_models * hid_dim] = [seq_length, B, num_models * hid_dim]

        ######## Second linear layer ########
        # y = _linear(y, self.linear2_weight, self.linear2_bias) #old
        y = _bmm_linear(y, self.linear2_weight, self.linear2_bias)  # new
        # [seq_length, B, num_models * hid_dim] @ [num_models * hid_dim, num_models * emb] = [seq_length, B, num_models * emb]

        z = self.M_2(y)  # [seq_length, B, num_models * emb] @ [num_models * emb, new_emb] = [seq_length, B, new_emb]

        z = inputs + z  # [seq_length, B, new_emb]

        if not for_features:
            return z  # [seq_length, B, new_emb]
        else:
            return z, y  # [seq_length, B, num_models * emb]

    # inputs in shape [seq_length, B, new_emb]
    # outputs in shape [seq_length, B, new_emb]
    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        dict = {}
        # Forward
        y = self.U_0(self.LN_mlp(inputs))
        # [seq_length, B, new_emb] @ [new_emb, emb * num_models] = [seq_length, B, emb * num_models]
        dict['LN-mlp'] = y.clone().detach().cpu().permute(1, 0, 2)[:, :, :self.emb]

        ######## First linear layer ########
        # y = _linear(y, self.linear1_weight, self.linear1_bias) #old
        y = _bmm_linear(y, self.linear1_weight, self.linear1_bias)  # new
        # [seq_length, B, num_models * emb] @ [emb * num_models, num_models * hid_dim] = [seq_length, B, num_models * hid_dim]
        dict['fc1'] = y.clone().detach().cpu().permute(1, 0, 2)[:, :, :self.hidden_dim]

        y = self.M_1(y)
        # [seq_length, B, num_models * hid_dim] @ [num_models * hid_dim, new_hid_dim] = [seq_length, B, new_hid_dim]
        y = self.activation(y)
        dict['gelu'] = y.clone().detach().cpu().permute(1, 0, 2)
        y = self.U_1(y)
        # [seq_length, B, new_hid_dim] @ [new_hid_dim, num_models * hid_dim] = [seq_length, B, num_models * hid_dim]

        ######## Second linear layer ########
        # y = _linear(y, self.linear2_weight, self.linear2_bias) #old
        y = _bmm_linear(y, self.linear2_weight, self.linear2_bias)  # new
        # [seq_length, B, num_models * hid_dim] @ [num_models * hid_dim, num_models * emb] = [seq_length, B, num_models * emb]
        dict['fc2'] = y.clone().detach().cpu().permute(1, 0, 2)[:, :, :self.emb]

        y = self.M_2(y)  # [seq_length, B, num_models * emb] @ [num_models * emb, new_emb] = [seq_length, B, new_emb]
        output = inputs + y
        dict['mlp_out'] = output.clone().detach().cpu().permute(1, 0, 2)

        return output, dict

    # Folding M,U matrices into the weights, and return the new weights
    def fold(self, device) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            # Create the M and U matrices
            U_0 = self.U_0.create_U_func()
            M_1 = self.M_1.create_M_func()
            U_1 = self.U_1.create_U_func()
            M_2 = self.M_2.create_M_func()

            # Create the new weights
            # new_linear1_weight = folding_M_U_into_weight(W=self.linear1_weight, M=M_1, U=U_0) #old
            new_linear1_weight = bmm_folding_M_U_into_weight(W=self.linear1_weight, M=M_1, U=U_0)  # new
            # [new_hid_dim, new_emb]

            new_linear1_bias = self.linear1_bias @ M_1
            # [num_models * hid_dim] @ [num_models * hid_dim, new_hid_dim] = [new_hid_dim]

            # new_linear2_weight = folding_M_U_into_weight(W=self.linear2_weight, M=M_2, U=U_1) #old
            new_linear2_weight = bmm_folding_M_U_into_weight(W=self.linear2_weight, M=M_2, U=U_1)  # new
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

    def end_of_training(self):
        pass


#############################################################
class MergeVITOutputFull(nn.Module):
    """
    Used for merging the MLP output layer.
    Including layer norm and a linear layer with no bias.
    linear_weight_list: List of the weights that project to the output dim. shape: [emb, output_dim]
    (this is different from the usual linear layer, which is [out_dim, emb])
    number_of_models: The number of models that will be merged.
    MU_type: The type of the M,U matrices.
    comp_factor: The merged model new hidden dimension will be emb * comp_factor.
    rank: The rank of the learned M,U matrix. -1 means full rank (emb).
    init_method: The method for initializing the learned parameters M and U.
    last_features_scale: Scaling the features of the last model, and fix it later in the folding.
    learn_tasks_sequentially: Whether to learn the tasks sequentially.
    curr_task_sequence_iter: The current task sequence iteration.
    prev_merged_model_U: The U matrix of the previous merged model. For initializing.
    """

    def __init__(self,
                 linear_weight_list: List[torch.Tensor],
                 LN_output_dict: Dict[str, torch.Tensor],
                 num_models: int,
                 MU_type: str,
                 comp_factor: float = 1,
                 rank: int = -1,
                 freeze_LN: bool = False,
                 MU_init_method: str = 'random',
                 learn_tasks_sequentially: bool = False,
                 curr_task_sequence_iter:int = 0,
                 prev_merged_model_U = None):

        super(MergeVITOutputFull, self).__init__()
        self.emb, self.out_dim = linear_weight_list[0].shape
        self.num_models = num_models
        self.scale = self.emb ** -0.5
        self.scale_out = self.out_dim ** -0.5
        self.comp_factor = comp_factor
        self.new_emb = int(self.emb * self.comp_factor)
        self.rank = rank
        self.MU_type = MU_type.lower()
        self.MU_init_method = MU_init_method.lower()
        self.learn_tasks_sequentially = learn_tasks_sequentially
        self.curr_task_sequence_iter = curr_task_sequence_iter

        # Saves and arrange the models weights
        for i in range(len(linear_weight_list)):
            linear_weight_list[i].requires_grad = False
            linear_weight_list[i] = linear_weight_list[i].T

        # self.linear_weight = nn.Parameter(custom_block_diag(linear_weight_list), requires_grad=False)  # [num_models * out_dim, num_models * emb] #old
        self.linear_weight = nn.Parameter(torch.stack(linear_weight_list, dim=0), requires_grad=False)  # [num_models, out_dim, emb] #new

        # Learned parameters
        self.LN_output = torch.nn.LayerNorm(self.new_emb)
        if LN_output_dict is not None:
            self.LN_output.load_state_dict(LN_output_dict)

        if freeze_LN:
            for param in self.LN_output.parameters():
                param.requires_grad = False

        # [new_emb, emb * num_models]
        self.U = UMatrix(emb=self.emb, new_emb=self.new_emb, num_models=self.num_models, rank=self.rank,
                         MU_type=self.MU_type, MU_init_method=self.MU_init_method, scale=self.scale,
                         comp_factor=self.comp_factor)

        # [num_models * out_dim, out_dim]
        self.M = MMatrix(emb=self.out_dim, new_emb=self.out_dim, num_models=self.num_models, rank=self.rank,
                         MU_type=self.MU_type, MU_init_method=self.MU_init_method, scale=self.scale_out,
                         comp_factor=self.comp_factor)

        # [out_dim, out_dim * num_models]
        num_models_output = self.num_models if not self.learn_tasks_sequentially else self.curr_task_sequence_iter + 2
        # If learn_tasks_sequentially is True, then the number of models is the current task sequence iteration + 2
        self.U_output = UMatrix(emb=self.out_dim, new_emb=self.out_dim, num_models=num_models_output, rank=self.rank,
                         MU_type=self.MU_type, MU_init_method=self.MU_init_method, scale=self.scale_out,
                         comp_factor=self.comp_factor)

        if prev_merged_model_U is not None: # Initialize the U matrix with the previous model U
            self.U_output.initialized_U_with_prev_model(prev_merged_model_U)

    # inputs in shape [seq_length, B, new_emb]
    # outputs in shape [B, num_models * out_dim]
    def forward(self, inputs: torch.Tensor, for_features: bool = False, with_u: bool = True) -> torch.Tensor:
        # Forward
        inputs = inputs.permute(1, 0, 2)  # [seq_length, B, new_emb] -> [B, seq_length, new_emb]
        x = self.U(self.LN_output(inputs[:, 0, :]))
        # [B, new_emb] @ [new_emb, emb * num_models] = [B, emb * num_models]

        x = _bmm_linear_output(x, self.linear_weight)
        # [B, emb * num_models] @ [emb * num_models, out_dim * num_models] = [B, out_dim * num_models]

        x = self.M(x)  # [B, num_models * out_dim] @ [num_models * out_dim, out_dim] = [B, out_dim]
        if with_u:
            x = self.U_output(x) # [B, out_dim] @ [out_dim, num_models * out_dim] = [B, num_models * out_dim]
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
            # new_linear_weight = folding_M_U_into_weight(W=self.linear_weight, M=M, U=U) #old
            new_linear_weight = bmm_folding_M_U_into_weight(W=self.linear_weight, M=M, U=U)  # new
            new_linear_weight = new_linear_weight.T  # Because this layer is activate in the VIT without transpose
            # [new_emb, out_dim]

            merged_weights_dict = {}
            merged_weights_dict['linear_weight'] = new_linear_weight.clone().to(device)
            merged_weights_dict['LN_out_weight'] = self.LN_output.weight.clone().to(device)
            merged_weights_dict['LN_out_bias'] = self.LN_output.bias.clone().to(device)
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

    def get_U_layer(self, device):
        return self.U_output.to(device)


#############################################################
class LayerNormUBlock(nn.Module):
    """
    A block of LayerNorm and U matrix.
    LN_dict: A dictionary of the LayerNorm parameters.
    number_of_models: The number of models that will be merged.
    MU_type: The type of the M,U matrices.
    comp_factor: The merged model new hidden dimension will be emb * comp_factor.
    rank: The rank of the learned M,U matrix. -1 means full rank (emb).
    init_method: The method for initializing the learned parameters M and U.
    last_features_scale: Scaling the features of the last model, and fix it later in the folding.
    """

    def __init__(self,
                 emb: int,
                 LN_dict: Dict[str, torch.Tensor],
                 num_models: int,
                 MU_type: str,
                 comp_factor: float = 1,
                 rank: int = -1,
                 freeze_LN: bool = False,
                 MU_init_method: str = 'random'):

        super(LayerNormUBlock, self).__init__()
        # self.emb = LN_dict['weight'].shape[0]
        self.emb = emb
        self.num_models = num_models
        self.scale = self.emb ** -0.5
        self.comp_factor = comp_factor
        self.new_emb = int(self.emb * self.comp_factor)
        self.rank = rank
        self.MU_type = MU_type.lower()
        self.MU_init_method = MU_init_method.lower()

        # Learned parameters
        self.LN = torch.nn.LayerNorm(self.new_emb)
        if LN_dict is not None:
            self.LN.load_state_dict(LN_dict)

        if freeze_LN:
            for param in self.LN.parameters():
                param.requires_grad = False

        self.initialize_U()

    def initialize_U(self):
        # U in shape [new_emb, num_models * emb]
        if self.MU_init_method not in ['random', 'average', 'first']:
            raise ValueError('Unknown MU_init_method: {}'.format(self.MU_init_method))
        if self.MU_type in ['low_rank', 'diagonal_and_low_rank'] and (self.rank < 1 or self.rank >= self.new_emb):
            raise ValueError('Can\'t use MU_type = {} and rank = {}'.format(self.MU_type, self.rank))

        #######################################
        if self.MU_type == 'diagonal':
            if self.MU_init_method == 'random':
                self.U_diag = nn.Parameter(self.scale * torch.randn(self.emb * self.num_models))

            elif self.MU_init_method == 'average':
                self.U_diag = nn.Parameter(torch.ones(self.emb * self.num_models))

        #######################################
        elif self.MU_type == 'low_rank':
            self.U_a = nn.Parameter(self.scale * torch.randn((self.new_emb, self.rank)) / math.sqrt(self.rank))
            self.U_b = nn.Parameter(
                self.scale * torch.randn((self.rank, self.emb * self.num_models)) / math.sqrt(self.rank))

        #######################################
        elif self.MU_type == 'diagonal_and_low_rank':
            if self.MU_init_method == 'random':
                self.U_diag = nn.Parameter(torch.zeros(self.emb * self.num_models))
                self.U_a = nn.Parameter(self.scale * torch.randn((self.new_emb, self.rank)) / math.sqrt(self.rank))
                self.U_b = nn.Parameter(
                    self.scale * torch.randn((self.rank, self.emb * self.num_models)) / math.sqrt(self.rank))

            elif self.MU_init_method == 'average':
                self.U_diag = nn.Parameter(torch.ones(self.emb * self.num_models))
                self.U_a = nn.Parameter(
                    LOW_RANK_FACTOR * self.scale * torch.zeros((self.new_emb, self.rank)) / self.rank)
                self.U_b = nn.Parameter(
                    LOW_RANK_FACTOR * self.scale * torch.randn((self.rank, self.emb * self.num_models)) / self.rank)

        #######################################
        elif self.MU_type == 'full':
            if self.MU_init_method == 'random':
                self.U = nn.Parameter(self.scale * torch.randn((self.new_emb, self.emb * self.num_models)))

            elif self.MU_init_method == 'average':
                self.U = nn.Parameter(torch.eye(self.emb).repeat(1, self.num_models))

        #######################################
        else:
            raise ValueError('Unsuppored MU_type: {}'.format(self.MU_type))

    def create_MU(self):
        if self.MU_type == 'diagonal':
            U = create_concatenated_diagonal_matrices(vec=self.U_diag, emb=self.emb, num_models=self.num_models,
                                                      dim_cat=1)

        elif self.MU_type == 'low_rank':
            U = self.U_a @ self.U_b

        elif self.MU_type == 'diagonal_and_low_rank':
            U = create_concatenated_diagonal_matrices(vec=self.U_diag, emb=self.emb, num_models=self.num_models,
                                                      dim_cat=1)
            U = U + self.U_a @ self.U_b

        elif self.MU_type == 'full':
            U = self.U

        return U

    # inputs in shape [seq_length, B, new_emb]
    # outputs in shape [B, seq_length, num_models * emb]
    # This sub-block will be at the end of the learned MU block, so the batch size is the first dimension of the output.
    def forward(self, inputs: torch.Tensor, for_features=False) -> torch.Tensor:
        # Create the M and U matrices
        U = self.create_MU()

        # Forward
        inputs = inputs.permute(1, 0, 2)  # [seq_length, B, new_emb] -> [B, seq_length, new_emb]
        if for_features:
            return inputs
        x = self.LN(inputs) @ U
        # [B, seq_length, new_emb] @ [new_emb, emb * num_models] = [B, seq_length, emb * num_models]
        return x

    # Folding M,U matrices into the weights, and return the new weights
    def fold(self, device) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            # Create the M and U matrices
            U = self.create_MU()

            merged_weights_dict = {}
            merged_weights_dict['LN_weight'] = self.LN.weight.clone().to(device)
            merged_weights_dict['LN_bias'] = self.LN.bias.clone().to(device)
            merged_weights_dict['U'] = U.clone().to(device)

        return merged_weights_dict

    # Getting the U and LN of this layer
    def get_U_LN(self):
        with torch.no_grad():
            U = self.create_MU()

        LN_state_dict = self.LN.state_dict()
        return U.detach().clone(), LN_state_dict


#############################################################
class LinearBlock(nn.Module):
    """
    A linear block.
    number_of_models: The number of models that will be merged.
    comp_factor: The merged model new hidden dimension will be emb * comp_factor.
    rank: The rank of the learned M,U matrix. -1 means full rank (emb).
    MU_type: The type of the M,U matrices.
    init_method: The method for initializing the learned parameters M and U.
    """

    def __init__(self,
                 emb: int,
                 num_models: int,
                 MU_type: str,
                 comp_factor: float = 1,
                 rank: int = -1,
                 MU_init_method: str = 'identity'):

        super(LinearBlock, self).__init__()
        # self.emb = LN_dict['weight'].shape[0]
        self.emb = emb
        self.num_models = num_models
        self.scale = self.emb ** -0.5
        self.comp_factor = comp_factor
        self.new_emb = int(self.emb * self.comp_factor)
        self.rank = rank
        self.MU_type = MU_type.lower()
        self.MU_init_method = MU_init_method.lower()

        # Learned parameters
        self.initialize_weights()

    def initialize_weights(self):
        # W in shape [new_emb, new_emb]
        #######################################
        if self.MU_type == 'diagonal':
            if self.MU_init_method == 'random':
                self.W_diag = nn.Parameter(self.scale * torch.randn(self.emb))

            if self.MU_init_method == 'average':
                self.W_diag = nn.Parameter(torch.ones(self.emb))

        #######################################
        elif self.MU_type == 'low_rank':
            self.W_a = nn.Parameter(self.scale * torch.randn((self.emb, self.rank)) / math.sqrt(self.rank))
            self.W_b = nn.Parameter(self.scale * torch.randn((self.rank, self.emb)) / math.sqrt(self.rank))

        #######################################
        elif self.MU_type == 'diagonal_and_low_rank':
            if self.MU_init_method == 'random':
                self.W_diag = nn.Parameter(torch.zeros(self.emb))
                self.W_a = nn.Parameter(self.scale * torch.randn((self.emb, self.rank)) / math.sqrt(self.rank))
                self.W_b = nn.Parameter(self.scale * torch.randn((self.rank, self.emb)) / math.sqrt(self.rank))

            if self.MU_init_method == 'average':
                self.W_diag = nn.Parameter(torch.ones(self.emb))
                self.W_a = nn.Parameter(LOW_RANK_FACTOR * self.scale * torch.zeros((self.emb, self.rank)) / self.rank)
                self.W_b = nn.Parameter(LOW_RANK_FACTOR * self.scale * torch.randn((self.rank, self.emb)) / self.rank)

        #######################################
        elif self.MU_type == 'full':
            if self.MU_init_method == 'random':
                self.W = nn.Parameter(self.scale * torch.randn(self.emb))

            elif self.MU_init_method == 'average':
                self.W = nn.Parameter(torch.eye(self.emb))

        #######################################
        else:
            raise ValueError('Unsuppored MU_type: {}'.format(self.MU_type))

    def create_W(self):
        if self.MU_type == 'diagonal':
            W = create_concatenated_diagonal_matrices(vec=self.W_diag, emb=self.emb, num_models=self.num_models,
                                                      dim_cat=1)

        elif self.MU_type == 'low_rank':
            W = self.W_a @ self.W_b

        elif self.MU_type == 'diagonal_and_low_rank':
            W_diag = create_concatenated_diagonal_matrices(vec=self.W_diag, emb=self.emb, num_models=self.num_models,
                                                           dim_cat=1)
            W = self.W_a @ self.W_b + W_diag

        elif self.MU_type == 'full':
            W = self.W

        return W

    # inputs in shape [seq_length, B, new_emb]
    # outputs in shape [seq_length, B, new_emb]
    def forward(self, inputs: torch.Tensor, for_features=False) -> torch.Tensor:
        # Create the W matrix
        W = self.create_W()

        # Forward
        # [B, seq_length, new_emb] @ [new_emb, new_emb] = [B, seq_length, new_emb]
        return inputs @ W

    # Folding M,U matrices into the weights, and return the new weights
    def fold(self, device) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            # Create the W matrix
            W = self.create_W()

            merged_weights_dict = {}
            merged_weights_dict['LN_weight'] = W.clone().to(device)

        return merged_weights_dict
