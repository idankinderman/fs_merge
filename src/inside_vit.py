import torch

import math
import pickle
import os
from pathlib import Path
import tqdm

Tensor = torch.Tensor

from vision_datasets.common import maybe_dictionarize

def linear(input, weight, bias=None):
    r"""
linear(input, weight, bias=None) -> Tensor

Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

Shape:

    - Input: :math:`(*, in\_features)` where `*` means any number of
      additional dimensions, including none
    - Weight: :math:`(out\_features, in\_features)` or :math:`(in\_features)`
    - Bias: :math:`(out\_features)` or :math:`()`
    - Output: :math:`(*, out\_features)` or :math:`(*)`, based on the shape of the weight
"""
    return input @ weight.T + bias

#
# multihead attention
#
def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b = None,
):
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            # q torch.Size([T, batch_size, d]) | w torch.Size([d*3, d]) | b torch.Size([d*3])
            return linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


# Process images chunk by chunk, similar to what the convolutional layers do, only in a linear manner.
# input_tensor - B,3,N,N
# conv_weights - embedding_size,3,patch_size,patch_size
def perform_conv_as_linear(input_tensor, conv_weights):
    B, _, N, _ = input_tensor.shape
    emb, _, patch_size, _ = conv_weights.shape

    conv_weights_reshape =  conv_weights.view(emb, 3 * patch_size * patch_size)

    # Calculate the number of chunks along the last two dimensions
    chunks_y = N // patch_size
    chunks_x = N // patch_size

    # Split the tensor along the last two dimensions
    results_list = []
    conv_inputs = []
    for i in range(chunks_y):
        for j in range(chunks_x):
            sub_tensor = input_tensor[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            result, conv_input = conv_as_linear(sub_tensor, conv_weights_reshape, patch_size) # shape: B , emb_size
            results_list.append(result)
            conv_inputs.append(conv_input) # [B, 3 * patch_size * patch_size]

    output = torch.stack(results_list, dim=2)
    conv_inputs = torch.stack(conv_inputs, dim=0) # [B * greed**2, 3 * patch_size * patch_size]
    return output, conv_inputs


# Converting the conv action to linear layer action.
# sub_tensor - B,3,patch_size,patch_size
# conv_weights_reshape - embedding_size,3*patch_size*patch_size
def conv_as_linear(sub_tensor, conv_weights_reshape, patch_size):
    # Reshape sub_tensor to shape B, 3*patch_size*patch_size
    B, _, _, _ = sub_tensor.shape
    sub_tensor_reshaped = sub_tensor.reshape(B, 3 * patch_size * patch_size)

    # Matrix multiplication
    # [B, 3 * patch_size * patch_size] @ [3 * patch_size * patch_size , emb_size] = [B , emb_size]
    output = sub_tensor_reshaped @ conv_weights_reshape.T

    return output, sub_tensor_reshaped


# Using inputs the create features from the VIT model
def extract_vit_features_from_inputs(model, inputs, extract_type='before_fc', classification_head=None):
    if extract_type not in ['after_ln', 'all', 'before_ln', 'before_skip', 'before_fc', 'none']:
        raise ValueError("extract_type should be one of ['after_ln', 'before_ln', 'before_skip', 'before_fc', 'all', 'none']")

    features_dict = {}
    #conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
    # x = model.module.image_encoder.model.visual.conv1(inputs) # shape = [*, width, grid, grid]
    # Takes each patch with 3 channels, and gives it width channels, no bias
    # x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    #features_dict['input'] = inputs.clone().detach().cpu()  # tmp
    x, conv_inputs = perform_conv_as_linear(inputs, model.module.image_encoder.model.visual.conv1.weight)
    x = x.to(inputs.device)

    if extract_type in ['all', 'before_fc']:
        # [B * greed**2, 3 * patch_size * patch_size] -> # [B, greed**2, 3 * patch_size * patch_size]
        features_dict['conv_inputs'] = conv_inputs.clone().detach().cpu().reshape(x.shape[0], -1, conv_inputs.shape[-1])

    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    # So now we have T = grid ** 2 "tokens", each in size width

    if extract_type == 'all':
        features_dict['conv'] = x.clone().detach().cpu()

    x = torch.cat(
        [model.module.image_encoder.model.visual.class_embedding.to(x.dtype) +
         torch.zeros(x.shape[0], 1, x.shape[-1],
         dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    # Concatenate the special 'class' token, which is learnable

    if extract_type == 'all':
        pass
        #features_dict['cls'] = x.clone().detach().cpu()

    x = x + model.module.image_encoder.model.visual.positional_embedding.to(x.dtype) # T, d
    # Adding the positional embeddings, which are learnable

    if extract_type == 'all':
        pass
        #features_dict['pos'] = x.clone().detach().cpu()

    x = model.module.image_encoder.model.visual.ln_pre(x)
    # Using LayerNorm

    if extract_type in ['all', 'before_ln']:
        features_dict['ln_pre'] = x.clone().detach().cpu()

    x = x.permute(1, 0, 2)  # NLD -> LND

    # Now using the transformer
    #x = model.module.image_encoder.model.visual.transformer(x)
    for layer_num, ResidualAttentionBlock in enumerate(model.module.image_encoder.model.visual.transformer.resblocks):
        ###########
        # MultiheadAttention part
        z = ResidualAttentionBlock.ln_1(x) # LayerNorm
        if extract_type in ['after_ln', 'all', 'before_fc']:
            features_dict['LN-att-{}'.format(layer_num)] = z.clone().detach().cpu().permute(1, 0, 2)  # LND->NLD

        # MultiheadAttention
        #z = ResidualAttentionBlock.attn(z, z, z, need_weights=False, attn_mask=None)[0] # nn.MultiheadAttention(d_model, n_head)
        tgt_len, bsz, embed_dim = z.shape
        num_heads = ResidualAttentionBlock.attn.num_heads
        head_dim = embed_dim // num_heads

        q, k, v = _in_projection_packed(z, z, z,
                                        ResidualAttentionBlock.attn.in_proj_weight,
                                        ResidualAttentionBlock.attn.in_proj_bias)

        if extract_type == 'all':
            features_dict['q-{}'.format(layer_num)] = q.clone().detach().cpu().permute(1, 0, 2)  # LND->NLD
            features_dict['k-{}'.format(layer_num)] = k.clone().detach().cpu().permute(1, 0, 2)  # LND->NLD
            features_dict['v-{}'.format(layer_num)] = v.clone().detach().cpu().permute(1, 0, 2)  # LND->NLD

        # reshape q, k, v for multihead attention and make em batch first
        q = q.reshape(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.reshape(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        v = v.reshape(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

        # update source sequence length after adjustments
        src_len = k.size(1)

        B, Nt, E = q.shape
        q_scaled = q / math.sqrt(E)
        attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = attn_output_weights.softmax(dim=-1)
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)

        if extract_type in ['before_fc', 'all']:
            features_dict['before-out-att-{}'.format(layer_num)] = attn_output.clone().detach().cpu().reshape(bsz, tgt_len, embed_dim) # N,L,D

        attn_output = linear(attn_output,
                             ResidualAttentionBlock.attn.out_proj.weight,
                             ResidualAttentionBlock.attn.out_proj.bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        #if extract_type == 'all':
        #    features_dict['scores-{}'.format(layer_num)] = attn_output_weights.clone().detach().cpu().permute(1, 0, 2)  # LND->NLD

        if extract_type in ['all', 'before_skip', 'before_fc']:
            features_dict['attn-{}'.format(layer_num)] = attn_output.clone().detach().cpu().permute(1, 0, 2)  # LND->NLD


        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.mean(dim=1)

        #y = x + ResidualAttentionBlock.ln_attn(z) #LayerNorm(d_model) if scale_attn else nn.Identity(), so this is Identity
        y = x + attn_output

        if extract_type in ['all', 'before_ln']:
            features_dict['attn_out-{}'.format(layer_num)] = y.clone().detach().cpu().permute(1, 0, 2)  # LND->NLD


        ###########
        # MLP part
        t = ResidualAttentionBlock.ln_2(y) #LayerNorm
        if extract_type in ['after_ln', 'all', 'before_fc']:
            features_dict['LN-mlp-{}'.format(layer_num)] = t.clone().detach().cpu().permute(1, 0, 2)  # LND->NLD

        t = ResidualAttentionBlock.mlp[0](t) # nn.Linear(d_model, mlp_width)
        # the mlp[1] is Identity (and not LayerNorm) so not needed

        if extract_type == 'all':
            features_dict['fc1-{}'.format(layer_num)] = t.clone().detach().cpu().permute(1, 0, 2)  # LND->NLD

        t = ResidualAttentionBlock.mlp[2](t) # act_layer()
        #t = gelu(t)

        if extract_type in ['all', 'before_fc']:
            features_dict['gelu-{}'.format(layer_num)] = t.clone().detach().cpu().permute(1, 0, 2)  # LND->NLD

        t = ResidualAttentionBlock.mlp[3](t) # nn.Linear(mlp_width, d_model)

        if extract_type in ['all', 'before_skip', 'before_fc']:
            features_dict['fc2-{}'.format(layer_num)] = t.clone().detach().cpu().permute(1, 0, 2)  # LND->NLD

        x = y + t

        if extract_type in ['all', 'before_ln']:
            features_dict['mlp_out-{}'.format(layer_num)] = x.clone().detach().cpu().permute(1, 0, 2)  # LND->NLD

    x = x.permute(1, 0, 2)  # LND -> NLD
    x_cls = model.module.image_encoder.model.visual.ln_post(x[:, 0, :]) # LayerNorm
    #x_all = model.module.image_encoder.model.visual.ln_post(x) # just for the features

    if extract_type in ['after_ln', 'all', 'before_fc']:
        features_dict['LN-out'] = x_cls.clone().detach().cpu() # ND

    if model.module.image_encoder.model.visual.proj is not None:
        x_cls = x_cls @ model.module.image_encoder.model.visual.proj # linear from width to output_dim

    features_dict['output'] = x_cls.clone().detach().cpu()

    if classification_head:
        logits = classification_head(x_cls)
        features_dict['logits'] = logits.clone().detach().cpu()

    return features_dict
