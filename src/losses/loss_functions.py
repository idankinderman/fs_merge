import torch

from utils import scale_and_multiply


def create_loss_function(degree, loss_weights, loss_layer_num, loss_type='rec', reg_coeff=0.0):
    if loss_type.lower() in ['rec', 'mse', 'rec_with_ids']:

        if degree % 2 == 0:
            def loss_function(x, y):
                diff = (x - y) ** degree
                return diff.mean()

        elif degree % 2 == 1:
            def loss_function(x, y):
                diff = torch.abs(x - y) ** degree
                return diff.mean()


    elif loss_type.lower() in ['rec_with_inner_att', 'rec_with_inner_mlp', 'rec_with_inner_att_ids',
                               'rec_with_inner_mlp_ids']:
        # The loss also takes into account the inner features
        layer_name = 'att' if 'att' in loss_type.lower() else 'mlp'

        # inner_target_scales: The scales of the inner targets.
        # ids: ids of the images (for what dataset they belong).
        # We assume that logits, targets and all inner features were already cut according to the ids (which chooses the right model).
        def loss_function(logits, inner_features, batch, inner_target_scales):
            ids = batch['ids']  # (batch_size)
            loss_dict = {}
            loss = ((batch['labels'] - logits) ** degree).mean()
            loss_dict['output'] = loss.item()
            # for layer_num, layer_num_weight in zip(loss_layer_num, loss_weights):
            for i in range(len(loss_layer_num)):
                layer_num = loss_layer_num[i]
                residual = batch[f'inner_target_{layer_num}_{layer_name}'] - inner_features[
                    f'inner_{layer_num}_{layer_name}']  # (batch_size, seq_len, emb_dim)
                curr_loss = (residual ** degree).mean(dim=(1, 2))  # (batch_size)
                curr_loss = scale_and_multiply(data=curr_loss, ids=ids, weight_vector=1/inner_target_scales[layer_num])  # normalize the loss by the scale of the inner target
                curr_loss = loss_weights[i] * curr_loss.mean()  # the regularization term
                loss += curr_loss
                loss_dict[f'{layer_num}_{layer_name}'] = curr_loss.item()

            return loss, loss_dict

    else:
        raise ValueError("loss_type must be either 'rec' or 'rec_with_merged_features'")

    return loss_function



def create_loss_function_for_dist(loss_layer_num, loss_type='rec', reg_coeff=0.0):
    if loss_type.lower() in ['rec', 'mse', 'rec_with_ids']:
        def loss_function(x, y):
            diff = (x - y) ** 2
            return diff.mean()


    elif loss_type.lower() in ['rec_with_inner_att', 'rec_with_inner_mlp', 'rec_with_inner_att_ids',
                               'rec_with_inner_mlp_ids']:
        # The loss also takes into account the inner features
        layer_name = 'att' if 'att' in loss_type.lower() else 'mlp'

        def loss_function(logits, inner_features, batch):
            loss_dict = {}
            loss = ((batch['labels'] - logits) ** 2).mean() # (batch_size, output_dim)
            loss_dict['output'] = loss.item()

            for i in range(len(loss_layer_num)):
                layer_num = loss_layer_num[i]
                residual = batch[f'inner_target_{layer_num}_{layer_name}'] - inner_features[f'inner_{layer_num}_{layer_name}']  # (batch_size, seq_len, emb_dim)
                curr_loss = reg_coeff * (residual ** 2).mean()
                loss += curr_loss
                loss_dict[f'{layer_num}_{layer_name}'] = curr_loss.item()

            return loss, loss_dict

    else:
        raise ValueError("loss_type must be either 'rec' or 'rec_with_merged_features'")

    return loss_function