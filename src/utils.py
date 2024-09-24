import os

import torch
import pickle
from tqdm import tqdm
import math
import random

import numpy as np


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def torch_load_old(save_path, device=None):
    with open(save_path, 'rb') as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def torch_save(model, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.cpu(), save_path)


def torch_load(save_path, device=None):
    model = torch.load(save_path, map_location='cuda:0')
    if device is not None:
        model = model.to(device)
    return model



def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, 'to'):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


def get_probs(inputs, classifier):
    if hasattr(classifier, 'predict_proba'):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


def print_args_in_groups(args, group_size=5):
    # Convert the Namespace to a list of arguments as strings
    args_as_list = []
    for arg, value in vars(args).items():
        args_as_list.append(f"--{arg} {value}")

    text = ""
    for i in range(0, len(args_as_list), group_size):
        text += ' '.join(args_as_list[i:i+group_size])
    return text


def set_seed(seed, fully_deterministic=True):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if fully_deterministic:
            torch.backends.cudnn.deterministic = True



class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# slice the data according to the ids
# data: (batch_size, num_models * emb)
# ids: (batch_size)
# return: (batch_size, emb)
def batch_index_slicing(data, ids, num_models):
    batch_size = data.shape[0]
    emb = data.shape[-1] // num_models
    #print("batch_index_slicing. data.shape: {} | ids.shape: {} | emb: {} | num_models: {} | batch_size: {}".format(data.shape, ids.shape, emb, num_models, batch_size))
    data = data.reshape(batch_size, num_models, emb)
    data = data.transpose(1, 2) # (batch_size, emb, num_models)
    dummy = ids.unsqueeze(1).unsqueeze(2).expand(ids.size(0), data.size(1), 1) # (batch_size, emb, 1)
    data = data.gather(2, dummy) # (batch_size, emb, 1)
    data = data.squeeze(2) # (batch_size, emb)
    return data


# slice the data according to the ids
# data: (batch_size, seq_len, num_models * emb)
# ids: (batch_size)
# return: (batch_size, seq_len, emb)
def batch_seq_index_slicing(data, ids, num_models):
    batch_size, seq_len = data.shape[0], data.shape[1]
    emb = data.shape[-1] // num_models
    #print("batch_seq_index_slicing. data.shape: {} | ids.shape: {} | emb: {} | num_models: {} | batch_size: {} | seq_len: {}".format(data.shape, ids.shape, emb, num_models, batch_size, seq_len))
    data = data.reshape(batch_size, seq_len, num_models, emb)
    data = data.transpose(2, 3) # (batch_size, seq_len, emb, num_models)
    dummy = ids.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(ids.size(0), data.size(1), data.size(2), 1) # (batch_size, seq_len, emb, 1)
    data = data.gather(3, dummy) # (batch_size, seq_len, emb, 1)
    data = data.squeeze(3) # (batch_size, seq_len, emb)
    return data


# scales a PyTorch tensor based on a list of scale factors and indices.
# ids: (batch_size)
# weight_vector: (num_models)
# data: (batch_size)
def scale_and_multiply(data, ids, weight_vector):
    # Create a tensor of weights according to ids using inner_target_scales
    weights = weight_vector.index_select(0, ids)
    # Perform element-wise multiplication
    data = data * weights
    return data
