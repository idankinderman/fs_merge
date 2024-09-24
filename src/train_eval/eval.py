import os
import json
import tqdm
import time

from sklearn.metrics import confusion_matrix

import torch
import numpy as np

import utils
from vision_datasets.common import get_dataloader, maybe_dictionarize
from heads import get_classification_head, get_multi_heads
from modeling import ImageEncoder, ImageClassifier, MultiHeadImageClassifier

from vision_datasets.registry import get_dataset


def var_mean(tensor: torch.Tensor):
    return torch.mean(torch.var(tensor, dim=0))

def softmax_entropy(x):
    return -(x.softmax(-1) * x.log_softmax(-1)).sum(-1).mean()

def mse_loss(x, y):
        squared_diff = (x - y) ** 2
        return squared_diff.mean()


def get_dataloader(dataset, data_type):
    if data_type == 'train':
        dataloader = dataset.train_loader
    elif data_type == 'val':
        dataloader = dataset.val_loader
    elif data_type == 'early_stopping':
        dataloader = dataset.early_stopping_loader
    elif data_type == 'test':
        dataloader = dataset.test_loader
    else:
        raise ValueError(f'Unknown data_type: {data_type}')
    return dataloader

# Evaluate the loss of a model on a given dataset.
# Mostly for MSE loss on the feature reconstruction task, so no classification head is needed.
# If we are evaluate a merge layer, then what_is_trained == 'merge_layer' and num_of_models is the number of models in the merge.
def eval_single_dataset_loss(model, dataset, compute_loss, loss_type, data_type='train', what_is_trained=None, num_of_models=1,
                             with_all_plots=False):
    model.eval()
    dataloader = get_dataloader(dataset, data_type)

    sum_loss = 0.0
    sum_var = 0.0
    sum_ent = 0.0
    sum_loss_split = [0.0] * num_of_models
    sum_var_split = [0.0] * num_of_models
    sum_ent_split = [0.0] * num_of_models
    inner_loss_dict = {}
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inner_target_scales = getattr(dataset, "inner_target_scales", None)
            loss, output, targets, loss_dict = compute_loss(model, batch, return_output=True,
                                                 inner_target_scales=inner_target_scales)
            sum_loss += loss.item()

            # Compute the variance and entropy of the output
            if with_all_plots:
                sum_var += var_mean(output).item()
                sum_ent += softmax_entropy(output).item()

            # If you use loss with inner features, show those inner losses
            if loss_type in ['rec_with_inner_att', 'rec_with_inner_mlp', 'rec_with_inner_att_ids', 'rec_with_inner_mlp_ids']:
                for key in loss_dict:
                    if key in inner_loss_dict:
                        inner_loss_dict[key] += loss_dict[key]
                    else:
                        inner_loss_dict[key] = loss_dict[key]

    # Compute the mean loss, variance and entropy
    mean_loss = sum_loss / len(dataloader)
    mean_var = sum_var / len(dataloader)
    mean_ent = sum_ent / len(dataloader)
    sum_loss_split = [loss / len(dataloader) for loss in sum_loss_split]
    sum_var_split = [var / len(dataloader) for var in sum_var_split]
    sum_ent_split = [ent / len(dataloader) for ent in sum_ent_split]
    metrics = {f'{data_type}-loss': mean_loss, f'{data_type}-loss-split': sum_loss_split,
               f'{data_type}-var': mean_var, f'{data_type}-var-split': sum_var_split,
               f'{data_type}-entropy': mean_ent, f'{data_type}-entropy-split': sum_ent_split}

    if loss_type in ['rec_with_inner_att', 'rec_with_inner_mlp', 'rec_with_inner_att_ids', 'rec_with_inner_mlp_ids']:
        for key in inner_loss_dict:
            metrics["inner-{}".format(key)] = inner_loss_dict[key] / len(dataloader)

    return metrics


# Evaluate the model on a specific dataset.
# The task is classification, so a classification head is needed.
# dataset - the dataset to evaluate on. If None, will load the dataset according to dataset_name.
def eval_single_dataset(image_encoder, dataset_name, args, head_path=None, head_num=None, classification_head=None,
                        data_type='train', dataset=None, with_prints=True, U_output=None):
    print("head_num in eval_single_dataset", head_num)
    # Prepare the model
    if classification_head is None:
        classification_head = get_classification_head(args, dataset_name, image_encoder=image_encoder,
                                                      head_path=head_path, head_num=head_num)
    model = ImageClassifier(image_encoder, classification_head, U_output=U_output)
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model.to('cuda')
    model.eval()

    # Load the dataset
    if dataset is None:
        dataset = get_dataset(
            dataset_name,
            model.module.val_preprocess,
            location=args.data_location,
            #batch_size=min(int(args.batch_size * 1.5), 1152)
            batch_size=1040
        )

    dataloader = get_dataloader(dataset, data_type)
    device = args.device
    print("dataloader with {} batches, each of size {}".format(len(dataloader), dataloader.batch_size))

    with torch.no_grad():
        top1, correct, n, entropy = 0., 0., 0., 0.
        for i, data in enumerate(dataloader):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)
            logits = utils.get_logits(x, model)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)
            entropy += softmax_entropy(logits).item() * y.size(0)

        top1 = 100 * correct / n
        entropy = entropy / n

    metrics = {f'top1-{data_type}': top1, f'entropy-{data_type}': entropy}
    if with_prints:
        print(f'Done evaluating on {dataset_name}. Accuracy: {top1:.2f}%')
    
    return metrics

# Evaluate the model on all vision_datasets in args.eval_datasets.
# In out merge there is a U_output matrix in shape [out_dim, num_models * out_dim],
# which reconstruct the original output of the models from the merged output.
def evaluate(image_encoder, args, U_output=None, head_path=None, head_num=None):
    if args.eval_datasets is None:
        return
    info = vars(args)
    metric_dict = {}
    for dataset_num, dataset_name in enumerate(args.eval_datasets):

        if U_output is not None:
            # U_output is a matrix of shape [out_dim, num_models * out_dim]
            # Need to take the part of U_output that corresponds to the current dataset
            out_dim = U_output.shape[0]
            U_curr = U_output[:, dataset_num * out_dim: (dataset_num + 1) * out_dim]
        else:
            U_curr = None

        print('Evaluating on', dataset_name)
        data_type_list = ['val'] if dataset_name == 'ImageNet' else ['val', 'test']
        for data_type in data_type_list:
            results = eval_single_dataset(image_encoder, dataset_name, args, data_type=data_type,
                                          U_output=U_curr, with_prints=False, head_path=head_path,
                                          head_num=head_num)

            for key, val in results.items():
                #if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                    #print(f"{dataset_name} {key}: {val:.4f}")
                info[dataset_name + ':' + key] = val

                if 'top1' in key:
                    metric_dict['{}_{}_acc'.format(dataset_name, data_type)] = results[key]

                elif 'entropy' in key:
                    metric_dict['{}_{}_entropy'.format(dataset_name, data_type)] = results[key]

    return info, metric_dict



# Evaluate the model on all vision_datasets in args.eval_datasets. Use the multi-head classifier.
# It means that we get logits from all heads, and choose as prediction the entry with the highest value.
# In out merge there is a U_output matrix in shape [out_dim, num_models * out_dim],
# which reconstruct the original output of the models from the merged output.
def multi_head_evaluate(image_encoder, args, U_output=None, head_path=None, head_num=None):
    if args.eval_datasets is None:
        return

    head_list, U_output_list = get_multi_heads(args, U_output=U_output, head_path=head_path, head_num=head_num)
    model = MultiHeadImageClassifier(image_encoder, head_list, U_output_list)
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model.to('cuda')
    model.eval()

    data_type_list = ['val', 'test']
    metric_dict = {}
    for dataset_num, dataset_name in enumerate(args.eval_datasets):
        dataset = get_dataset(
            dataset_name,
            model.module.val_preprocess,
            location=args.data_location,
            # batch_size=min(int(args.batch_size * 1.5), 1152)
            batch_size=1000
        )

        head_place = model.module.get_head_place(dataset_num) # Before the right head we have #head_place entries

        for data_type in data_type_list:
            dataloader = get_dataloader(dataset, data_type)
            device = args.device

            with torch.no_grad():
                top1, correct, n = 0., 0., 0.
                for i, data in enumerate(dataloader):
                    data = maybe_dictionarize(data)
                    x = data['images'].to(device)
                    y = data['labels'].to(device)
                    y = y + head_place
                    logits = model(x)
                    pred = logits.argmax(dim=1, keepdim=True).to(device)
                    correct += pred.eq(y.view_as(pred)).sum().item()
                    n += y.size(0)

                top1 = 100 * correct / n

            metric_dict['joint_{}_{}_acc'.format(dataset_name, data_type)] = top1

    return metric_dict


def perform_eval(args):
    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, train_dataset)

    # Check if checkpoints already exist
    zs_path = os.path.join(args.save, train_dataset, 'checkpoint_0.pt')
    ft_path = os.path.join(args.save, train_dataset, f'checkpoint_{args.epochs}.pt')
    if os.path.exists(zs_path) and os.path.exists(ft_path):
        print(f'Skipping fine-tuning because {ft_path} exists.')
        return zs_path, ft_path

    assert train_dataset is not None, "Please provide a training dataset."
    if args.load is not None and args.load.endswith('pt'):
        image_encoder = ImageEncoder.load(args.load)
    else:
        print('Building image encoder.')
        image_encoder = ImageEncoder(args, keep_lang=False)

    print("The image encoder is:\n", image_encoder, "\n\n")

    classification_head = get_classification_head(args, train_dataset)

    print("The classification_head is:\n", classification_head, "\n\n")

    model = ImageClassifier(image_encoder, classification_head)

    print("The model is:\n", model, "\n\n")

    model.freeze_head()

    preprocess_fn = model.train_preprocess

    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)

    model = model.cuda()

    # Evaluate
    image_encoder = model.module.image_encoder
    evaluate(image_encoder, args)

    if args.save is not None:
        zs_path = os.path.join(ckpdir, 'zeroshot.pt')
        ft_path = os.path.join(ckpdir, 'finetuned.pt')
        image_encoder.save(ft_path)
        return zs_path, ft_path
