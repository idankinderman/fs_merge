import os
import torch
from tqdm import tqdm
from pathlib import Path

import open_clip

from vision_datasets.templates import get_templates
from vision_datasets.registry import get_dataset

from modeling import ClassificationHead, ImageEncoder


def build_classification_head(model, dataset_name, template, data_location, device):
    template = get_templates(dataset_name)
    logit_scale = model.logit_scale
    dataset = get_dataset(
        dataset_name,
        None,
        location=data_location
    )
    model.eval()
    model.to(device)

    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(dataset.classnames):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = open_clip.tokenize(texts).to(device) # tokenize
            embeddings = model.encode_text(texts) # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()
        
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

    return classification_head


# Will prefer to check if head exists in args.save_dir.
# If not, will look at head_path.
# If not, will create a head by using image_encoder.
def get_classification_head(args, dataset, image_encoder, head_path=None, head_num=None):
    Path(os.path.join(args.save_dir, 'heads')).mkdir(parents=True, exist_ok=True)
    if head_num is not None:
        # a trained head
        filename = os.path.join(args.save_dir, 'heads', f'trained_head_{dataset}_{head_num}.pt')
    else:
        filename = os.path.join(args.save_dir, 'heads', f'head_{dataset}.pt')
    if os.path.exists(filename):
        return ClassificationHead.load(filename)

    if head_path is not None and os.path.exists(head_path):
        full_head_path = os.path.join(head_path, f'head_{dataset}.pt')
        return ClassificationHead.load(full_head_path)

    print(f'Did not find classification head for {args.model} on {dataset} at {filename}, building one from scratch.')
    model = image_encoder.model

    template = get_templates(dataset)
    classification_head = build_classification_head(model, dataset, template, args.data_location, args.device)
    os.makedirs(args.save, exist_ok=True)
    classification_head.save(filename)
    return classification_head


# Get a list of classification heads for each dataset in args.eval_datasets.
# Also create a list of U_outputs for them. U_output is in shape [out_dim, num_models * out_dim],
def get_multi_heads(args, U_output=None, head_path=None, head_num=None):
    head_list = []
    U_output_list = []
    if U_output is None:
        U_output_list = None

    for dataset_num, dataset_name in enumerate(args.eval_datasets):
        classification_head = get_classification_head(args, dataset_name, image_encoder=None,
                                                      head_path=head_path, head_num=head_num)
        head_list.append(classification_head)

        if U_output is not None:
            # U_output is a matrix of shape [out_dim, num_models * out_dim]
            # Need to take the part of U_output that corresponds to the current dataset
            out_dim = U_output.shape[0]
            U_curr = U_output[:, dataset_num * out_dim: (dataset_num + 1) * out_dim]
            U_output_list.append(U_curr)

    return head_list, U_output_list


