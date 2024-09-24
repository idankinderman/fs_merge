import os
import time
from datetime import datetime
from pathlib import Path
import json

from typing import List

import torch

from args import parse_arguments
from vision_datasets.common import get_dataloader, maybe_dictionarize
from vision_datasets.registry import get_dataset
from train_eval.trainer import Trainer
from train_eval.eval import evaluate
from modeling import ImageEncoder, ImageClassifier, MultiHeadImageClassifier
from utils import cosine_lr, LabelSmoothing, print_args_in_groups, set_seed
from heads import get_classification_head

from visualization import save_train_loss_plots, save_acc_plot

import vision_datasets as datasets

def creating_new_finetune_exp(data_location : str,
                              model_type : str,
                              exp_name : str,
                              datasets_to_eval : List[str],
                              batch_size : int,
                              lr : float,
                              pretrained_dict : dict,
                              use_same_pretrained : bool,
                              make_new_heads : bool = False):

    # Parsing the arguments
    args = parse_arguments()
    args.data_location = data_location
    args.model = model_type
    args.exp_name = exp_name
    args.save = f'../experiments/{model_type}'
    args.eval_datasets = datasets_to_eval
    args.batch_size = batch_size
    args.scheduler_type = 'cosine_lr'
    args.lr = lr
    args.use_same_pretrained = use_same_pretrained
    args.make_new_heads = make_new_heads
    args.devices = list(range(torch.cuda.device_count()))

    # creating dir for saving the results
    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d")
    hour_time = now.strftime("%H_%M")
    curr_time = "{}_{}".format(date_time, hour_time)

    save_dir = os.path.join(args.save, args.exp_name)
    args.save_dir = save_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    ckp_dir = os.path.join(save_dir, 'checkpoints')
    args.ckp_dir = ckp_dir
    Path(ckp_dir).mkdir(parents=True, exist_ok=True)

    loss_dir = os.path.join(save_dir, 'loss_graphs')
    args.loss_dir = loss_dir
    Path(loss_dir).mkdir(parents=True, exist_ok=True)

    # Saving descriptor
    descriptor = curr_time + "\n" + print_args_in_groups(args) + "\n"
    if use_same_pretrained:
        descriptor += "Same pre-trained model\n"
    else:
        descriptor += "Different pre-trained model\n"
    descriptor += "Pre-trained models: {}\n".format(str(pretrained_dict))

    path_to_save_desc = os.path.join(save_dir, "descriptor.txt")
    args.path_to_save_desc = path_to_save_desc
    with open(path_to_save_desc, 'w') as f:
        f.write(descriptor)

    # Save the args
    args_dict = vars(args)
    path_to_save_args = os.path.join(args.save_dir, "args.json")
    with open(path_to_save_args, "w") as f:
        json.dump(args_dict, f)

    return args


# Evaluate zero-shot (pre-trained) model on all the vision_datasets.
# If use_same_pretrained is True, then the same pre-trained model is used for all the vision_datasets.
def zero_shot_eval(args, datasets_to_train, pretrained_dict, use_same_pretrained):
    zs_eval_text = ""
    for i, dataset in enumerate(datasets_to_train):
        print("\n------Evaluating zero-shot model number {}, later will finetune on {} ------\n".format(i, dataset))
        pretrained = pretrained_dict[dataset]

        # Fetching the Image Encoder
        image_encoder = load_pre_trained_model(args, pretrained_dict[dataset])

        # Evaluate zero-shot model
        print("\nEvaluating zero-shot model.")
        _, metric_dict = evaluate(image_encoder, args)

        if use_same_pretrained:
            zs_eval_text += "\n\nEvaluating zero-shot model with pretrained {}.\n{}\n". \
                format(pretrained, str(metric_dict))

            # Saving zero-shot model
            #zs_model_path = os.path.join(args.ckp_dir, 'zero_shot.pt')
            #image_encoder.save(zs_model_path)
            break

        else:
            zs_eval_text += "\n\nEvaluating zero-shot model number {}, used for task {}, with pretrained {}.\n{}\n".\
                format(i, dataset, pretrained, str(metric_dict))

            # Save the zero-shot model
            #zs_model_path = os.path.join(args.ckp_dir, dataset + "_zero_shot.pth")
            #image_encoder.save(zs_model_path)

            # Delete the classification heads
            #if make_new_heads:
            #    delete_classification_heads(args)

        del image_encoder

    with open(args.path_to_save_desc, 'a+') as f:
        f.write(zs_eval_text)


def finetune(args, image_encoder, pre_trained_model):
    # Getting the model ready
    train_dataset = args.train_dataset
    assert train_dataset is not None, "Please provide a training dataset."
    classification_head = get_classification_head(args, train_dataset, image_encoder=image_encoder)
    model = ImageClassifier(image_encoder, classification_head)

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    # Train and plot the loss
    print("\nTraining model on {} dataset.".format(train_dataset))
    trainer = Trainer(args=args, loss_fn=loss_fn, print_per_epoch=2)
    model = trainer.train_model(model=model, epochs=args.epochs)
    save_train_loss_plots(loss_list=trainer.train_loss, path_to_save=args.loss_dir, title=train_dataset)
    save_acc_plot(train_acc=trainer.train_acc, val_acc=trainer.val_acc, test_acc=trainer.test_acc,
                       path_to_save=args.loss_dir, title=train_dataset)

    # Evaluate finetuned model
    now = datetime.now()
    eval_date_time = now.strftime("%Y_%m_%d")
    eval_hour_time = now.strftime("%H_%M")
    eval_time = "{}_{}".format(eval_date_time, eval_hour_time)

    print("\nEvaluating finetuned model.")
    image_encoder = model.module.image_encoder
    _, results = evaluate(image_encoder, args)
    ft_eval_text = f"\n\n{eval_time} : Evaluating {train_dataset} finetuned model, pre-trained from {pre_trained_model}:\n{results}\n"
    with open(args.path_to_save_desc, 'a+') as f:
        f.write(ft_eval_text)

    # Saving finetuned model
    ft_model_path = os.path.join(args.ckp_dir, 'pre_trained_{}_finetuned_{}.pt'.format(pre_trained_model, train_dataset))
    image_encoder.save(ft_model_path)

    print("\nDone, the experiment was saved in {}".format(args.save_dir))
    return model


def delete_classification_heads(args):
    heads_dir = os.path.join(args.save_dir, 'heads')
    for filename in os.listdir(heads_dir):
        print("Deleting file {}".format(filename))
        file_path = os.path.join(heads_dir, filename)
        os.remove(file_path)


def load_pre_trained_model(args, pretrained_name):
    if pretrained_name.startswith("our_"):
        pretrained_path = "/home/edank/task-vectors/pre_trained/{}/checkpoints/{}.pt". \
            format(args.model, pretrained_name.replace('our_', ''))
        image_encoder = torch.load(pretrained_path)
    else:
        image_encoder = ImageEncoder(args, keep_lang=False, pretrained=pretrained_name)

    return image_encoder

if __name__ == '__main__':

    ######## Experiment parameters ########
    set_seed(seed=42)
    exp_name = 'exp'
    data_location = '../data'
    #models = ['ViT-B-32', 'ViT-B-16', 'ViT-L-14']
    model_type = 'ViT-L-14'
    datasets_to_train = ['EuroSAT', 'GTSRB', 'MNIST', 'Cars', 'DTD', 'RESISC45', 'CIFAR10', 'CIFAR100', 'SVHN']
    datasets_to_eval = datasets_to_train
    use_same_pretrained = False
    make_new_heads = False

    if model_type == 'ViT-B-16':
        pretrained_dict = {'EuroSAT': 'our_pre_trained2', 'GTSRB': 'our_dist3', 'Cars': 'openai',
                           'MNIST': 'our_dist4', 'RESISC45': 'our_dist5', 'SVHN': 'our_dist6',
                           'CIFAR10': 'our_dist7', 'CIFAR100': 'our_dist7', 'DTD': 'our_dist8'}

    else:
        raise ValueError("The model type is not supported.")

    if use_same_pretrained:
        pretrained_dict = {dataset: 'openai' for dataset in datasets_to_train}

    ######## Training parameters ########
    batch_size = 32*torch.cuda.device_count()
    lr = 1e-5

    epochs = {
        'Cars': 36,
        'EuroSAT': 14,
        'GTSRB': 10,
        'DTD': 77,
        'MNIST': 6,
        'RESISC45': 13,
        'SUN397': 15,
        'SVHN': 5,
        'ImageNet': 4,
        'CIFAR10': 10,
        'CIFAR100': 30,
    }

    if model_type == 'ViT-L-14':
        for dataset in epochs:
            epochs[dataset] = epochs[dataset] + 4

    ######## The experiment ########
    args = creating_new_finetune_exp(data_location=data_location,
                                    model_type=model_type,
                                    exp_name=exp_name,
                                    datasets_to_eval=datasets_to_eval,
                                    batch_size=batch_size,
                                    lr=lr,
                                    pretrained_dict=pretrained_dict,
                                    use_same_pretrained=use_same_pretrained,
                                    make_new_heads=make_new_heads)


    # Evaluate zero-shot (pre-trained) model on all the vision_datasets.
    zero_shot_eval(args, datasets_to_train=datasets_to_train, pretrained_dict=pretrained_dict, use_same_pretrained=use_same_pretrained)
    print("\n", "-"*30, "Done with the zero-shot model", "-"*30, "\n")
    #raise ValueError

    for dataset in datasets_to_train:
        print('='*100, '\n', f'Finetuning {model_type} on {dataset}', '\n', '='*100)
        args.epochs = epochs[dataset]
        args.train_dataset = dataset
        args.batch_size = batch_size

        image_encoder = load_pre_trained_model(args, pretrained_dict[dataset])

        # Finetuning, evaluating and saving the model
        model = finetune(args, image_encoder)
        del model, image_encoder

    print("\n", "-" * 30, "Done with the finetuning", "-" * 30, "\n")