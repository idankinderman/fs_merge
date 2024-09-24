import os
import argparse
from typing import Dict
import torch

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the vision_datasets.",
    )
    parser.add_argument(
        "--eval-vision_datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which vision_datasets to use for evaluation. Split by comma, e.g. MNIST,EuroSAT. "
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        type=lambda x: x.split(","),
        help="Which dataset(s) to patch on.",
    )

    # Experiment
    parser.add_argument(
        "--exp_num",
        type=int,
        default=None,
        help="The number of the experiment."
    )

    parser.add_argument(
        "--server_num",
        type=int,
        default=None,
        help="In what server are we running."
    )

    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="The directory to save the experiments."
    )
    parser.add_argument(
        "--ckp_dir",
        type=str,
        default=None,
        help="The directory to save the checkpoints."
    )
    parser.add_argument(
        "--path_to_save_desc",
        type=str,
        default=None,
        help="The path to the descriptor file."
    )
    parser.add_argument(
        "--loss_dir",
        type=str,
        default=None,
        help="The directory to save the loss graphs."
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default=None,
        help="The directory to save the features."
    )
    parser.add_argument(
        "--features_dir_tmp1",
        type=str,
        default=None,
        help="A tmp directory to save the features while creating them."
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="..",
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        default='/gscratch/efml/gamaga/.cache/open_clip',
        help='Directory for caching models from OpenCLIP'
    )
    parser.add_argument(
        "--use_same_pretrained",
        type=bool,
        default=True,
        help='Whether to use the same pretrained model for all vision_datasets'
    )
    parser.add_argument(
        "--make_new_heads",
        type=bool,
        default=True,
        help='Whether to make new heads for each dataset'
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="The devices to use for training and evaluation.",
    )

    # Training
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate."
    )
    parser.add_argument(
        "--lr-diag",
        type=float,
        default=None,
        help="Learning rate for the diagonal of the weight."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
        help="Weight decay"
    )
    parser.add_argument(
        "--ls",
        type=float,
        default=0.0,
        help="Label smoothing."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default='cosine_lr',
        help="What type of scheduler to use. Must be in [None, 'cosine_lr', 'steplr']"
    )
    parser.add_argument(
        "--StepLR_step_size",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--StepLR_gamma",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--distributed",
        type=str,
        default="data_parallel",
        help="What kind of distributed training to use. data_parallel or distributed_data_parallel or None",
    )

    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args
