import torch
import argparse
import warnings

from extraction import extract_features
from extraction import extract_features_clip
from extraction import extract_layers

if __name__ == '__main__':
    torch.manual_seed(42)

    ##############  parser ##############
    parser = argparse.ArgumentParser(description='Process some variables.')
    parser.add_argument('--model_type', type=str, default='ViT-B-16', help='The type of the model')
    parser.add_argument('--aug_factor', type=int, default=10, help='The augmentation factor')
    parser.add_argument('--with_mixup', type=bool, default=True, help='Whether to use mixup')
    parser.add_argument('--extract_type', type=str, default='all', help='The type of the extraction')
    parser.add_argument('--num_features_per_dataset', type=int, default=100, help='The number of features per dataset')
    parser.add_argument('--datasets_for_features', type=str, nargs='+',
        default=['Cars', 'DTD', 'EuroSAT', 'GTSRB', 'MNIST', 'RESISC45', 'SVHN', 'CIFAR10', 'CIFAR100'],
        help='List of datasets to use'
    )

    args = parser.parse_args()

    print()
    #####################################

    if args.extract_type == 'layers':
        extract_layers.extract_layers_from_model(model_type=args.model_type)

    elif args.model_type in ['ViT-B-16', 'ViT-L-14']:
        extract_features.feature_extraction(model_type=args.model_type,
                                            aug_factor=args.aug_factor,
                                            with_mixup=args.with_mixup,
                                            extract_type=args.extract_type,
                                            num_features_per_dataset=args.num_features_per_dataset,
                                            datasets_for_features=args.datasets_for_features)