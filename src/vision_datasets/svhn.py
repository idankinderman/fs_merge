import os
import torch
from torchvision.datasets import SVHN as PyTorchSVHN
from torch.utils.data import random_split


class SVHN:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 val_split_ratio=0.15,):

        # to fit with repo conventions for location
        torch.manual_seed(42)
        modified_location = os.path.join(location, 'svhn')

        train_dataset_full = PyTorchSVHN(
            root=modified_location,
            download=True,
            split='train',
            transform=preprocess
        )

        # Calculate the size of the splits
        val_size = int(len(train_dataset_full) * val_split_ratio)
        train_size = len(train_dataset_full) - val_size

        # Split the dataset
        self.train_dataset, self.val_dataset = random_split(train_dataset_full, [train_size, val_size])

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = PyTorchSVHN(
            root=modified_location,
            download=True,
            split='test',
            transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        self.classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
