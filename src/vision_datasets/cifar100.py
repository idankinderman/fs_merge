import os
import torch
from torchvision.datasets import CIFAR100 as PyTorchCIFAR100
from torch.utils.data import random_split


class CIFAR100:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 val_split_ratio=0.15,):

        torch.manual_seed(42)

        train_dataset_full = PyTorchCIFAR100(
            root=location, download=True, train=True, transform=preprocess
        )

        # Calculate the size of the splits
        val_size = int(len(train_dataset_full) * val_split_ratio)
        train_size = len(train_dataset_full) - val_size

        # Split the dataset
        self.train_dataset, self.val_dataset = random_split(train_dataset_full, [train_size, val_size])

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, num_workers=num_workers
        )

        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        self.test_dataset = PyTorchCIFAR100(
            root=location, download=True, train=False, transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.classnames = self.test_dataset.classes

        print("CIFAR100: train", len(self.train_dataset), "test", len(self.test_dataset), "classes", len(self.classnames))


