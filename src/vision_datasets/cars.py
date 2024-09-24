import os
import torch
import torchvision.datasets as datasets
from torch.utils.data import random_split


class Cars:
    # Was taken from https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16,
                 val_split_ratio=0.15,):

        # Data loading code
        torch.manual_seed(42)
        data_dir = os.path.join(location, 'car_data')
        train_dataset_full = datasets.ImageFolder(f"{data_dir}/train", transform=preprocess)

        # Calculate the size of the splits
        val_size = int(len(train_dataset_full) * val_split_ratio)
        train_size = len(train_dataset_full) - val_size

        # Split the dataset
        self.train_dataset, self.val_dataset = random_split(train_dataset_full, [train_size, val_size])

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = datasets.ImageFolder(f"{data_dir}/test", transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )

        idx_to_class = dict((v, k)
                            for k, v in train_dataset_full.class_to_idx.items())

        self.classnames = [idx_to_class[i].replace(
            '_', ' ') for i in range(len(idx_to_class))]

        del train_dataset_full



