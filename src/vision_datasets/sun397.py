import os
import torch
import torchvision.datasets as datasets
from torch.utils.data import random_split



class SUN397:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 val_split_ratio=0.15,):

        torch.manual_seed(42)
        train_dataset_full = torch.load(os.path.join(location, 'SUN397', 'SUN397_train.pt'))
        self.test_dataset = torch.load(os.path.join(location, 'SUN397', 'SUN397_test.pt'))

        # Calculate the size of the splits
        val_size = int(len(train_dataset_full) * val_split_ratio)
        train_size = len(train_dataset_full) - val_size

        # Split the dataset
        self.train_dataset, self.val_dataset = random_split(train_dataset_full, [train_size, val_size])

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, num_workers=num_workers
        )

        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # Load the classes names
        with open(os.path.join(location, 'SUN397', 'SUN397_class_names.pt'), 'r') as f:
            self.classnames = [line.strip() for line in f.readlines()]


class SUN397_old:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 train_ratio=0.85):
        location = "code"
        dataset = datasets.SUN397(
            root=location, download=True, transform=preprocess
        )

        self.train_size = int(len(dataset) * train_ratio)
        self.test_size = len(dataset) - self.train_size

        self.train_dataset, self.test_dataset = torch.utils.data.random_split(dataset, [self.train_size, self.test_size])

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, num_workers=num_workers
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        idx_to_class = dict((v, k)
                            for k, v in dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace('_', ' ') for i in range(len(idx_to_class))]