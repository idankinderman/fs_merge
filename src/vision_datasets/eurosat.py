import os
import torch
import torchvision.datasets as datasets
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Dataset, random_split

import numpy as np

from PIL import Image
from glob import glob
import re

def pretify_classname(classname):
    l = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', classname)
    l = [i.lower() for i in l]
    out = ' '.join(l)
    if out.endswith('al'):
        return out + ' area'
    return out


class EuroSATBase:
    def __init__(self,
                 preprocess,
                 test_split,
                 location='~/vision_datasets',
                 batch_size=32,
                 num_workers=16,
                 val_split_ratio=0.15,):

        # Data loading code
        torch.manual_seed(42)
        traindir = os.path.join(location, 'EuroSAT')

        train_dataset_full = datasets.ImageFolder(f"{traindir}/train", transform=preprocess)

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

        self.test_dataset = datasets.ImageFolder(f"{traindir}/test", transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )

        #Getting the classes names
        import itertools
        import random
        images_paths = [glob(f'{traindir}/train/{folder}/*.jpg') for folder in os.listdir(f"{traindir}/train")]
        images_paths = list(itertools.chain.from_iterable(images_paths))
        random.shuffle(images_paths)
        self.classes_names = {class_name: label for label, class_name in enumerate(os.listdir(f"{traindir}/train"))}
        idx_to_class = dict((v, k)
                            for k, v in self.classes_names.items())
        self.classnames = [idx_to_class[i].replace('_', ' ') for i in range(len(idx_to_class))]

        ours_to_open_ai = {
            'annualcrop': 'annual crop land',
            'forest': 'forest',
            'herbaceousvegetation': 'brushland or shrubland',
            'highway': 'highway or road',
            'industrial': 'industrial buildings or commercial buildings',
            'pasture': 'pasture land',
            'permanentcrop': 'permanent crop land',
            'residential': 'residential buildings or homes or apartments',
            'river': 'river',
            'sealake': 'lake or sea',
        }
        for i in range(len(self.classnames)):
            self.classnames[i] = ours_to_open_ai[self.classnames[i].lower()]

        del train_dataset_full


class EuroSAT(EuroSATBase):
    def __init__(self,
                 preprocess,
                 location='~/vision_datasets',
                 batch_size=32,
                 num_workers=16):
        super().__init__(preprocess, 'test', location, batch_size, num_workers)


class EuroSATVal(EuroSATBase):
    def __init__(self,
                 preprocess,
                 location='~/vision_datasets',
                 batch_size=32,
                 num_workers=16):
        super().__init__(preprocess, 'val', location, batch_size, num_workers)
