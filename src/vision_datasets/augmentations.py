import torch
from torchvision import transforms

from timm.data import Mixup
import random
import numpy as np
from PIL import ImageFilter, ImageOps

##############################################
# This code is taken from deit\augment.py
# https://github.com/facebookresearch/deit
##############################################

def create_pre_process_augmentation(preprocess):
    mean = np.array(preprocess.transforms[3].mean)
    std = np.array(preprocess.transforms[3].std)

    mean2 = -1 * mean / std
    std2 = 1 / std

    img_size = 224
    color_jitter = 0.2
    #primary_tfl = [preprocess.transforms[0], preprocess.transforms[1]]
    primary_tfl = []
    secondary_tfl = [transforms.Normalize(mean2.tolist(), std2.tolist()), # To cancel the normalization
                     transforms.ToPILImage(),
                     transforms.RandomCrop(img_size, padding=4, padding_mode='reflect'),
                     transforms.RandomHorizontalFlip(),
                     transforms.RandomChoice([gray_scale(p=1.0),
                                              Solarization(p=1.0),
                                              GaussianBlur(p=1.0)]),
                     #transforms.ColorJitter(color_jitter, color_jitter, color_jitter),
                     ]
    final_tfl = [transforms.ToTensor(), preprocess.transforms[3]]

    transform_final = transforms.Compose(primary_tfl + secondary_tfl + final_tfl)
    return transform_final

class gray_scale(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img



class horizontal_flip(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2 ,activate_pred=False):
        self.p = p
        self.transf = transforms.RandomHorizontalFlip(p=1.0)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img


def get_mixup_fn(nb_classes, mixup=0.8, cutmix=1.0, cutmix_minmax=None, mixup_prob=1.0, mixup_switch_prob=0.5,
                 mixup_mode='batch', smoothing=0.1):

    return Mixup(mixup_alpha=mixup, cutmix_alpha=cutmix, cutmix_minmax=cutmix_minmax,
                     prob=mixup_prob, switch_prob=mixup_switch_prob, mode=mixup_mode,
                     label_smoothing=smoothing, num_classes=nb_classes)