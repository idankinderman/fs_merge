import torch
from torch.utils.data import Dataset
import warnings

def abs_mean(tensor: torch.Tensor):
    return torch.mean(torch.abs(tensor))

def square_mean(tensor: torch.Tensor):
    return torch.mean(tensor ** 2)

def ones_func(tensor: torch.Tensor):
    return torch.Tensor([1.0])


class FeaturesDatasetHolder:
    """
    This class is used to create and hold the features dataloaders.
    """

    def __init__(self, train_dataset, val_dataset, batch_size, num_workers,
                 train_aug_dataset=None, test_dataset=None, early_stopping_dataset=None, train_shuffle=True):

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.scales = getattr(train_dataset, 'scales', None)
        self.inner_target_scales = getattr(train_dataset, "inner_target_scales", None)

        self.train_dataset = train_dataset
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers
        )

        if train_aug_dataset is not None:
            # This dataset, if exists, already have the augmentations.
            self.train_aug_dataset = train_aug_dataset
            self.train_aug_loader = torch.utils.data.DataLoader(
                self.train_aug_dataset, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers
            )

        if val_dataset is not None:
            self.val_dataset = val_dataset
            self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )

        if test_dataset is not None:
            self.test_dataset = test_dataset
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )

        if early_stopping_dataset is not None:
            self.early_stopping_dataset = early_stopping_dataset
            self.early_stopping_loader = torch.utils.data.DataLoader(
                self.early_stopping_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )


class ImagesDataset(Dataset):
    """
    This class is used to handle the images that we will create features from.
    The inputs are images in shape [B, C, H, W].
    The targets are the labels of the images.
    The data_ids are the ids of the images ().
    """
    def __init__(self, inputs, labels, transform=None):
        self.inputs = inputs
        self.labels = labels
        self.transform = transform
        self.num_samples = inputs.shape[0]
        self.scales = None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.inputs[idx]

        if self.transform:
            image = self.transform(image)

        return {'images' : image, 'labels' : self.labels[idx]}


class DatasetFeaturesVIT(Dataset):
    """
    This class is used to handle the data which trains a VIT merger.
    The inputs are images in shape [B, C, H, W], or features in shape [B, seq_length, emb].
    The targets_list is a list of features, each in shape [B, seq_length, emb], or [B, emb] in case of the last layer.
    The ids are the ids of the images (for what dataset they belong).

    normalize_scale: 0 - don't normalize, -1 - normalize according to the first features,
    else - normalize according to normalize_scale value.
    target_dim_cat: The dimension to concatenate the targets.
    """
    def __init__(self, inputs, targets_list, ids, normalize_scale=0.0, scales=None, target_dim_cat=0, print_info=True):
        # Inputs
        self.inputs = inputs
        self.scales = scales

        # Helpers
        self.target_dim_cat = target_dim_cat
        if self.scales is None:
            targets_list = self.normalize_target_scale(targets_list, normalize_scale)
        else:
            targets_list = self.normalize_target_according_to_list(targets_list, self.scales)
        self.targets = torch.cat(targets_list, dim=target_dim_cat)
        if ids is None:
            ids = torch.zeros(inputs.shape[0])
        self.ids = ids

        self.num_samples = inputs.shape[0]

        if print_info:
            self.print_info()

    def print_info(self):
        print("inputs.shape: {} | targets.shape: {} | ids.shape: {}".format(self.inputs.shape, self.targets.shape, self.ids.shape))
        print()

    def normalize_target_scale(self, targets_list, normalize_scale):
        if normalize_scale == 0:
            # Won't normalize
            pass

        elif normalize_scale == -1:
            # Normalize according to the first features
            self.scales = []
            abs_mean_first = abs_mean(targets_list[0])
            self.scales.append(1.0)

            for i in range(1, len(targets_list)):
                curr_scale = (abs_mean_first / abs_mean(targets_list[i])).item()
                targets_list[i] = targets_list[i] * curr_scale
                self.scales.append(curr_scale)

        elif normalize_scale > 0:
            # Normalize according to the normalize_scale value
            self.scales = []
            for i in range(len(targets_list)):
                curr_scale = (normalize_scale / abs_mean(targets_list[i])).item()
                targets_list[i] = targets_list[i] * curr_scale
                self.scales.append(curr_scale)

        else:
            raise ValueError("normalize_scale must be -1, 0 or positive float.")

        return targets_list


    def normalize_target_according_to_list(self, targets_list, scales):
        for i in range(len(targets_list)):
            targets_list[i] = targets_list[i] * scales[i]
        return targets_list


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {'images' : self.inputs[idx], 'labels' : self.targets[idx], 'ids' : self.ids[idx]}


class DatasetFeaturesBERT(DatasetFeaturesVIT):
    """
    This class is used to handle the data which trains a BERT merger.
    The inputs are tokens in shape [B, seq_length].
    attention_mask and token_type_ids are in shape [B, seq_length].
    The targets_list is a list of features, each in shape [B, emb].
    The ids are the ids of the text (for what dataset they belong).

    normalize_scale: 0 - don't normalize, -1 - normalize according to the first features,
    else - normalize according to normalize_scale value.
    target_dim_cat: The dimension to concatenate the targets.
    """

    def __init__(self, inputs, attention_mask, token_type_ids, targets_list, ids, normalize_scale=0.0, scales=None, target_dim_cat=0):
        super().__init__(inputs, targets_list, ids, normalize_scale, scales, target_dim_cat, print_info=False)
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.print_info()

    def print_info(self):
        print("inputs.shape: {} | targets.shape: {} | ids.shape: {} | attention_mask.shape: {} | token_type_ids.shape: {}"
              .format(self.inputs.shape, self.targets.shape, self.ids.shape, self.attention_mask.shape, self.token_type_ids.shape))
        print()

    def __getitem__(self, idx):
        return {'input_ids' : self.inputs[idx], 'targets' : self.targets[idx], 'ids' : self.ids[idx],
                'attention_mask' : self.attention_mask[idx], 'token_type_ids' : self.token_type_ids[idx]}


class DatasetInnerFeaturesVIT(DatasetFeaturesVIT):
    """
    This class is used to handle the data which trains the merger of the VIT pre-process stage.
    It use as target the many inner features of the VIT model, from different layers.
    inputs: are images in shape [B, C, H, W], or features in shape [B, seq_length, emb].
    targets_list: a list of features, each in shape [B, seq_length, emb], or [B, emb] in case of the last layer.
    inner_target_dict: a dictionary, with keys of layer_num, and values of lists of PyTorch tensors with shape [B, seq_length, emb].
    inner_target_layers: a list of integers, each is the layer number from which the inner features are taken.
    ids: ids of the images (for what dataset they belong).
    loss_type: The type of loss to use.
    scale_inner_type: how to compute the scale of the inner features. Then it will be used for weighting the MSE of the inner features in the loss.
    normalize_scale: 0 - don't normalize, -1 - normalize according to the first features,
    else - normalize according to normalize_scale value.
    target_dim_cat: The dimension to concatenate the targets.
    inner_target_scales: The scales of the inner targets. If None, will calculate them.
    """
    def __init__(self, inputs, targets_list, inner_target_dict, inner_target_layers, ids, loss_type, scale_inner_type,
                 normalize_scale=0.0, scales=None, inner_target_scales=None, target_dim_cat=0):

        # Inputs
        self.inputs = inputs
        self.scales = scales

        # Helpers
        self.target_dim_cat = target_dim_cat
        self.loss_type = loss_type.lower()
        self.scale_inner_type = scale_inner_type
        self.layer_name = 'att' if 'att' in self.loss_type else 'mlp'
        self.num_models = len(targets_list)
        self.inner_target_layers = inner_target_layers

        # Targets scale
        if self.scales is None:
            targets_list = self.normalize_target_scale(targets_list, normalize_scale)
        else:
            targets_list = self.normalize_target_according_to_list(targets_list, self.scales)

        # Inner targets scale
        if inner_target_scales is None:
            self.inner_target_scales =  self.get_inner_targets_scales(inner_target_dict)
        else:
            self.inner_target_scales = inner_target_scales

        for key in self.inner_target_scales:
            self.inner_target_scales[key] = self.inner_target_scales[key].cuda()

        # Targets
        self.targets = torch.cat(targets_list, dim=target_dim_cat)
        for layer_num in inner_target_dict:
            inner_target_dict[layer_num] = torch.cat(inner_target_dict[layer_num], dim=target_dim_cat)
            print("at the end: inner_target_dict[{}] shape is {}".format(layer_num, inner_target_dict[layer_num].shape))
        self.inner_target_dict = inner_target_dict

        # Ids
        if ids is None:
            ids = torch.zeros(inputs.shape[0])
        self.ids = ids
        self.num_samples = inputs.shape[0]

        print("inputs.shape: {} | targets.shape: {} | ids.shape: {}".format(self.inputs.shape, self.targets.shape, self.ids.shape))
        print()


    def __getitem__(self, idx):
        item = {'images': self.inputs[idx], 'labels': self.targets[idx], 'ids' : self.ids[idx]}
        # Add elements from inner_target_list
        for layer_num in self.inner_target_layers:
            item[f'inner_target_{layer_num}_{self.layer_name}'] = self.inner_target_dict[layer_num][idx]
        return item


    def get_inner_targets_scales(self, inner_target_dict):
        inner_targets_scales = {}

        if self.scale_inner_type == 'l1':
            scale_func = abs_mean
        elif self.scale_inner_type == 'l2':
            scale_func = square_mean
        elif self.scale_inner_type == 'ones':
            scale_func = ones_func
        else:
            raise ValueError("scale_inner_type must be 'l1', 'l2' or 'ones'.")

        for layer_num in self.inner_target_layers:
            inner_targets_scales[layer_num] = [scale_func(elem).item() for elem in inner_target_dict[layer_num]]
            inner_targets_scales[layer_num] = torch.tensor(inner_targets_scales[layer_num])

        return inner_targets_scales


class DatasetFeaturesWithMergeVIT(Dataset):
    """
    This class is used to handle the data which trains the merger of the VIT pre-process stage.
    The inputs are images in shape [B, C, H, W], or features in shape [B, seq_length, emb].
    The targets_list is a list of features, each in shape [B, seq_length, emb * num_models],
    or [B, emb * num_models] in case of the last layer.
    The merged_targets are the targets after the merge, in shape [B, seq_length, emb].
    """
    def __init__(self, inputs, targets_list, merged_targets):
        self.inputs = inputs
        self.targets = torch.cat(targets_list, dim=-1)
        self.merged_targets = merged_targets
        self.num_samples = inputs.shape[0]


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {'images' : self.inputs[idx], 'labels' : self.targets[idx], 'merged_features' : self.merged_targets[idx]}


class DatasetFeaturesImageText(Dataset):
    """
    This class is used to create a dataset that contains images and texts, and their corresponding features.
    """
    def __init__(self, images, texts, features_images, features_texts):
        self.images = images # B x 3 x img_size x img_size
        self.texts = texts # B x tokens_per_text
        self.features_images = features_images # B x emb
        self.features_texts = features_texts # B x emb
        self.num_samples = len(self.images)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {'images' : self.images[idx], 'texts' : self.texts[idx],
                'features_images' : self.features_images[idx], 'features_texts' : self.features_texts[idx]}