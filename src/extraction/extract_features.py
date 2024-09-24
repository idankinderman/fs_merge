import os
import argparse
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

from datetime import datetime
import tqdm

import torch

from vision_datasets.common import maybe_dictionarize
from vision_datasets.registry import get_dataset
from vision_datasets.features_dataset import ImagesDataset, FeaturesDatasetHolder
from vision_datasets.augmentations import create_pre_process_augmentation, get_mixup_fn
from inside_vit import extract_vit_features_from_inputs
from args import parse_arguments

from modeling import ImageClassifier, ModuleWrapper
from heads import get_classification_head
from merges.average_merge import AverageMerge


def save_tensor_as_image(tensor, path):
    """
    Saves a PyTorch tensor as an image plot.

    Parameters:
    - tensor: A PyTorch tensor representing the image, of shape (H, W) or (C, H, W)
    - path: The file path where the image will be saved
    """
    # Check if the tensor has 3 dimensions (C, H, W), convert it to (H, W, C) for plotting
    tensor = tensor.cpu()
    if tensor.dim() == 3:
        # Assuming the tensor is in the format (C, H, W)
        tensor = tensor.permute(1, 2, 0)

    # Convert tensor to numpy array
    image = tensor.numpy()

    # Plotting the image
    plt.imshow(image)
    plt.axis('off')  # Turn off axis numbers and ticks

    # Saving the image
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the plot to free up memory


# Example usage
# Create a sample tensor representing an image (for example, a 2D grayscale image of size 256x256)
sample_tensor = torch.rand(256, 256)  # Change this to your actual tensor

# Specify the path where you want to save the image
save_path = 'your_image_path_here.png'  # Change this to your desired path

# Save the tensor as an image
save_tensor_as_image(sample_tensor, save_path)


# Load a specific model
def load_model(dir_path, model_name):
    model_full_name = "{}.pt".format(model_name)
    file_path = os.path.join(dir_path, 'checkpoints', model_full_name)
    model = torch.load(file_path)
    return model


class FeatureExtractorVit:
    """
    Used in order to extract features from the inner layers of a VIT.
    """
    def __init__(self,
                 model_type: str,
                 dir_path: str,
                 train_preprocess,
                 val_preprocess,
                 num_features_per_dataset: int,
                 aug_factor: int = 0,
                 with_mixup: bool = False):

        self.model_type = model_type
        self.dir_path = dir_path
        self.head_path = os.path.join(dir_path, 'heads')
        self.preprocess = val_preprocess
        self.aug_preprocess = create_pre_process_augmentation(train_preprocess)

        self.num_features_per_dataset = num_features_per_dataset
        self.aug_factor = aug_factor
        self.with_mixup = with_mixup

        self.model_path = os.path.join(self.dir_path, 'checkpoints')
        self.args = parse_arguments()
        self.args.data_location = '../data'

        self.features_dir = os.path.join(self.dir_path, "features_{}".format(num_features_per_dataset))
        Path(self.features_dir).mkdir(parents=True, exist_ok=True)

        self.loaded_args = self.load_models_args()


    def load_models_args(self):
        # Load the arguments from the file
        args = parse_arguments()
        args.data_location = '../data'
        args.model = self.model_type
        args.save = self.dir_path
        args.save_dir = self.dir_path
        args.devices = list(range(torch.cuda.device_count()))
        return args

    def get_loader(self, dataset, data_type, shuffle=False):
        if data_type == 'train':
            data_loader = getattr(dataset, 'train_loader', None)
        elif data_type == 'augmented_train':
            data_loader = getattr(dataset, 'train_aug_loader', None)
        elif data_type == 'val':
            data_loader = getattr(dataset, 'val_loader', None)
        elif data_type == 'test':
            data_loader = getattr(dataset, 'test_loader', None)
        else:
            raise Exception("Unknown data_type: {}".format(data_type))

        if data_loader is not None:
            data_loader.shuffle = shuffle
        return data_loader


    # Creating and saving the dataset that will be used for creating the features.
    # All the model's features should be created from the same images, and in the same order.
    def create_dataset_for_features(self, data_name_for_features: str, curr_features_dir: str):
        image_datasets = {}
        data_type_list = ['train', 'val']

        # Loop over the train and val
        for data_type in data_type_list:
            file_name = '{}_{}'
            file_name = file_name + '_{}'.format(data_name_for_features)

            # check if input exist
            path_input = os.path.join(curr_features_dir, file_name.format('input', data_type))
            path_labels = os.path.join(curr_features_dir, file_name.format('labels', data_type))
            if os.path.exists(path_input) and os.path.exists(path_labels):
                with open(path_input, "rb") as f:
                    all_inputs = pickle.load(f)
                with open(path_labels, "rb") as f:
                    all_labels = pickle.load(f)

                print(f"Used existing dataset {data_name_for_features} {data_type} dataset."
                      f" inputs shape: {all_inputs.shape} | labels shape: {all_labels.shape}")

                # Creating dataset from the images
                image_datasets[data_type] = ImagesDataset(all_inputs, all_labels)

            else: # Create the dataset
                num_features_per_dataset = self.num_features_per_dataset if data_type == 'train' else 64

                all_inputs = []
                all_labels = []
                curr_num_samples = 0

                dataset = get_dataset(
                    data_name_for_features,
                    preprocess=self.preprocess,
                    location=self.args.data_location,
                    batch_size=128,
                )

                self.nb_classes = len(dataset.classnames)
                print("Number of classes: ", self.nb_classes)

                data_loader = self.get_loader(dataset, data_type)

                for batch in tqdm.tqdm(data_loader, desc="Creating {} dataset from {}".format(data_type, data_name_for_features)):
                    batch = maybe_dictionarize(batch)
                    inputs = batch['images']
                    labels = batch['labels']
                    curr_num_samples += labels.shape[0]
                    if curr_num_samples > num_features_per_dataset:
                        inputs = inputs[:num_features_per_dataset - (curr_num_samples - labels.shape[0])]
                        labels = labels[:num_features_per_dataset - (curr_num_samples - labels.shape[0])]

                    all_inputs.append(inputs)
                    all_labels.append(labels)

                    if curr_num_samples >= num_features_per_dataset:  # We have enough features
                        break

                all_inputs = torch.cat(all_inputs, dim=0)
                all_labels = torch.cat(all_labels, dim=0)

                print("Finished creating the {} {} dataset. inputs shape: {} | labels shape: {}"
                      .format(data_name_for_features, data_type, all_inputs.shape, all_labels.shape))

                # Creating dataset from the images
                image_datasets[data_type] = ImagesDataset(all_inputs, all_labels)

                # Saving the dataset and the labels
                path = os.path.join(curr_features_dir, file_name.format('input', data_type))
                with open(path, "wb") as f:
                    pickle.dump(all_inputs, f)

                path = os.path.join(curr_features_dir, file_name.format('labels', data_type))
                with open(path, "wb") as f:
                    pickle.dump(all_labels, f)

        # Create train augmentation dataset
        if self.aug_factor > 0:
            image_datasets['augmented_train'] = self.create_aug_dataset_for_features(image_datasets['train'],
                                                                                     curr_features_dir,
                                                                                     data_name_for_features)
        else:
            image_datasets['augmented_train'] = None

        return FeaturesDatasetHolder(train_dataset=image_datasets['train'],
                                     train_aug_dataset=image_datasets['augmented_train'],
                                     val_dataset=image_datasets['val'],
                                     batch_size=16,
                                     num_workers=16,
                                     train_shuffle=False)


    def create_aug_dataset_for_features(self, train_dataset, curr_features_dir, data_name_for_features):
        data_type = 'augmented_train'
        path_input = os.path.join(curr_features_dir, "augmented_train_input_{}".format(data_name_for_features))
        path_labels = os.path.join(curr_features_dir, "augmented_train_labels_{}".format(data_name_for_features))
        if os.path.exists(path_input) and os.path.exists(path_labels):
            with open(path_input, "rb") as f:
                all_inputs = pickle.load(f)
            with open(path_labels, "rb") as f:
                all_labels = pickle.load(f)

            print(f"Used existing augmented dataset {data_name_for_features} {data_type} dataset."
                  f" inputs shape: {all_inputs.shape} | labels shape: {all_labels.shape}")

        else:
            all_inputs, all_labels = [], []

            train_augmented_dataset_src = ImagesDataset(train_dataset.inputs, train_dataset.labels, transform=self.aug_preprocess)

            data_loader = torch.utils.data.DataLoader(train_augmented_dataset_src, batch_size=128, shuffle=False, num_workers=16)

            mixup_fn = get_mixup_fn(nb_classes=self.nb_classes) if self.with_mixup else None

            for i in range(self.aug_factor):
                for batch in data_loader:
                    batch = maybe_dictionarize(batch)

                    inputs = batch['images'].to('cuda')
                    labels = batch['labels'].to('cuda')

                    if mixup_fn is not None:
                        inputs, labels = mixup_fn(inputs, labels)

                    all_inputs.append(inputs)
                    all_labels.append(labels)

            all_inputs = torch.cat(all_inputs, dim=0).to('cpu')
            all_labels = torch.cat(all_labels, dim=0).to('cpu')

            print("Finished creating the {} {} dataset. inputs shape: {} | labels shape: {}"
                  .format(data_name_for_features, data_type, all_inputs.shape, all_labels.shape))

            # Saving the dataset and the labels
            path = os.path.join(curr_features_dir, "augmented_train_input_{}".format(data_name_for_features))
            with open(path, "wb") as f:
                pickle.dump(all_inputs, f)

            path = os.path.join(curr_features_dir, "augmented_train_labels_{}".format(data_name_for_features))
            with open(path, "wb") as f:
                pickle.dump(all_labels, f)

        # Creating dataset from the images
        return ImagesDataset(all_inputs, all_labels)


    # Load the model and extract features from it, using the dataset.
    def extract_features_from_model(self, dataset, model_name, features_dir_curr_model, features_dir_temp, extract_type='none'):
        dataset_name = model_name.replace('finetuned_', '')
        image_encoder = load_model(self.dir_path, model_name)
        image_classifier = ImageClassifier(image_encoder, classification_head=None)
        model = ModuleWrapper(image_classifier) # extract_vit_features expects the image_encoder to be wrapped in a ModuleWrapper
        model = model.cuda()
        model.eval()

        # Extracting the features from the VIT model
        features_dict = None
        data_type_list = ['train', 'augmented_train', 'val']

        for data_type in data_type_list:
            data_loader = self.get_loader(dataset, data_type)
            if data_loader == None:
                continue

            i = -1
            samples_so_far = 0
            for batch in tqdm.tqdm(data_loader, desc="Extracting {} features".format(data_type)):
                i += 1
                samples_so_far += batch['images'].shape[0]
                batch = maybe_dictionarize(batch)
                inputs = batch['images'].to('cuda:0')
                extract_type_curr = 'none' if data_type == 'augmented_train' and samples_so_far >= 250 else extract_type
                classification_head = get_classification_head(self.loaded_args, dataset_name, image_encoder=None, head_path=self.head_path)
                features_dict = extract_vit_features_from_inputs(model, inputs, extract_type=extract_type_curr, classification_head=classification_head)

                # Saving the features in a tmp directory inside features_dir_tmp1, from each layer in different file
                for layer_name in features_dict.keys():
                    path_to_save_samples = os.path.join(features_dir_temp, '{}_{}_{}'.
                                                        format(layer_name, data_type, i))

                    with open(path_to_save_samples, "wb") as f:
                        pickle.dump(features_dict[layer_name].cpu(), f)


        # Concatenating the features of the current model, which was saved in features_dir_tmp1
        self.concat_features(features_dict.keys(), features_dir_temp, features_dir_curr_model)


    # Concatenating the features in the temporary directory features_dir_temp, and saves them in features_dir_curr_model
    def concat_features(self, layer_names, features_dir_temp, features_dir_curr_model):
        for layer_name in tqdm.tqdm(layer_names, desc="Concatenating features"):
            data_type_list = ['train', 'augmented_train', 'val']

            for data_type in data_type_list:
                features_list = []

                # Checks how much relevant files there are in the directory
                features_layer_name = '{}_{}_'.format(layer_name, data_type)
                files = os.listdir(features_dir_temp)
                matching_files = [f for f in files if f.startswith(features_layer_name)]
                num_features_files = len(matching_files)

                if num_features_files == 0:
                    continue

                # Loading all the saved features from the temp directory
                for i in range(num_features_files):
                    file_path = os.path.join(features_dir_temp, features_layer_name + str(i))
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            curr_feature_dict = pickle.load(f)
                        features_list.append(curr_feature_dict)

                # Concatenate the tensors along the batch dimension and store in the result dictionary
                features_dict_cat = torch.cat(features_list, dim=0)

                path_final_features = os.path.join(features_dir_curr_model, '{}_{}'.format(layer_name, data_type))
                with open(path_final_features, "wb") as f:
                    pickle.dump(features_dict_cat, f)


    # Deleting the tmp features
    def delete_tmp_features(self, feature_dir: str, remove_dir: bool = False):
        for filename in os.listdir(feature_dir):
            file_path = os.path.join(feature_dir, filename)
            os.remove(file_path)

        if remove_dir:
            os.rmdir(feature_dir)


    def mse_loss(self, x, y):
        squared_diff = (x - y) ** 2
        return squared_diff.mean()

def feature_extraction(model_type, aug_factor, with_mixup, extract_type, num_features_per_dataset, datasets_for_features):
    print(f"\nExtracting features from {model_type} with aug_factor {aug_factor} and with_mixup {with_mixup}"
          f" and extract_type {extract_type}\n\n")

    if model_type == 'ViT-B-16':
        exp_name = '4_1_24_diff_pretrained_finetune'

    elif model_type == 'ViT-L-14':
        exp_name = '9_3_24_diff_pretrained_finetuned'

    dir_path = os.path.join('..', 'experiments', model_type, exp_name)
    model_names = ["finetuned_{}".format(data_name) for data_name in datasets_for_features]

    #########################################################
    image_encoder = load_model(dir_path, model_names[0])

    # print("image_encoder.val_preprocess: ", image_encoder.val_preprocess, "\n")
    # print("image_encoder.train_preprocess: ", image_encoder.train_preprocess, "\n")

    print("\n\n", "=" * 20, "Creating features with {} images per dataset. aug_factor = {}. total = {}"
          .format(num_features_per_dataset, aug_factor,
                  num_features_per_dataset + aug_factor * num_features_per_dataset),
          "=" * 20, "\n")

    feature_extractor = FeatureExtractorVit(model_type=model_type,
                                            dir_path=dir_path,
                                            train_preprocess=image_encoder.train_preprocess,
                                            val_preprocess=image_encoder.val_preprocess,
                                            num_features_per_dataset=num_features_per_dataset,
                                            aug_factor=aug_factor,
                                            with_mixup=with_mixup)

    # Extract features using all the vision_datasets
    for data_name_for_features in datasets_for_features:
        print("\n\n", "-" * 20, "Creating features with {} images".format(data_name_for_features), "-" * 20, "\n")

        # Path for current features
        features_dir_curr_dataset = os.path.join(feature_extractor.features_dir,
                                                 "dataset_{}".format(data_name_for_features))
        Path(features_dir_curr_dataset).mkdir(parents=True, exist_ok=True)

        # Path for temporary features
        features_dir_temp = os.path.join(features_dir_curr_dataset, "temp_features")
        Path(features_dir_temp).mkdir(parents=True, exist_ok=True)

        print("features_dir_curr_dataset: ", features_dir_curr_dataset)
        print("features_dir_temp: ", features_dir_temp)

        # The images we will use for creating the features
        dataset_for_features = feature_extractor.create_dataset_for_features(
            data_name_for_features=data_name_for_features,
            curr_features_dir=features_dir_curr_dataset)

        # Extract features for the models
        for model_name in model_names:
            if data_name_for_features != model_name.replace('finetuned_', ''):
                continue

            print("\n", "Creating features with {} images for {} model".format(data_name_for_features, model_name))

            # Path for current features
            features_dir_curr_model = os.path.join(features_dir_curr_dataset, "model_{}".format(model_name))
            Path(features_dir_curr_model).mkdir(parents=True, exist_ok=True)
            print("features_dir_curr_model: ", features_dir_curr_model)

            feature_extractor.extract_features_from_model(dataset=dataset_for_features,
                                                          features_dir_curr_model=features_dir_curr_model,
                                                          features_dir_temp=features_dir_temp,
                                                          model_name=model_name,
                                                          extract_type=extract_type)

            # Delete the temporary features
            feature_extractor.delete_tmp_features(feature_dir=features_dir_temp)

        #########################################################

        # Writing descriptor of the features
        now = datetime.now()
        date_time = now.strftime("%Y_%m_%d")
        hour_time = now.strftime("%H_%M")
        curr_time = "{}_{}".format(date_time, hour_time)

        features_desc = "date: {} \nnumber of features per model: {}\nmodel type: {}\nmodel names: {} " \
                        "features created from {} \naug_factor {}, with_mixup: {}" \
            .format(curr_time, num_features_per_dataset, model_type, model_names, datasets_for_features,
                    aug_factor, with_mixup)
        desc_dir = os.path.join(feature_extractor.features_dir, 'features_desc.txt')
        with open(desc_dir, 'a+') as f:
            f.write(features_desc)

if __name__ == '__main__':
    """
    This used in order to extract features from the inner layers of a number of VIT.
    The 'dir_path' determines the directory in which the models are saved.
    The 'model_names' determine the models from which the features will be extracted.
    The features are saved in new directory inside the 'dir_path' directory.
    """
    torch.manual_seed(42)

    ##############  parser ##############
    parser = argparse.ArgumentParser(description='Process some variables.')
    parser.add_argument('--model_type', type=str, default='ViT-B-16', help='The type of the model')
    parser.add_argument('--aug_factor', type=int, default=20, help='The augmentation factor')
    parser.add_argument('--with_mixup', type=bool, default=True, help='Whether to use mixup')
    parser.add_argument('--extract_type', type=str, default='none', help='The type of the extraction')
    parser.add_argument('--num_features_per_dataset', type=str, default=50, help='The number of features per dataset')
    args = parser.parse_args()
    model_type = args.model_type
    aug_factor = args.aug_factor
    with_mixup = args.with_mixup
    num_features_per_dataset = args.num_features_per_dataset
    extract_type = args.extract_type

    feature_extraction(model_type=model_type, aug_factor=aug_factor, with_mixup=with_mixup, extract_type=extract_type,
         num_features_per_dataset=num_features_per_dataset)


