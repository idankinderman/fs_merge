from datetime import datetime
import os
import shutil
from pathlib import Path
import json
import pickle
import argparse
from typing import List

from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from transformers import BertForSequenceClassification

from extraction.extract_layers import LayersExtractorVit

from train_eval.eval import evaluate, eval_single_dataset, multi_head_evaluate
from train_eval.eval_text_task import evaluate_text_datasets
from train_eval.eval_COCO import eval_clip_on_coco
from train_eval.classification_head_trainer import train_classification_head

from args import parse_arguments
from vision_datasets.features_dataset import FeaturesDatasetHolder, ImagesDataset, DatasetFeaturesVIT,\
    DatasetFeaturesWithMergeVIT, DatasetInnerFeaturesVIT, DatasetFeaturesBERT, DatasetFeaturesImageText


class GeneralMerge(metaclass=ABCMeta):
    """
    Merging two nets into one.
    """
    def __init__(self,
                 model_type : str,
                 experiment_name : str,
                 experiment_dir : str,
                 path_for_models : str,
                 models_to_merge : List[str],
                 models_indexes: List[int] | None = None,
                 datasets_to_eval : List[str] | None = None,
                 transformer_type: str = None,
                 pre_trained: str = None,
                 batch_size : int = None,
                 features_dir: str = 'features',
                 learn_tasks_sequentially: bool = False,
                 train_classification_head: bool = False,
                 descriptor : str = None):

        print("start init")
        now = datetime.now()
        date_time = now.strftime("%Y_%m_%d")
        hour_time = now.strftime("%H_%M_%S")
        exp_full_name = "{}_{}_{}".format(date_time, hour_time, experiment_name)
        self.params = {
            # This experiment
            'experiment_name': experiment_name,
            'exp_full_name': exp_full_name,
            'experiment_dir': experiment_dir,
            'path_for_models': path_for_models,
            'path_for_layers': os.path.join(path_for_models, 'layers'),
            'features_dir': features_dir,
            'descriptor': descriptor,
            'date_time': date_time,
            'hour_time': hour_time,
            'models_to_merge' : models_to_merge,
            'models_indexes' : models_indexes,
            'num_models_to_merge' : len(models_to_merge),
            'datasets_to_eval' : datasets_to_eval if datasets_to_eval else models_to_merge,

            # Data
            'batch_size' : batch_size,

            # Model
            'model_type' : model_type,
            'transformer_type': transformer_type,
            'pre_trained': pre_trained,

            # Other
            'train_classification_head': train_classification_head,
            'learn_tasks_sequentially': learn_tasks_sequentially,
            'devices' : list(range(torch.cuda.device_count())),
        }

        #self.load_models()
        self.load_args()
        if self.params['model_type'] != 'bert':
            self.params['use_same_pretrained'] = getattr(self.loaded_args, 'use_same_pretrained', True)

        self.metric_dicts = None
        self.mean_train_acc = [0]
        self.mean_train_features_acc = []
        self.mean_val_acc = []
        self.mean_test_acc = []
        self.mean_train_entropy = []
        self.mean_val_entropy = []
        self.mean_test_entropy = []
        self.merged_model_loss_dict = {}

        # Sanity check
        if self.params['models_indexes'] is not None and \
                len(self.params['models_indexes']) != len(set(self.params['models_indexes'])):
            raise Exception("The indexes of the models to merge are not unique.")

        if self.params['model_type'] == 'bert':
            self.params['path_for_models'] = "/home/edank/text-transformers/merging-text-transformers-main/models/trained/multiberts"

        if self.params['models_indexes'] is None:
            self.params['models_indexes'] = [None] * len(self.params['models_to_merge'])

        if self.params['model_type'].lower() == 'clip' and\
                (self.params['transformer_type'] is None or self.params['pre_trained'] is None):
            raise ValueError("For CLIP models, transformer_type and pre_trained must be required.")


    def create_merge_dir(self):
        self.params['exp_full_name'] += "_{}".format(self.params['merge_type'])
        self.params['path_to_save'] = os.path.join(os.path.dirname(__file__), '..', '..', 'merge_nets',
                                                   self.params['model_type'], self.params['experiment_dir'],
                                                   self.params['exp_full_name'])
        Path(self.params['path_to_save']).mkdir(parents=True, exist_ok=True)

        self.params['path_to_save_desc'] = os.path.join(self.params['path_to_save'], 'descriptor.txt')
        self.params['path_to_save_params'] = os.path.join(self.params['path_to_save'], 'params')



    # Load all the models in the checkpoint directory
    def load_all_models(self):
        #self.zero_shot_model = torch.load(os.path.join(directory_path, 'zero_shot.pt'))
        self.models_dict = {}
        for i, model_name in enumerate(self.params['models_to_merge']):
            self.models_dict[model_name] = self.load_model(model_name, i)


    def get_all_models_state_dicts(self):
        models_state_dicts = {}
        for i, model_name in enumerate(self.params['models_to_merge']):
            model = self.load_model(model_name, i)
            models_state_dicts[model_name] = model.state_dict()
        return models_state_dicts


    # Load a specific model
    def load_model(self, model_name, model_number=None):
        print(model_name, model_number)
        if self.params['model_type'] == 'bert':
            path = f"{self.params['path_for_models']}/{model_name}/seed_{self.params['models_indexes'][model_number]}"
            model = BertForSequenceClassification.from_pretrained(path)
            self.params['bert_config'] = model.config
            model = model.bert
        else:
            directory_path = os.path.join(self.params['path_for_models'], 'checkpoints')
            model_full_name = "{}.pt".format(model_name)
            file_path = os.path.join(directory_path, model_full_name)
            model = torch.load(file_path)
        return model


    def build_path_for_specific_features(self, features_path, model_name, model_num, what_to_load):
        if self.params['model_type'] != 'bert':
            train_path = os.path.join(features_path, f"dataset_{model_name}", f"{what_to_load}_train_{model_name}")
            val_path = os.path.join(features_path, f"dataset_{model_name}", f"{what_to_load}_val_{model_name}")
            aug_train_path = os.path.join(features_path, f"dataset_{model_name}", f"augmented_train_{what_to_load}_{model_name}")
            early_stopping_path = None
            if what_to_load == 'output':
                train_path = os.path.join(features_path, f"dataset_{model_name}", f"model_finetuned_{model_name}", "output_train")
                val_path = os.path.join(features_path, f"dataset_{model_name}", f"model_finetuned_{model_name}", "output_val")
                aug_train_path = os.path.join(features_path, f"dataset_{model_name}", f"model_finetuned_{model_name}", "output_augmented_train")

        else:
            if self.params['model_type'] != 'bert':
                base_path = os.path.join(self.params['path_for_models'], model_name,
                                         f"seed_{self.params['models_indexes'][model_num]}",
                                         f"features_{self.params['num_features_train']}")
            else:
                base_path = os.path.join(self.params['path_for_models'], model_name,
                                         f"seed_{self.params['models_indexes'][model_num]}",
                                         f"features_{self.params['num_features_train']}")

            train_path = os.path.join(base_path, f"{what_to_load}_train")
            val_path = os.path.join(base_path, f"{what_to_load}_eval")
            aug_train_path = None
            early_stopping_path = os.path.join(base_path, f"{what_to_load}_early_stopping")

        return train_path, val_path, aug_train_path, early_stopping_path

    # Load features dataset
    # dataset_size: the number of samples to load from each dataset
    # simple_loading: if True, load the first dataset_size samples from the dataset
    def load_feature_dataset(self, dataset_path, dataset_size):
        if dataset_path is None:
            return None

        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)

        dataset = dataset[:dataset_size]
        dataset.requires_grad = False

        return dataset


    def save_merged_model(self, U_output=None, U_output_layer=None):
        if self.params['model_type'] == 'bert':
            self.merged_model.save_pretrained(self.params['path_to_save'], 'merged_model')
        else:
            print("Saved merged model in ", self.params['path_to_save'])
            torch.save(self.merged_model.cpu(), os.path.join(self.params['path_to_save'], 'merged_model.pt'))
            if U_output is not None:
                torch.save(U_output, os.path.join(self.params['path_to_save'], 'U_output.pt'))
            if U_output_layer is not None:
                torch.save(U_output_layer, os.path.join(self.params['path_to_save'], 'U_output_layer.pt'))

    def delete_saved_models(self):
        model_files = ['merged_model.pt', 'U_output.pt', 'U_output_layer.pt']
        for model_file in model_files:
            file_path = os.path.join(self.params['path_to_save'], model_file)
            if os.path.exists(file_path):
                os.remove(file_path)

    def extract_layers_from_merged_model(self):
        print("Extracting layers from the merged model.")
        layers_extractor = LayersExtractorVit(dir_path=self.params['path_to_save'], models_path=self.params['path_to_save'])
        layers_extractor.extract_layers(model_name='merged_model')

    def delete_directory(self, path_to_dir):
        if os.path.exists(path_to_dir) and os.path.isdir(path_to_dir):
            shutil.rmtree(path_to_dir)


    def get_model_hyper_parameters(self, model):
        if self.params['model_type'] == 'bert':
            self.params['num_layers'] = len(model.encoder.layer)
            self.params['num_heads'] = model.encoder.layer[0].attention.self.num_attention_heads
            self.params['head_size'] = model.encoder.layer[0].attention.self.attention_head_size
            self.params['mlp_activation'] = model.encoder.layer[0].intermediate.intermediate_act_fn
            self.params['pooled_activation'] = model.pooler.activation
            self.params['dict_size'] = model.embeddings.word_embeddings.weight.shape[0]
            self.params['emb'] = model.embeddings.word_embeddings.weight.shape[1]
            self.params['out_dim'] = model.pooler.dense.weight.shape[0]
            self.params['patch_size'] = None
            self.params['seq_len'] = None

            print(
                f"Model hyper-parameters: num_layers {self.params['num_layers']} | num_heads {self.params['num_heads']} "
                f"| head_size {self.params['head_size']} | mlp_activation {self.params['mlp_activation']} "
                f"| pooled_activation {self.params['pooled_activation']} | dict_size {self.params['dict_size']} "
                f"| emb {self.params['emb']} | out_dim {self.params['out_dim']}")

        elif self.params['model_type'] == 'clip':
            pass

        else: # ViT
            self.params['num_layers'] = len(model.model.visual.transformer.resblocks)
            self.params['num_heads'] = model.model.visual.transformer.resblocks[0].attn.num_heads
            self.params['mlp_activation'] = model.model.visual.transformer.resblocks[0].mlp[2]
            self.params['emb'] = model.model.visual.positional_embedding.shape[1]
            self.params['out_dim'] = model.model.visual.proj.shape[1]
            self.params['patch_size'] = model.model.visual.conv1.weight.shape[2]

            if self.params['model_type'] == 'ViT-B-16':
                self.params['seq_len'] = 197
            elif self.params['model_type'] == 'ViT-L-14':
                self.params['seq_len'] = 256

            print(
                f"Model hyper-parameters: num_layers {self.params['num_layers']} | num_heads {self.params['num_heads']} "
                f"| mlp_activation {self.params['mlp_activation']} | emb {self.params['emb']} "
                f"| out_dim {self.params['out_dim']} | patch_size {self.params['patch_size']} | seq_len {self.params['seq_len']}")



    # Load the args of the finetuned models
    def load_args(self):
        # Load the arguments from the file
        path_args = os.path.join(self.params['path_for_models'], 'args.json')
        if self.params['model_type'] != 'bert' and os.path.isfile(path_args):
            args = argparse.ArgumentParser()
            with open(path_args, "r") as f:
                args.__dict__ = json.load(f)
        else:
            args = parse_arguments()
            args.data_location = '../data'
            args.model = self.params['model_type']
            #args.exp_name =
            args.save = self.params['path_for_models']
            args.eval_datasets = self.params['datasets_to_eval']
            #args.use_same_pretrained = use_same_pretrained
            args.devices = list(range(torch.cuda.device_count()))

        if self.params['batch_size'] is not None:
            args.batch_size = self.params['batch_size']

        self.loaded_args = args
        if self.loaded_args.model != self.params['model_type']:
            raise Exception("The loaded model type is not the same as the model type of the merge class.")


    # Merge the models
    @abstractmethod
    def merge(self):
        pass


    def get_merged_model(self):
        return self.merged_model


    # Loading a specific layer from all the models need to be merged
    def load_models_layers(self, layer_name, layer_num=None) -> List[torch.Tensor]:

        models_use = ['prev_merged_model', self.params['models_to_merge'][-1]] if \
            (self.params['learn_tasks_sequentially'] and self.curr_task_sequence_iter > 0) else self.params['models_to_merge']
        models_indexes_use = [None, self.params['models_indexes'][-1]] if \
            (self.params['learn_tasks_sequentially'] and self.curr_task_sequence_iter > 0) else self.params['models_indexes']

        layer_list = []
        for model_index, model_name in zip(models_indexes_use, models_use):
            if self.params['model_type'] == 'bert':
                layer_path = os.path.join(self.params['path_for_models'], model_name,
                                          f"seed_{model_index}", "layers",
                                          f"bert_{layer_name}.pt")
                if layer_num is not None:
                    layer_path = layer_path.format(layer_num)

                layer_params = torch.load(layer_path)

            else:
                if model_name == 'prev_merged_model': # Load the layer from the previous merged model
                    layer_path = os.path.join(self.params['path_to_save'], 'layers', 'merged_model', layer_name)
                else:
                    layer_path = os.path.join(self.params['path_for_layers'], model_name, layer_name)

                if layer_num is not None:
                    layer_path = layer_path + '_' + str(layer_num)

                with open(layer_path, 'rb') as f:
                    layer_params = pickle.load(f)

            layer_list.append(layer_params)

        return layer_list


    def load_layer(self, model_name, layer_name):
        layer_path = os.path.join(self.params['path_for_layers'], model_name, layer_name)
        with open(layer_path, 'rb') as f:
            layer_params = pickle.load(f)

        return layer_params


    # Create a state dict of layer norm average from all the models
    def get_average_layer_norm(self, layer_name, first=False):
        new_LN_state_dict = {}
        if self.params['model_type'] == 'bert':
            LN_weight = self.load_models_layers(layer_name=layer_name + '_weight', layer_num=None)
            LN_bias = self.load_models_layers(layer_name=layer_name + '_bias', layer_num=None)
            LN_state_dicts = [{'weight': weight, 'bias': bias} for weight, bias in zip(LN_weight, LN_bias)]
        else:
            LN_state_dicts = self.load_models_layers(layer_name=layer_name, layer_num=None)

        if first:
            for key in LN_state_dicts[0].keys():
                new_LN_state_dict[key] = LN_state_dicts[0][key]

        else:
            for key in LN_state_dicts[0].keys():
                params = 0
                for i in range(len(LN_state_dicts)):
                    params += LN_state_dicts[i][key]
                new_LN_state_dict[key] = params / len(self.params['models_to_merge'])

        return new_LN_state_dict

    def eval_model_on_datasets(self, model, U_output=None, used_train_features=False, multi_head_eval=False):
        if self.params['model_type'] == 'clip':
            data_type_list = ['val']
            metrics = {'val': None, 'test': None}
            for data_type in data_type_list:
                metrics[data_type] = eval_clip_on_coco(model=model,
                                                       train_preprocess=self.clip_train_preprocess,
                                                       transform=self.clip_transform,
                                                       data_type=data_type)

            self.mean_val_acc = metrics['val']
            self.mean_test_acc = metrics['test']
            self.k_vals = metrics[data_type_list[0]]['k_vals']

            eval_text = ""
            for data_type in data_type_list:
                eval_text += f"\n{data_type}, Text-to-image Recall@K"
                for k, x in zip(self.k_vals, metrics[data_type]['text-to-image']):
                    eval_text += f" R@{k}: {100 * x:.2f}%"

                eval_text += f"\n{data_type}, Image-to-text Recall@K"
                for k, x in zip(self.k_vals, metrics[data_type]['image-to-text']):
                    eval_text += f" R@{k}: {100 * x:.2f}%"

            return eval_text

        else:
            self.loaded_args.eval_datasets = self.params['datasets_to_eval']
            self.loaded_args.save_dir = self.params['path_to_save']
            self.loaded_args.devices = list(range(torch.cuda.device_count()))

            eval_text = self.eval_transformer_on_datasets(model=model, U_output=U_output, multi_head_eval=multi_head_eval,
                                                     used_train_features=used_train_features)

            if self.params['train_classification_head']:
                if 'ViT' not in self.params['model_type']:
                    raise Exception("Training classification head is only supported for ViT models.")
                train_classification_head(self, model, U_output=U_output)
                for i in range(3):
                    eval_text += self.eval_transformer_on_datasets(model=model, U_output=U_output,
                                                                  multi_head_eval=multi_head_eval,
                                                                  used_train_features=used_train_features,
                                                                  used_trained_head=True,
                                                                  head_num=i)

            return eval_text

    # Evaluate the merged model on the vision_datasets
    # U_output: In out merge there is a U_output matrix in shape [out_dim, num_models * out_dim],
    # which reconstruct the original output of the models from the merged output.
    # used_train_features: If True, the train_features dataset will be used for evaluation
    # used_trained_head: If True, the trained head will be used for evaluation
    def eval_transformer_on_datasets(self, model, U_output = None, used_train_features=False,
                                     multi_head_eval=False, used_trained_head=False, head_num=None):
        print("\nEvaluating the merged model.")
        if self.params['model_type'] == 'bert':
            model_dirs = [os.path.join(self.params['path_for_models'], model_name, f"seed_{models_index}") for model_name, models_index in
                          zip(self.params['models_to_merge'], self.params['models_indexes'])]
            _, metric_dict = evaluate_text_datasets(model,
                                                    eval_datasets=self.params['datasets_to_eval'],
                                                    model_dirs=model_dirs,
                                                    U_output=U_output)

            if multi_head_eval:
                multi_head_eval = False
                raise Warning("Multi-head evaluation is not supported for text models.")

        else:
            # if self.params.get('merge_type', None) == 'ours_block':
            #    self.loaded_args.batch_size = self.params['batch_size']
            if used_trained_head:
                head_path = os.path.join(self.params['path_to_save'], 'heads')
            else:
                head_path = os.path.join(self.params['path_for_models'], 'heads')

            _, metric_dict = evaluate(model, self.loaded_args, U_output=U_output, head_path=head_path, head_num=head_num)

            # Evaluate the model on the train_features
            if used_train_features:
                for data_num, dataset_name in enumerate(self.params['datasets_to_eval']):

                    # Build the train_features dataset
                    train_features_dataset = self.build_train_features_dataset(dataset_name=dataset_name)

                    if U_output is not None:
                        # U_output is a matrix of shape [out_dim, num_models * out_dim]
                        # Need to take the part of U_output that corresponds to the current dataset
                        out_dim = U_output.shape[0]
                        U_curr = U_output[:, data_num * out_dim: (data_num + 1) * out_dim]
                    else:
                        U_curr = None

                    results = eval_single_dataset(model, dataset_name, args=self.loaded_args, data_type='train', U_output=U_curr,
                                                  dataset=train_features_dataset, with_prints=False, head_path=head_path,
                                                  head_num=head_num)

                    metric_dict['{}_train_features_acc'.format(dataset_name)] = results['top1-train']

        # Save the results
        self.mean_train_acc.append(
            np.mean([metric_dict[key] for key in metric_dict.keys() if ('train_acc' in key and 'acc' in key)]))
        self.mean_train_features_acc.append(
            np.mean([metric_dict[key] for key in metric_dict.keys() if ('train_features' in key and 'acc' in key)]))
        self.mean_val_acc.append(
            np.mean([metric_dict[key] for key in metric_dict.keys() if ('val' in key and 'acc' in key)]))
        self.mean_test_acc.append(
            np.mean([metric_dict[key] for key in metric_dict.keys() if ('test' in key and 'acc' in key)]))

        # Multi-head evaluation
        if multi_head_eval:
            multi_head_metric_dict = multi_head_evaluate(model, self.loaded_args, U_output=U_output,
                                                         head_path=head_path, head_num=head_num)
            self.mean_val_acc_multi = np.mean([multi_head_metric_dict[key] for key in multi_head_metric_dict.keys() if ('val' in key and 'acc' in key)])
            self.mean_test_acc_multi = np.mean([multi_head_metric_dict[key] for key in multi_head_metric_dict.keys() if ('test' in key and 'acc' in key)])
            metric_dict.update(multi_head_metric_dict)

        if self.metric_dicts is None:
            self.metric_dicts = {}

        for key in metric_dict.keys():
            if key not in self.metric_dicts:
                self.metric_dicts[key] = []
            self.metric_dicts[key].append(metric_dict[key])

        eval_text = "\n\nEvaluating merged model:{}".format(str(metric_dict))

        if multi_head_eval:
            eval_text += "\nTrain mean: {} |  Train features mean: {} | Val mean: {} | Val joint mean: {} | Test per-task mean: {} | Test joint mean: {}".\
                format(self.mean_train_acc[-1], self.mean_train_features_acc[-1], self.mean_val_acc[-1],
                       self.mean_val_acc_multi, self.mean_test_acc[-1], self.mean_test_acc_multi)
        else:
            eval_text += "\nTrain mean: {} |  Train features mean: {} | Val mean: {} | Test mean: {}". \
                format(self.mean_train_acc[-1], self.mean_train_features_acc[-1], self.mean_val_acc[-1],
                       self.mean_test_acc[-1])

        eval_text += "\n\nAccuracies: {}".format(str(self.filter_and_round_metrics(metric_dict)))
        print(eval_text)

        return eval_text


    # Create the dataset of the train features
    def build_train_features_dataset(self, dataset_name):
        features_path = os.path.join(self.params['path_for_models'], 'features_{}'.format(self.params['num_features_train']))
        input_train_path = os.path.join(features_path, "dataset_{}".format(dataset_name), "input_train_{}".format(dataset_name))
        labels_train_path = os.path.join(features_path, "dataset_{}".format(dataset_name), "labels_train_{}".format(dataset_name))

        input_train = self.load_feature_dataset(dataset_path=input_train_path, dataset_size=self.params['num_features_train'])
        labels_train = self.load_feature_dataset(dataset_path=labels_train_path, dataset_size=self.params['num_features_train'])

        image_dataset = ImagesDataset(input_train, labels_train)
        dataset = FeaturesDatasetHolder(train_dataset=image_dataset, val_dataset=None, test_dataset=None,
                                        batch_size=1152, num_workers=16, train_shuffle=False)

        return dataset


    def create_features_dataset_for_merger(self, input_train, input_val, input_early_stopping,
                                           ids_train, ids_val, ids_early_stopping,
                                           target_train_list, target_val_list, target_early_stopping_list,
                                           inner_target_train_dict, inner_target_val_dict,
                                           attention_mask_train, attention_mask_val, attention_mask_early_stopping,
                                           token_type_train, token_type_val, token_type_early_stopping,
                                           normalize_features, train_shuffle, batch_size):

        # Parameters for normalizing the features
        if normalize_features:
            normalize_scale = self.params.get('normalize_scale', 0.0)
            print("Normalizing features with normalize_scale={}".format(normalize_scale))
        else:
            normalize_scale = 0.0  # No normalization

        early_stopping_dataset = None

        if self.params['model_type'] != 'bert':
            if self.params['loss_type'] in ['rec_with_inner_att', 'rec_with_inner_mlp', 'rec_with_inner_att_ids', 'rec_with_inner_mlp_ids']:
                # The training will require the inputs for the merger and the original models outputs.
                # It will also require the inner targets (features) of the models
                train_dataset = DatasetInnerFeaturesVIT(inputs=input_train, targets_list=target_train_list,
                                                        inner_target_dict=inner_target_train_dict, ids=ids_train,
                                                        inner_target_scales=None,
                                                        scale_inner_type=self.params['scale_inner_type'],
                                                        inner_target_layers=self.params['loss_layer_num'],
                                                        loss_type=self.params['loss_type'],
                                                        normalize_scale=normalize_scale,
                                                        target_dim_cat=0)

                train_scales = train_dataset.scales if normalize_features else None

                val_dataset = DatasetInnerFeaturesVIT(inputs=input_val, targets_list=target_val_list,
                                                      inner_target_dict=inner_target_val_dict, ids=ids_val,
                                                      inner_target_scales=train_dataset.inner_target_scales,
                                                      scale_inner_type=self.params['scale_inner_type'],
                                                      inner_target_layers=self.params['loss_layer_num'],
                                                      loss_type=self.params['loss_type'],
                                                      normalize_scale=normalize_scale,
                                                      target_dim_cat=0,
                                                      scales=train_scales,)  # Use the same scales as the train dataset

            else:
                # The training will require only the inputs for the merger and the original models outputs.
                train_dataset = DatasetFeaturesVIT(inputs=input_train, targets_list=target_train_list,
                                                   ids=ids_train,
                                                   normalize_scale=normalize_scale,
                                                   target_dim_cat=0)

                train_scales = train_dataset.scales if normalize_features else None

                val_dataset = DatasetFeaturesVIT(inputs=input_val, targets_list=target_val_list,
                                                 ids=ids_val,
                                                 normalize_scale=normalize_scale,
                                                 target_dim_cat=0,
                                                 scales=train_scales,)  # Use the same scales as the train dataset

        else:
            # BERT models
            train_dataset = DatasetFeaturesBERT(inputs=input_train, attention_mask=attention_mask_train,
                                                token_type_ids=token_type_train,
                                                targets_list=target_train_list, ids=ids_train,
                                                normalize_scale=normalize_scale, target_dim_cat=0)

            train_scales = train_dataset.scales if normalize_features else None

            val_dataset = DatasetFeaturesBERT(inputs=input_val, attention_mask=attention_mask_val,
                                                token_type_ids=token_type_val,
                                                targets_list=target_val_list, ids=ids_val,
                                                normalize_scale=normalize_scale, target_dim_cat=0,
                                                scales=train_scales,)

            if self.params['with_early_stopping']:
                early_stopping_dataset = DatasetFeaturesBERT(inputs=input_early_stopping, attention_mask=attention_mask_early_stopping,
                                                             token_type_ids=token_type_early_stopping,
                                                             targets_list=target_early_stopping_list, ids=ids_early_stopping,
                                                             normalize_scale=normalize_scale, target_dim_cat=0,
                                                             scales=train_scales,)

        batch_size = batch_size if batch_size is not None else self.params['batch_size']
        return FeaturesDatasetHolder(train_dataset=train_dataset,
                                     val_dataset=val_dataset,
                                     early_stopping_dataset=early_stopping_dataset,
                                     #test_dataset=test_dataset,
                                     batch_size=batch_size,
                                     num_workers=16,
                                     train_shuffle=train_shuffle)

    def create_features_dataset_clip(self):
        features_path = os.path.join(self.params['path_for_models'],
                                     'features_{}'.format(self.params['num_features_train']))

        _datasets = {}
        for data_type in ['train', 'val']:
            dataset_size = self.params['num_features_train'] if data_type == 'train' else self.params['num_features_test']

            images = self.load_feature_dataset(dataset_path=os.path.join(features_path, f"images_{data_type}"), dataset_size=dataset_size)
            texts = self.load_feature_dataset(dataset_path=os.path.join(features_path, f"text_{data_type}"), dataset_size=dataset_size)
            features_images = self.load_feature_dataset(dataset_path=os.path.join(features_path, f"image_features_{data_type}"), dataset_size=dataset_size)
            features_texts = self.load_feature_dataset(dataset_path=os.path.join(features_path, f"text_features_{data_type}"), dataset_size=dataset_size)
            _datasets[data_type] = DatasetFeaturesImageText(images=images, texts=texts,  features_images=features_images, features_texts=features_texts)

        return FeaturesDatasetHolder(train_dataset=_datasets['train'],
                                     val_dataset=_datasets['val'],
                                     # test_dataset=test_dataset,
                                     batch_size=self.params['batch_size'],
                                     num_workers=16,
                                     train_shuffle=True)


    # Create a new dictionary with filtered keys and rounded values
    def filter_and_round_metrics(self, metric_dict):

        filtered_metrics = {key: round(value, 2) for key, value in metric_dict.items() if "acc" in key and 'train_acc' in key}
        filtered_metrics.update({key: round(value, 2) for key, value in metric_dict.items() if "acc" in key and 'train_features' in key})
        filtered_metrics.update({key: round(value, 2) for key, value in metric_dict.items() if "acc" in key and 'val' in key})
        filtered_metrics.update({key: round(value, 2) for key, value in metric_dict.items() if "acc" in key and 'test' in key})
        return filtered_metrics


    def save_experiment_info(self, text):
        with open(self.params['path_to_save_params'], "wb") as f:  # saving the dataset
            pickle.dump(self.params, f, pickle.HIGHEST_PROTOCOL)

        desc = self.__str__() + '\n\n' + text
        with open(self.params['path_to_save_desc'], 'a+') as f:
            f.write(desc)


    def __str__(self):
        text = "Merge type: " + self.params['merge_type'] + "\n" + self.params['descriptor'] + "\n"
        for i, key in enumerate(self.params.keys()):
            if key not in ['descriptor', 'path_for_layers', 'path_for_zs_layers',
                           'path_to_save', 'path_to_save_desc', 'path_to_save_params', 'features_path',
                           'plots_path']:
                if (i + 1) % 4 == 0:
                    text += "\n"
                text += "{} : {}, ".format(key, self.params[key])

        return text