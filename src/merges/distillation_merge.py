import os
import copy
from pathlib import Path

from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import open_clip

from train_eval.trainer import Trainer
from modeling import ImageEncoder, ImageClassifier, ModuleWrapper, ClassificationHead, CLIPSharedTransformer
from visualization import plot_dict, merge_training_plots, mul_training_plots
from losses.loss_functions import create_loss_function_for_dist

from merges.general_merge import GeneralMerge
from merges.slerp_merge import flatten_weights, assign_weights
from merges.regmean_merge import RegMeanMerge


class DistillationMerge(GeneralMerge):
    """
    Merging models by distillation.
    model_type: The type of the model to merge.
    transformer_type: The type of the transformer to use, needed only for CLIP models.
    pre_trained: The pre-trained model to use, needed only for CLIP models.
    experiment_name: The name of the current merge experiment.
    experiment_dir: The directory to save the experiment.
    path_for_models: The path for the models to merge.
    models_to_merge: The names of the models to merge.
    models_indexes: The indexes of the models to merge. Needed only in BERT case.
    batch_size: The batch size for the training of the merge layer.
    datasets_to_eval: The vision_datasets to evaluate the merged model on.
    separate_datasets: Each models features where created only from the original model train set.
    num_features_train: The number of features from each dataset, to use for training the merge layer.
    num_features_test: The number of features from each dataset, to use for testing the merge layer.
    num_features_aug_train: The number of features from each dataset, to use for training the merge layer, after augmentation.
    distributed: If to train the model in a distributed manner.
    epochs: The number of epochs to train the merge layer.
    batch_size: The batch size for the training of the merge layer.
    lr: The learning rate for the training of the merge layer.
    wd: The weight decay for the training of the merge layer.
    scheduler_type: The type of the scheduler for the training of the merge layer.
    warmup_length: The warmup length for the training of the merge layer.
    StepLR_step_size: The step size for the StepLR scheduler.
    StepLR_gamma: The gamma for the StepLR scheduler.
    init: The initialization method for the merged model.
    loss_type: The loss for training the model. "rec" means to reconstruct the features of the original models.
    'rec_with_inner' means to reconstruct the features of the original models, including inner features.
    loss_layer_num: if loss_type == 'rec_with_inner', this list contains the layers to use for the loss.
    reg_coeff: The coefficient for the regularization of the inner features.
    clip_grad_norm: Whether to clip the gradient norm in the merge layer.
    coefficients: The coefficients for the models to merge, in case we will use average merge init.
    If none, then will use simple average.
    normalize_scale: Normalizing the scale of the features. If 0 then don't normalize. If -1 to normalize them to have
    the same scale as the first features. For any other number, normalize them to this number.
    train_classification_head: Whether to train a classification head on the merged model.
    """
    def __init__(self,
                 model_type: str,
                 experiment_name: str,
                 experiment_dir: str,
                 path_for_models: str,
                 models_to_merge: List[str],
                 epochs: int,
                 batch_size: int,
                 lr: float,
                 wd: float,
                 num_features_train: int,
                 num_features_test: int = 300,
                 num_features_aug_train : int = 0,
                 transformer_type: str = None,
                 pre_trained: str = None,
                 scheduler_type: str | None = 'warmup',
                 distributed: str | None = 'data_parallel',
                 clip_grad_norm: bool = False,
                 with_early_stopping: bool = False,
                 datasets_to_eval: List[str] | None = None,
                 models_indexes: List[int] | None = None,
                 separate_datasets: bool = True,
                 init: str = None,
                 coefficients: Dict[str, float] | None = None,
                 normalize_scale: float = 0.5,
                 loss_type: str = 'rec',
                 loss_layer_num: List[int] | None = None,
                 reg_coeff: float = 0.0,
                 StepLR_step_size: int = 50,
                 StepLR_gamma: float = 0.2,
                 warmup_length: int = 168,
                 train_classification_head: bool = False,
                 descriptor: str = None):

        super(DistillationMerge, self).__init__(model_type=model_type,
                                                transformer_type=transformer_type,
                                                pre_trained=pre_trained,
                                                experiment_name=experiment_name,
                                                experiment_dir=experiment_dir,
                                                path_for_models=path_for_models,
                                                models_to_merge=models_to_merge,
                                                models_indexes=models_indexes,
                                                datasets_to_eval=datasets_to_eval,
                                                train_classification_head=train_classification_head,
                                                descriptor=descriptor)

        # Update the params

        # data for training
        self.params['num_features_train'] = num_features_train
        self.params['num_features_test'] = num_features_test
        self.params['num_features_aug_train'] = num_features_aug_train
        self.params['normalize_scale'] = normalize_scale
        self.params['separate_datasets'] = separate_datasets

        # training params
        self.params['distributed'] = distributed
        self.params['epochs'] = epochs
        self.params['batch_size'] = batch_size
        self.params['lr'] = lr
        self.params['wd'] = wd
        self.params['clip_grad_norm'] = clip_grad_norm
        self.params['with_early_stopping'] = with_early_stopping
        self.params['scheduler_type'] = scheduler_type
        self.params['warmup_length'] = warmup_length
        self.params['StepLR_step_size'] = StepLR_step_size
        self.params['StepLR_gamma'] = StepLR_gamma

        # Loss params
        self.params['loss_type'] = loss_type
        self.params['loss_layer_num'] = loss_layer_num
        self.params['reg_coeff'] = reg_coeff
        self.params['scale_inner_type'] = 'ones'  # The type of the scale for the inner features

        # General
        self.params['init'] = init.lower()
        self.params['merge_type'] = 'distillation'

        self.create_merge_dir()
        if coefficients is None:
            self.coefficients = {model_name: 1 / len(models_to_merge) for model_name in models_to_merge}
        else:
            assert len(coefficients) == len(models_to_merge)
            self.coefficients = coefficients

        # Sanity checks
        if self.params['loss_type'] in ['rec_with_inner_att', 'rec_with_inner_mlp', 'rec_with_inner_att_ids', 'rec_with_inner_mlp_ids'] \
            and self.params['loss_layer_num'] == None:
            raise Exception("loss_layer_num must be specified for loss_type = rec_with_inner")

        if self.params['model_type'] == 'bert' and \
                self.params['loss_type'] in ['rec_with_inner_att', 'rec_with_inner_mlp', 'rec_with_inner_att_ids', 'rec_with_inner_mlp_ids']:
            raise Exception("The loss type {} is not supported for BERT models.".format(self.params['loss_type']))


        # Update the args for the training of the MU matrices
        self.args_for_MU_training = copy.deepcopy(self.loaded_args)
        self.args_for_MU_training.distributed = distributed
        self.args_for_MU_training.scheduler_type = scheduler_type
        self.args_for_MU_training.warmup_length = warmup_length
        self.args_for_MU_training.StepLR_step_size = StepLR_step_size
        self.args_for_MU_training.StepLR_gamma = StepLR_gamma
        self.args_for_MU_training.batch_size = batch_size
        self.args_for_MU_training.eval_datasets = datasets_to_eval
        self.args_for_MU_training.wd = wd
        self.args_for_MU_training.lr = lr
        self.args_for_MU_training.lr_diag = None
        self.args_for_MU_training.gamma = 0.5
        self.args_for_MU_training.devices = list(range(torch.cuda.device_count()))

        self.losses_lists = {}

        # Create the loss function
        self.loss_func = create_loss_function_for_dist(loss_type=self.params['loss_type'],
                                                       loss_layer_num=self.params['loss_layer_num'],
                                                       reg_coeff=self.params['reg_coeff'])


    # Merge the models, evaluate, and save the experiment
    def merge(self,
              with_eval: bool = True,
              with_multi_head_eval: bool = False,
              with_save: bool = False):

        U_output = self.merge_models()
        dict_loss = self.losses_lists.get('val_epoch', None)
        if dict_loss is not None:
            self.params['last_val_loss'] = dict_loss['full-loss'][-1]
        if with_eval:
            eval_text = self.eval_model_on_datasets(self.merged_model, U_output,
                                                    used_train_features=True, multi_head_eval=with_multi_head_eval)
            self.save_experiment_info(eval_text)

        if with_save:
            self.save_merged_model()


    def average_merge(self):
        self.params['model_norms'] = {}
        weight_vec = 0
        with torch.no_grad():
            # Loading the models and flatting their weights
            for i, model_name in enumerate(self.params['models_to_merge']):
                model = self.load_model(model_name=model_name)
                curr_vec = flatten_weights(model=model)
                curr_vec_norm = np.linalg.norm(curr_vec)
                self.params['model_norms'][model_name] = curr_vec_norm
                print("norm of the curr model vec: ", curr_vec_norm)
                weight_vec = weight_vec + curr_vec * self.coefficients[model_name]
                del model

            merged_model_norm = np.linalg.norm(weight_vec)
            self.params['model_norms']['merged'] = merged_model_norm
            print("norm of the new model vec: ", merged_model_norm)

            # Assigning the weights to the merged model
            model = ImageEncoder(self.loaded_args, keep_lang=False, pretrained='openai')
            model = assign_weights(model=model, weights=weight_vec)
            return model

    def regmean_merge(self):
        regmean_merge = RegMeanMerge(
            model_type=self.params['model_type'],
            experiment_name=self.params['experiment_name'] + '_for_init',
            experiment_dir=self.params['experiment_dir'],
            path_for_models=self.params['path_for_models'],
            models_to_merge=self.params['models_to_merge'],
            datasets_to_eval=self.params['datasets_to_eval'],
            num_features_train=self.params['num_features_train'],
            num_features_aug_train=240,
            reg_coef=0.7,
            descriptor='using the regmean method for initialization',
            init='average', )

        regmean_merge.merge(with_eval=False, with_save=False, with_multi_head_eval=False)
        return regmean_merge.merged_model


    def model_init(self):
        if self.params['model_type'] == 'clip':
            clip, self.clip_train_preprocess, self.clip_transform = open_clip.create_model_and_transforms(self.params['transformer_type'],
                                                               pretrained=self.params['pre_trained'])

            # Initialize from the visual transformer
            conv_weights = self.load_layer(layer_name="conv", model_name='visual_encoder')
            class_embeddings = self.load_layer(layer_name="class_embedding", model_name='visual_encoder')
            pos_embedding = self.load_layer(layer_name="pos_embedding", model_name='visual_encoder')
            ln_pre = self.load_layer(layer_name="ln_pre_state_dict", model_name='visual_encoder')
            ln_post = self.load_layer(layer_name="ln_post_state_dict", model_name='visual_encoder')
            proj = self.load_layer(layer_name="out_proj", model_name='visual_encoder')
            visual_weights = {'conv': conv_weights, 'class_embedding': class_embeddings, 'pos_embedding': pos_embedding,
                              'ln_pre': ln_pre, 'ln_post': ln_post, 'proj': proj}

            token_embedding = self.load_layer(layer_name="token_embedding", model_name='text_encoder')
            pos_embedding = self.load_layer(layer_name="pos_embedding", model_name='text_encoder')
            ln_final = self.load_layer(layer_name="ln_final_state_dict", model_name='text_encoder')
            text_projection = self.load_layer(layer_name="text_projection", model_name='text_encoder')
            text_weights = {'token_embedding': token_embedding, 'pos_embedding': pos_embedding,
                            'ln_final': ln_final, 'text_projection': text_projection}

            clip_shared = CLIPSharedTransformer(transformer=clip.visual.transformer,
                                                visual_weights=visual_weights,
                                                text_weights=text_weights)

            clip_shared.freeze_non_transformer()
            return clip_shared


        if self.params['model_type'] == 'bert' and self.params['init'] in ['average', 'random', 'pre-trained', 'pretrained', 'pre_trained']:
            raise Exception("Only 'first' init method is supported for BERT models.")

        # Get the model hyper parameters
        pretrained = 'openai'
        if self.params['model_type'] == 'ViT-L-14':
            pretrained = 'laion400m_e32'

        if self.params['init'] == 'average':
            return self.average_merge()
        elif self.params['init'] == 'random':
            return ImageEncoder(self.loaded_args, keep_lang=False, pretrained=pretrained, random_init=True)
        elif self.params['init'] in ['pre-trained', 'pretrained', 'pre_trained']:
            return ImageEncoder(self.loaded_args, keep_lang=False, pretrained=pretrained)
        elif self.params['init'] == 'first':
            return self.load_model(model_name=self.params['models_to_merge'][0], model_number=0)
        elif self.params['init'] == 'regmean':
            return self.regmean_merge()
        else:
            raise NotImplementedError("The init method {} is not implemented".format(self.params['init']))

    def merge_models(self):
        # Initializing the merged model
        merged_model = self.model_init()
        self.get_model_hyper_parameters(merged_model)

        if self.params['epochs'] == 0:
            self.merged_model = merged_model
            return

        # Creating the dataset
        dataset = self.create_features_dataset()

        # Train
        self.params['num_parameters'] = sum(p.numel() for p in merged_model.parameters() if p.requires_grad)
        if self.params['loss_type'] in ['rec_with_inner_att', 'rec_with_inner_mlp', 'rec_with_inner_att_ids','rec_with_inner_mlp_ids']:
            merged_model = VITWrapper(model=merged_model, loss_layer_num=self.params['loss_layer_num'], loss_type=self.params['loss_type'])

        print("\n--------Starting distillation --------")
        what_is_trained = 'bert_distillation' if self.params['model_type'] == 'bert' else 'VIT_distillation'
        trainer = Trainer(args=self.args_for_MU_training, loss_fn=self.loss_func,
                              clip_grad_norm=self.params['clip_grad_norm'], loss_type=self.params['loss_type'],
                              with_eval=True, eval_type='loss_test', what_is_trained=what_is_trained,
                              models_to_merge=self.params['models_to_merge'],
                              with_early_stopping=self.params['with_early_stopping'],)

        merged_model = trainer.train_model(model=merged_model,
                                           epochs=self.params['epochs'],
                                           dataset=dataset,
                                           with_all_plots=False,
                                           check_loss_no_train=False)

        if self.params['loss_type'] in ['rec_with_inner_att', 'rec_with_inner_mlp', 'rec_with_inner_att_ids','rec_with_inner_mlp_ids']:
            self.merged_model = merged_model.module.model
        else:
            self.merged_model = merged_model.module

        # Saving the training statistics and create plots
        self.save_trainer_statistics(trainer)

        U_output = None if self.params['normalize_scale'] == 0 else self.create_U_output(scales=dataset.scales)
        return U_output

    def create_features_dataset(self):
        if self.params['model_type'] == 'clip':
            return self.create_features_dataset_clip()
        else:
            return self.create_features_dataset_vit()


    def create_features_dataset_vit(self):
        #path = f"{self.params['path_for_models']}/{model_name}/seed_{self.params['models_indexes'][model_number]}"
        features_path = os.path.join(self.params['path_for_models'], 'features_{}'.format(self.params['num_features_train']))

        # 1.1. Load the inputs and ids
        input_train_list, input_val_list, input_early_stopping_list, ids_train_list, ids_val_list, ids_early_stopping_list = [], [], [], [], [], []
        for model_num, model_name in enumerate(self.params['models_to_merge']):
            model_name = model_name.replace('finetuned_', '')

            #input_train_path = os.path.join(features_path, "dataset_{}".format(model_name), "input_train_{}".format(model_name))
            #input_val_path = os.path.join(features_path, "dataset_{}".format(model_name), "input_val_{}".format(model_name))
            input_train_path, input_val_path, input_aug_train_path, input_early_stopping_path = \
                self.build_path_for_specific_features(features_path, model_name, model_num, what_to_load='input')

            input_train_curr = self.load_feature_dataset(dataset_path=input_train_path, dataset_size=self.params['num_features_train'])
            input_val_curr = self.load_feature_dataset(dataset_path=input_val_path, dataset_size=self.params['num_features_test'])

            input_train_list.append(input_train_curr)
            input_val_list.append(input_val_curr)

            ids_train_list.append(torch.full((self.params['num_features_train'],), model_num, dtype=torch.int64))
            ids_val_list.append(torch.full((self.params['num_features_test'],), model_num, dtype=torch.int64))

            if not self.params['separate_datasets']:
                raise NotImplementedError("The separate_datasets=False is not implemented yet")

            # 1.2. Load the augmented inputs
            if self.params['num_features_aug_train'] > 0 and self.params['model_type'] != 'bert':
                #input_aug_train_path = os.path.join(features_path, "dataset_{}".format(model_name), "augmented_train_input_{}".format(model_name))
                input_aug_train_curr = self.load_feature_dataset(dataset_path=input_aug_train_path, dataset_size=self.params['num_features_aug_train'])
                input_train_list.append(input_aug_train_curr)
                ids_train_list.append(torch.full((self.params['num_features_aug_train'],), model_num, dtype=torch.int64))

            # 1.3. Load the early_stopping inputs
            if self.params['with_early_stopping'] and self.params['model_type'] == 'bert':
                input_early_stopping_curr = self.load_feature_dataset(dataset_path=input_early_stopping_path, dataset_size=200)
                input_early_stopping_list.append(input_early_stopping_curr)
                ids_early_stopping_list.append(torch.full((200,), model_num, dtype=torch.int64))

        input_train = torch.cat(input_train_list, dim=0)
        input_val = torch.cat(input_val_list, dim=0)
        ids_train = torch.cat(ids_train_list, dim=0)
        ids_val = torch.cat(ids_val_list, dim=0)
        input_early_stopping = None
        ids_early_stopping = None
        if self.params['with_early_stopping'] and self.params['model_type'] == 'bert':
            input_early_stopping = torch.cat(input_early_stopping_list, dim=0)
            ids_early_stopping = torch.cat(ids_early_stopping_list, dim=0)

        # 2.1. Load the targets from each original model
        target_train_list, target_val_list, target_early_stopping_list = [], [], []
        for model_num, model_name in enumerate(self.params['models_to_merge']):
            model_name = model_name.replace('finetuned_', '')

            #target_path = os.path.join(features_path, "dataset_{}".format(model_name), "model_finetuned_{}".format(model_name), "output_{}")
            target_train_path, target_val_path, target_aug_train_curr, target_early_stopping_path = \
                self.build_path_for_specific_features(features_path, model_name, model_num, what_to_load='output')

            target_train = self.load_feature_dataset(dataset_path=target_train_path, dataset_size=self.params['num_features_train'])
            target_val = self.load_feature_dataset(dataset_path=target_val_path, dataset_size=self.params['num_features_test'])

            # 2.2 Load the augmented targets
            if self.params['num_features_aug_train'] > 0 and self.params['model_type'] != 'bert':
                target_aug_train_curr = self.load_feature_dataset(dataset_path=target_aug_train_curr,
                                                                 dataset_size=self.params['num_features_aug_train'])
                target_train = torch.cat([target_train, target_aug_train_curr], dim=0)

            # 2.3. Load the early_stopping inputs
            if self.params['with_early_stopping'] and self.params['model_type'] == 'bert':
                target_early_stopping_curr = self.load_feature_dataset(dataset_path=target_early_stopping_path, dataset_size=200)
                target_early_stopping_list.append(target_early_stopping_curr)

            target_train_list.append(target_train)
            target_val_list.append(target_val)

        # 3. Load inner features for training. This isn't implemented for BERT models.
        # The keys are the layer numbers, the values are lists of tensors,
        # each tensor is the inner features of a model, with shape (N, seq_len, emb_dim)
        inner_target_train_dict, inner_target_val_dict = None, None
        if self.params['loss_type'] in ['rec_with_inner_att', 'rec_with_inner_mlp', 'rec_with_inner_att_ids','rec_with_inner_mlp_ids']:

            inner_target_train_dict, inner_target_val_dict = {}, {}
            for layer_num in self.params['loss_layer_num']:  # The layer number of the inner target
                inner_target_train_dict[layer_num] = []
                inner_target_val_dict[layer_num] = []
                layer_name = 'attn-{}'.format(layer_num) if 'att' in self.params['loss_type'] else 'fc2-{}'.format(layer_num)

                for model_name in self.params['models_to_merge']:  # forwards on the features created by each finetuned model
                    model_name = model_name.replace('finetuned_', '')

                    inner_train_curr, inner_val_curr = \
                        self.load_relevant_features(features_path=features_path, model_name=model_name, layer_name=layer_name)

                    inner_target_train_dict[layer_num].append(inner_train_curr)
                    inner_target_val_dict[layer_num].append(inner_val_curr)

        # 4. Load attention_mask and token_type_ids for BERT models
        attention_mask_train, attention_mask_val, attention_mask_early_stopping, token_type_train, token_type_val,\
            token_type_early_stopping = None, None, None, None, None, None
        if self.params['model_type'] == 'bert':
            attention_mask_train_list, attention_mask_val_list, attention_mask_early_stopping_list, \
                token_type_train_list, token_type_val_list, token_type_early_stopping_list = [], [], [], [], [], []
            for model_num, model_name in enumerate(self.params['models_to_merge']):
                model_name = model_name.replace('finetuned_', '')

                mask_train_path, mask_val_path, _, mask_early_stopping_path = \
                    self.build_path_for_specific_features(features_path, model_name, model_num, what_to_load='attention_mask')

                token_type_train_path, token_type_val_path, _, token_type_early_stopping_path = \
                    self.build_path_for_specific_features(features_path, model_name, model_num, what_to_load='token_type_ids')

                mask_train_curr = self.load_feature_dataset(dataset_path=mask_train_path, dataset_size=self.params['num_features_train'])
                mask_val_curr = self.load_feature_dataset(dataset_path=mask_val_path, dataset_size=self.params['num_features_test'])
                token_type_train_curr = self.load_feature_dataset(dataset_path=token_type_train_path, dataset_size=self.params['num_features_train'])
                token_type_val_curr = self.load_feature_dataset(dataset_path=token_type_val_path, dataset_size=self.params['num_features_test'])

                if self.params['with_early_stopping']:
                    mask_early_stopping_curr = self.load_feature_dataset(dataset_path=mask_early_stopping_path, dataset_size=200)
                    token_type_early_stopping_curr = self.load_feature_dataset(dataset_path=token_type_early_stopping_path, dataset_size=200)
                    attention_mask_early_stopping_list.append(mask_early_stopping_curr)
                    token_type_early_stopping_list.append(token_type_early_stopping_curr)

                attention_mask_train_list.append(mask_train_curr)
                attention_mask_val_list.append(mask_val_curr)
                token_type_train_list.append(token_type_train_curr)
                token_type_val_list.append(token_type_val_curr)

            attention_mask_train = torch.cat(attention_mask_train_list, dim=0)
            attention_mask_val = torch.cat(attention_mask_val_list, dim=0)
            token_type_train = torch.cat(token_type_train_list, dim=0)
            token_type_val = torch.cat(token_type_val_list, dim=0)
            if self.params['with_early_stopping']:
                attention_mask_early_stopping = torch.cat(attention_mask_early_stopping_list, dim=0)
                token_type_early_stopping = torch.cat(token_type_early_stopping_list, dim=0)

        # 5. Create the dataset
        normalize_features = self.params['normalize_scale'] != 0.0
        dataset = self.create_features_dataset_for_merger\
             (input_train=input_train, input_val=input_val, input_early_stopping=input_early_stopping,
              ids_train=ids_train, ids_val=ids_val, ids_early_stopping=ids_early_stopping,
              target_train_list=target_train_list, target_val_list=target_val_list,
              target_early_stopping_list=target_early_stopping_list,
              inner_target_train_dict=inner_target_train_dict,
              inner_target_val_dict=inner_target_val_dict,
              attention_mask_train=attention_mask_train, attention_mask_val=attention_mask_val,
              attention_mask_early_stopping=attention_mask_early_stopping,
              token_type_train=token_type_train, token_type_val=token_type_val,
              token_type_early_stopping=token_type_early_stopping,
              normalize_features=normalize_features,
              train_shuffle=True, batch_size=self.params['batch_size'])

        return dataset


    # Load the target using the features_path, data_name, model_name and layer_name.
    # Including the train, val and maybe augmented train.
    def load_relevant_features(self, features_path, model_name, layer_name):
        curr_feature_path = os.path.join(features_path, "dataset_{}".format(model_name),
                                         "model_finetuned_{}".format(model_name), "{}".format(layer_name) + '_{}')

        target_train = self.load_feature_dataset(dataset_path=curr_feature_path.format('train'),
                                                 dataset_size=self.params['num_features_train'])
        target_val = self.load_feature_dataset(dataset_path=curr_feature_path.format('val'),
                                               dataset_size=self.params['num_features_test'])

        # Load the augmented targets
        if self.params['num_features_aug_train'] > 0:
            target_aug_train = self.load_feature_dataset(dataset_path=curr_feature_path.format('augmented_train'),
                                                         dataset_size=self.params['num_features_aug_train'])
            target_train = torch.cat([target_train, target_aug_train], axis=0)

        return target_train, target_val


    def save_trainer_statistics(self, trainer):
        self.losses_lists['train'] = trainer.train_loss
        self.losses_lists['train_epoch'] = None
        self.losses_lists['val_epoch'] = trainer.val_loss_each_epoch
        self.losses_lists['end_of_epoch_step_num'] = trainer.end_of_epoch_step_num
        #self.losses_lists['features_scale'] = trainer.features_scale
        self.losses_lists['lr'] = trainer.lr_list
        if self.params['model_type'] != 'clip':
            self.losses_lists['var_val_epoch'] = trainer.val_var_each_epoch
            self.losses_lists['entropy_val_epoch'] = trainer.val_ent_each_epoch


    def training_plots(self):
        # Plot loss
        if self.params['model_type'] == 'clip':
            mul_training_plots(train_loss=self.losses_lists['train'],
                               val_loss_each_epoch=self.losses_lists['val_epoch'],
                               end_of_epoch_step_num=self.losses_lists['end_of_epoch_step_num'],
                               title='Distillation training - loss graph',
                               save_path=os.path.join(self.params['plots_path'], "loss_graph"))
        else:
            merge_training_plots(train_loss=self.losses_lists['train'],
                                 train_loss_each_epoch=self.losses_lists['train_epoch'],
                                 val_loss_each_epoch=self.losses_lists['val_epoch'],
                                 end_of_epoch_step_num=self.losses_lists['end_of_epoch_step_num'],
                                 horizontal_line=None,
                                 #features_scales=self.losses_lists['features_scale'],
                                 title='Distillation training - loss graph',
                                 save_path=os.path.join(self.params['plots_path'], "loss_graph"),
                                 only_full=True)

        # Plot lr
        lr_dict = {'lr' : self.losses_lists['lr']}
        plot_dict(dict=lr_dict, x_list=None,
                  title='Distillation training - Learning Rate',
                  save_path=os.path.join(self.params['plots_path'], "lr_graph"),
                  y_axis='Learning Rate', x_axis='epochs')


    # In the case where we trained features with a normalized scale, we need to create the U_output
    # in order to normalize them back
    def create_U_output(self, scales, device='cuda:0'):
        if self.params['model_type'] == 'clip':
            return None
        U_output = torch.eye(self.params['out_dim']).repeat(1, len(self.params['models_to_merge']))
        with torch.no_grad():
            for i, scale in enumerate(scales):
                U_output[:, i * self.params['out_dim'] : (i + 1) * self.params['out_dim']] /= scale
        return U_output.to(device)



###########################################################################################

class VITWrapper(nn.Module):
    """
    Used when we need the inner features of the VIT model.
    """
    def __init__(self, model, loss_layer_num, loss_type):
        super(VITWrapper, self).__init__()
        self.model = model
        self.train_preprocess = model.train_preprocess
        self.loss_layer_num = loss_layer_num
        self.loss_type = loss_type

    def forward(self, x):
        inner_features_dict = {}

        ##### Pre-processing #####
        x = self.model.model.visual.conv1(x) # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat(
            [self.model.model.visual.class_embedding.to(x.dtype) +
             torch.zeros(x.shape[0], 1, x.shape[-1],
                         dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        x = x + self.model.model.visual.positional_embedding.to(x.dtype)  # seq_len, emb
        x = self.model.model.visual.ln_pre(x) # LayerNorm
        x = x.permute(1, 0, 2)  # [batch, seq_len, emb] -> [seq_len, batch, emb]

        for i, ResidualAttentionBlock in enumerate(self.model.model.visual.transformer.resblocks):
            ##### MultiheadAttention #####
            z = ResidualAttentionBlock.ln_1(x)  # LayerNorm
            z = ResidualAttentionBlock.attn(z, z, z, need_weights=False, attn_mask=None)[0]

            if i in self.loss_layer_num:
                # Adding mlp features
                inner_features_dict[f'inner_{i}_att'] = z.permute(1, 0, 2)  # [batch, seq_len, num_models*emb]

            z = x + z

            ##### MLP #####
            t = ResidualAttentionBlock.ln_2(z)  # LayerNorm
            t = ResidualAttentionBlock.mlp(t)
            x = z + t

        ##### Post-processing #####
        x = x.permute(1, 0, 2)  # [seq_len, batch, emb] -> [batch, seq_len, emb]
        x_cls = self.model.model.visual.ln_post(x[:, 0, :])  # LayerNorm
        if self.model.model.visual.proj is not None:
            x_cls = x_cls @ self.model.model.visual.proj  # linear from width to output_dim

        return x_cls, inner_features_dict