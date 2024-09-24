import os

from typing import List, Dict

import torch
import torch.nn as nn

from modeling import ImageEncoder, ImageClassifier, ModuleWrapper, ClassificationHead


from merges.general_merge import GeneralMerge
from merges.slerp_merge import flatten_weights, assign_weights


class RegMeanMerge(GeneralMerge):
    """
    Merging models with RegMean (https://arxiv.org/abs/2212.09849).
    model_type: The type of the model to merge.
    experiment_name: The name of the current merge experiment.
    experiment_dir: The directory to save the experiment.
    path_for_models: The path for the models to merge.
    models_to_merge: The names of the models to merge.
    datasets_to_eval: The vision_datasets to evaluate the merged model on.
    num_features_train: The number of features from each dataset, to use for the merging.
    num_features_aug_train: The number of augmented features from each dataset, to use for the merging.
    reg_coef: The regularization coefficient for the merging. Used to decrease the non-diagonal items.
    init: The initialization method for the merged model.
    coefficients: The coefficients for averaging the non fully connected weights (biases, positional encoding, LN).
    If none, then will use simple average.
    train_classification_head: Whether to train a classification head on the merged model.
    """
    def __init__(self,
                 model_type: str,
                 experiment_name: str,
                 experiment_dir: str,
                 path_for_models: str,
                 models_to_merge: List[str],
                 num_features_train: int,
                 num_features_aug_train : int = 0,
                 datasets_to_eval: List[str] | None = None,
                 init: str = 'average',
                 reg_coef: float = 0.0,
                 coefficients: Dict[str, float] | None = None,
                 train_classification_head: bool = False,
                 descriptor: str = None):

        super(RegMeanMerge, self).__init__(model_type=model_type,
                                          experiment_name=experiment_name,
                                          experiment_dir=experiment_dir,
                                          path_for_models=path_for_models,
                                          models_to_merge=models_to_merge,
                                          datasets_to_eval=datasets_to_eval,
                                          train_classification_head=train_classification_head,
                                          descriptor=descriptor)

        # Update the params
        self.params['path_for_layers'] = os.path.join(self.params['path_for_models'], 'layers')
        self.params['num_features_train'] = num_features_train
        self.params['num_features_aug_train'] = num_features_aug_train
        self.params['init'] = init.lower()
        self.params['reg_coef'] = reg_coef
        self.params['merge_type'] = 'RegMean'

        self.create_merge_dir()

        if coefficients is None:
            self.coefficients = {model_name: 1 / len(models_to_merge) for model_name in models_to_merge}
        else:
            assert len(coefficients) == len(models_to_merge)
            self.coefficients = coefficients



    def model_init(self):
        if self.params['init'] == 'average':
            return self.average_merge()
        else:
            raise NotImplementedError(f"Init method {self.params['init']} is not implemented yet.")

    def average_merge(self):
        weight_vec = 0
        with torch.no_grad():
            # Loading the models and flatting their weights
            for i, model_name in enumerate(self.params['models_to_merge']):
                model = self.load_model(model_name=model_name)
                curr_vec = flatten_weights(model=model)
                weight_vec = weight_vec + curr_vec * self.coefficients[model_name]
                del model

            # Assigning the weights to the merged model
            pretrained = 'openai'
            if self.params['model_type'] == 'ViT-L-14':
                pretrained = 'laion400m_e32'
            model = ImageEncoder(self.loaded_args, keep_lang=False, pretrained=pretrained)
            model = assign_weights(model=model, weights=weight_vec)
            return model


    def regmean_layer_merge(self, w_list, activation_list, reg_coef, layer_num=None):
        """
        Merging the weights of the layers using the RegMean method.
        :param w_list: The weights to merge. Element from each model. In shape (dim_in, dim_out).
        :param activation_list: The activations of the layers to merge. Element from each model. In shape (N, dim_in)
        :param reg_coef: The regularization coefficient for the merging. Used to decrease the non-diagonal items.
        :return: The merged weights. In shape (dim_in, dim_out).
        """
        assert len(w_list) == len(activation_list)

        g_list = []
        for x in activation_list:
            g = x.T @ x
            g = reg_coef * g + (1 - reg_coef) * torch.diag(torch.diag(g))
            g_list.append(g)

        g_sum_inv = torch.stack(g_list, dim=2).sum(dim=2).inverse()

        weights_g_sum = torch.zeros_like(w_list[0])
        for w, g in zip(w_list, g_list):
            weights_g_sum += g @ w

        merged_w = g_sum_inv @ weights_g_sum
        return merged_w

    # Loading a list of features from a specific layer, element from each model
    # if with_seq is True, it means that those features are in shape (N, seq_len, dim), and we need to flatten them
    def load_list_features(self, layer_name, with_seq=False):
        features_list = []
        features_path = os.path.join(self.params['path_for_models'], 'features_{}'.format(self.params['num_features_train']))
        for model_num, model_name in enumerate(self.params['models_to_merge']):
            model_name = model_name.replace('finetuned_', '')
            feature_path = os.path.join(features_path, "dataset_{}".format(model_name),
                                        "model_finetuned_{}".format(model_name), layer_name)

            features = self.load_feature_dataset(dataset_path=feature_path + '_train', dataset_size=self.params['num_features_train'])

            if self.params['num_features_aug_train'] > 0:
                features_aug = self.load_feature_dataset(dataset_path=feature_path + '_augmented_train',
                                                     dataset_size=self.params['num_features_aug_train'])

                features = torch.cat([features, features_aug], dim=0)

            features_list.append(features)

        if with_seq:
            features_list = [feature.view(feature.shape[0]*feature.shape[1], feature.shape[2]) for feature in features_list]

        return features_list


    # Merge the models, evaluate, and save the experiment
    def merge(self,
              with_eval: bool = True,
              with_multi_head_eval: bool = False,
              with_save: bool = False):

        self.merge_models()
        eval_text = ""
        if with_eval:
            eval_text = self.eval_model_on_datasets(self.merged_model, used_train_features=True,
                                                    multi_head_eval=with_multi_head_eval)
        self.save_experiment_info(eval_text)

        if with_save:
            self.save_merged_model()

    def merge_models(self):
        # Initializing the merged model
        self.merged_model = self.model_init()
        self.get_model_hyper_parameters(self.merged_model)

        is_seq = True

        ##### Pre-processing block #####
        conv_weights = self.load_models_layers(layer_name="conv")
        for i, conv_weight in enumerate(conv_weights):
            conv_weight.requires_grad = False
            conv_weights[i] = conv_weight.view(self.params['emb'], 3 * self.params['patch_size'] ** 2).T

        conv_acts = self.load_list_features('conv_inputs', is_seq) # [N * grid ** 2, emb]

        new_conv_w = self.regmean_layer_merge(w_list=conv_weights,
                                              activation_list=conv_acts,
                                              reg_coef=self.params['reg_coef'])

        new_conv_weight = new_conv_w.T.view(self.params['emb'], 3, self.params['patch_size'], self.params['patch_size'])
        self.merged_model.model.visual.conv1.weight = nn.Parameter(new_conv_weight)

        ##### Transformer blocks #####
        for layer_num in range(self.params['num_layers']):
            ##### Attention #####
            # q, k, v weights
            in_proj_weights = self.load_models_layers(layer_name="att_in_proj_weight", layer_num=layer_num)
            q_w_list, k_w_list, v_w_list = [], [], []
            for w in in_proj_weights:
                q_w, k_w, v_w = torch.split(w, [self.params['emb'], self.params['emb'], self.params['emb']], dim=0)
                q_w_list.append(q_w.T)
                k_w_list.append(k_w.T)
                v_w_list.append(v_w.T)

            new_q_w = self.regmean_layer_merge(w_list=q_w_list,
                                               activation_list=self.load_list_features('LN-att-{}'.format(layer_num), is_seq),
                                               reg_coef=self.params['reg_coef'],
                                               layer_num=layer_num)

            new_k_w = self.regmean_layer_merge(w_list=k_w_list,
                                               activation_list=self.load_list_features('LN-att-{}'.format(layer_num), is_seq),
                                               reg_coef=self.params['reg_coef'])

            new_v_w = self.regmean_layer_merge(w_list=v_w_list,
                                               activation_list=self.load_list_features('LN-att-{}'.format(layer_num), is_seq),
                                               reg_coef=self.params['reg_coef'])

            new_in_proj_w = torch.cat([new_q_w.T, new_k_w.T, new_v_w.T], dim=0)
            self.merged_model.model.visual.transformer.resblocks[layer_num].attn.in_proj_weight = nn.Parameter(new_in_proj_w)

            # out proj
            out_proj_weights = self.load_models_layers(layer_name="att_out_proj_weight", layer_num=layer_num)
            out_proj_weights = [w.T for w in out_proj_weights]
            new_out_proj_w = self.regmean_layer_merge(w_list=out_proj_weights,
                                                      activation_list=self.load_list_features('before-out-att-{}'.format(layer_num), is_seq),
                                                      reg_coef=self.params['reg_coef'])

            self.merged_model.model.visual.transformer.resblocks[layer_num].attn.out_proj.weight = nn.Parameter(new_out_proj_w.T)

            ##### MLP #####
            # first fully connected
            fc1_weights = self.load_models_layers(layer_name="fc_1_weight", layer_num=layer_num)
            fc1_weights = [w.T for w in fc1_weights]
            new_fc1_w = self.regmean_layer_merge(w_list=fc1_weights,
                                                 activation_list=self.load_list_features('LN-mlp-{}'.format(layer_num), is_seq),
                                                 reg_coef=self.params['reg_coef'])

            self.merged_model.model.visual.transformer.resblocks[layer_num].mlp[0].weight = nn.Parameter(new_fc1_w.T)

            # second fully connected
            fc2_weights = self.load_models_layers(layer_name="fc_2_weight", layer_num=layer_num)
            fc2_weights = [w.T for w in fc2_weights]
            new_fc2_w = self.regmean_layer_merge(w_list=fc2_weights,
                                                 activation_list=self.load_list_features('gelu-{}'.format(layer_num), is_seq),
                                                 reg_coef=self.params['reg_coef'])

            self.merged_model.model.visual.transformer.resblocks[layer_num].mlp[3].weight = nn.Parameter(new_fc2_w.T)

            print("Done merging layer {}".format(layer_num))

        ##### Output block #####
        out_proj_weights = self.load_models_layers(layer_name="out_proj")
        new_out_proj_w = self.regmean_layer_merge(w_list=out_proj_weights,
                                                  activation_list=self.load_list_features('LN-out', not is_seq),
                                                  reg_coef=self.params['reg_coef'])

        self.merged_model.model.visual.proj = nn.Parameter(new_out_proj_w)









