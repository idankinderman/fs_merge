from typing import List, Dict
import os
from pathlib import Path
import copy

import torch
import torch.nn as nn
import open_clip


from visualization import merge_training_plots, plot_multiple_losses, plot_variance_and_entropy, mul_training_plots


from modeling import ImageEncoder, ImageClassifier, ModuleWrapper, ClassificationHead
from train_eval.trainer import Trainer
from merges.merge_layers_vit_full import VITPreProcessingMergerFull, MergeAttentionSubBlockFull, MergeMLPSubBlockFull, MergeVITOutputFull, VITMergerFull, LinearBlock
from merges.merge_layers_bert import BERTMergerFull, BERTEmbeddingsMerger, BERTAttentionMerger, BERTMLPMerger, BERTPoolerMerger
from merges.general_merge import GeneralMerge
from losses.loss_functions import create_loss_function


class FSMerge(GeneralMerge):
    """
    A family of methods for merging transformers, using M,U matrices for merge and unmerge.
    model_type: The type of the model to merge.
    transformer_type: The type of the transformer to use, needed only for CLIP models.
    pre_trained: The pre-trained model to use, needed only for CLIP models.
    experiment_name: The name of the current merge experiment.
    experiment_dir: The directory to save the experiment.
    path_for_models: The path for the models to merge.
    models_to_merge: The names of the models to merge.
    models_indexes: The indexes of the models to merge. Needed only in BERT case.
    distributed: If to train the model in a distributed manner.
    epochs: The number of epochs to train the merge layer.
    batch_size: The batch size for the training of the merge layer.
    lr: The learning rate for the training of the merge layer.
    lr_diag: The learning rate for the diagonal of the MU matrices in the merge layer.
    wd: The weight decay for the training of the merge layer.
    scheduler_type: The type of the scheduler for the training of the merge layer.
    warmup_length: The warmup length for the training of the merge layer.
    StepLR_step_size: The step size for the StepLR scheduler.
    StepLR_gamma: The gamma for the StepLR scheduler.
    'diag_then_low_rank' for training the diagonal and then the low rank.
    'diag_then_diag_low_rank' for training the diagonal and then the diagonal + low rank.
    MU_init_method: The method to use for the initialization of the MU matrices in the merge layer.
    MU_type: The type of the MU matrices in the merge layer.
    num_features_train: The number of features from each dataset, to use for training the merge layer.
    num_features_test: The number of features from each dataset, to use for testing the merge layer.
    num_features_aug_train: The number of features from each dataset, to use for training the merge layer, after augmentation.
    comp_factor: The compression factor for the merge layer.
    rank: The rank for the M,U in the merge layer. -1 means to have a full rank.
    loss_degree: The degree of the loss function for the merge layer.
    loss_weights: The weights for each one of the features for the MSE loss.
    loss_type: The loss for training the merge layers. "rec" means to reconstruct the features of the original models.
    'rec_with_ids' means that if the images are from task X, compute the loss only according to the features of task X.
    'rec_with_inner' means to reconstruct the features of the original models, including inner features.
    loss_layer_num: if loss_type == 'rec_with_inner', this list contains the layers to use for the loss.
    scale_inner_type: how to compute the scale of the inner features. Then it will be used for weighting the MSE of
    the inner features in the loss. 'l1' for the L1 norm, 'l2' for the L2 norm, 'ones' for using ones.
    reg_coeff: The coefficient for the regularization of the merge layer.
    coeff_original_weights: The coefficients for the original weights of the models to merge.
    clip_grad_norm: Whether to clip the gradient norm in the merge layer.
    normalize_scale: Normalizing the scale of the features. If 0 then don't normalize. If -1 to normalize them to have
    the same scale as the first features. For any other number, normalize them to this number.
    norm_U_scale: If true, and used "normalize_scale", than after training will normalize the scale of U so it will give outputs as the original.
    datasets_to_eval: The vision_datasets to evaluate the merged model on.
    learn_tasks_sequentially: If to learn the tasks sequentially. First merge the two first models, then merge the result with the third model, and so on.
    """
    def __init__(self,
                 model_type: str,
                 experiment_name: str,
                 experiment_dir: str,
                 path_for_models: str,
                 models_to_merge: List[str],
                 epochs: int | List[int],
                 batch_size: int,
                 lr: float,
                 wd: float,
                 scheduler_type: str | None,
                 MU_init_method: str,
                 MU_type: str,
                 num_features_train: int,
                 num_features_test: int = 64,
                 num_features_aug_train: int = 0,
                 transformer_type: str = None,
                 pre_trained: str = None,
                 lr_diag: float = None,
                 comp_factor: float = 1.0,
                 rank: int = -1,
                 loss_degree: int = 2,
                 loss_weights: List[float] | None = None,
                 loss_type: str = 'rec',
                 loss_layer_num: List[int] | None = None,
                 scale_inner_type : str = 'l1',
                 reg_coeff: float = 0.0,
                 coeff_original_weights: Dict[str, float] | None = None,
                 freeze_LN: bool = False,
                 use_dropout: bool = False,
                 clip_grad_norm: bool = False,
                 with_early_stopping: bool = False,
                 normalize_scale: float = 0.0,
                 norm_U_scale: bool = True,
                 distributed: str | None = 'data_parallel',
                 print_per_epoch: int = 8,
                 learn_tasks_sequentially: bool = False,
                 descriptor: str = None,
                 datasets_to_eval: List[str] | None = None,
                 models_indexes: List[int] | None = None,
                 warmup_length: int = 168,
                 StepLR_step_size: int = 50,
                 StepLR_gamma: float = 0.2,
                 merge_type: str = 'ours_full'):

        super(FSMerge, self).__init__(model_type=model_type,
                                      transformer_type=transformer_type,
                                      pre_trained=pre_trained,
                                      experiment_name=experiment_name,
                                      experiment_dir=experiment_dir,
                                      path_for_models=path_for_models,
                                      models_to_merge=models_to_merge,
                                      models_indexes=models_indexes,
                                      datasets_to_eval=datasets_to_eval,
                                      learn_tasks_sequentially=learn_tasks_sequentially,
                                      descriptor=descriptor)

        # Update the params
        # Paths
        self.params['path_for_zs_layers'] = os.path.join(self.params['path_for_layers'], 'zero_shot')
        # Merger layers architecture
        self.params['MU_init_method'] = MU_init_method
        self.params['MU_type'] = MU_type
        self.params['freeze_LN'] = freeze_LN
        self.params['use_dropout'] = use_dropout
        self.params['comp_factor'] = comp_factor
        self.params['rank'] = rank
        self.params['coeff_original_weights'] = coeff_original_weights
        # Merger layers data for training
        self.params['num_features_train'] = num_features_train
        self.params['num_features_test'] = num_features_test
        self.params['num_features_aug_train'] = num_features_aug_train
        self.params['normalize_scale'] = normalize_scale
        # Loss function
        self.params['loss_degree'] = loss_degree
        self.params['loss_weights'] = loss_weights
        self.params['loss_type'] = loss_type
        self.params['loss_layer_num'] = loss_layer_num
        self.params['reg_coeff'] = reg_coeff
        # Merger layers training
        self.params['distributed'] = distributed
        self.params['epochs'] = epochs if isinstance(epochs, list) else [epochs]
        self.params['batch_size'] = batch_size
        self.params['lr'] = lr
        self.params['lr_diag'] = lr_diag
        self.params['wd'] = wd
        self.params['clip_grad_norm'] = clip_grad_norm
        self.params['with_early_stopping'] = with_early_stopping
        self.params['scheduler_type'] = scheduler_type
        self.params['warmup_length'] = warmup_length
        self.params['StepLR_step_size'] = StepLR_step_size
        self.params['StepLR_gamma'] = StepLR_gamma
        # General
        self.params['print_per_epoch'] = print_per_epoch
        self.params['merge_type'] = merge_type
        self.params['norm_U_scale'] = norm_U_scale
        self.params['scale_inner_type'] = scale_inner_type.lower()

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
        self.args_for_MU_training.gamma = 0.5
        self.args_for_MU_training.devices = list(range(torch.cuda.device_count()))
        self.losses_lists = {}

        # Sanity checks
        if coeff_original_weights is not None:
            assert len(coeff_original_weights) == len(models_to_merge)

        if self.params['loss_type'] in ['rec_with_inner_att', 'rec_with_inner_mlp', 'rec_with_inner_att_ids', 'rec_with_inner_mlp_ids'] \
            and self.params['loss_layer_num'] == None:
            raise Exception("loss_layer_num must be specified for loss_type = rec_with_inner")

        if self.params['loss_type'] in ['rec_with_inner_att', 'rec_with_inner_mlp', 'rec_with_inner_att_ids', 'rec_with_inner_mlp_ids'] \
                and self.params['model_type'] == 'bert':
            raise Exception("loss_type = rec_with_inner is not supported for BERT models")

        if self.params['scale_inner_type'] not in ['l1', 'l2', 'ones']:
            raise ValueError('The scale_inner_type must be one of the following: l1, l2, ones.')

        if self.params['model_type'] == 'clip' and self.params['learn_tasks_sequentially']:
            raise ValueError('learn_tasks_sequentially is not supported for CLIP models')

        # Create the loss function
        self.loss_func = create_loss_function(degree=self.params['loss_degree'],
                                              loss_weights=self.params['loss_weights'],
                                              loss_type=self.params['loss_type'],
                                              loss_layer_num=self.params['loss_layer_num'],
                                              reg_coeff=self.params['reg_coeff'])

        self.create_merge_dir()

        # Here the training plots will be saved
        self.params['plots_path'] = os.path.join(self.params['path_to_save'], 'plots')
        Path(self.params['plots_path']).mkdir(parents=True, exist_ok=True)

        """
        # Here the merge layers will be saved
        self.params['merger_path'] = os.path.join(self.params['path_to_save'], 'merge_layers')
        Path(self.params['merger_path']).mkdir(parents=True, exist_ok=True)
        """

        # All the merged layers will be saved here
        self.merge_layers = {}

        # Get the model hyper parameters
        self.merged_model = self.create_dummy_model()
        self.get_model_hyper_parameters(self.merged_model)
        del self.merged_model

    def create_dummy_model(self):
        if self.params['model_type'] == 'bert':
            model = self.load_model(model_name=self.params['models_to_merge'][0], model_number=0)
        elif self.params['model_type'] == 'clip':
            model, self.clip_train_preprocess, self.clip_transform = \
                open_clip.create_model_and_transforms(self.params['transformer_type'],pretrained=self.params['pre_trained'])
        else:
            pretrained = 'openai'
            if self.params['model_type'] == 'ViT-L-14':
                pretrained = 'laion400m_e32'
            model = ImageEncoder(self.loaded_args, keep_lang=False, pretrained=pretrained)

        return model


    # Merge the models, evaluate, and save the experiment
    # with_eval - evaluate the merged model
    # eval_every_seq_iter - evaluate the merged model after each iteration of the sequential learning
    # with_multi_head_eval - evaluate the merged model with multi-head
    # with_save - save the merged model
    # with_all_plots - create all the plots
    # verification - if True, use verify the models inner layers
    def merge(self,
              with_eval: bool = True,
              eval_every_seq_iter: bool = False,
              with_multi_head_eval: bool = False,
              with_save: bool = False,
              verification: bool = False):

        task_sequence_iters = len(self.params['models_to_merge']) -1 if self.params['learn_tasks_sequentially'] else 1
        models_to_merge_original = self.params['models_to_merge']
        datasets_to_eval_original = self.params['datasets_to_eval']
        eval_text = ""
        self.verification = verification

        for curr_task_sequence_iter in range(task_sequence_iters):
            self.curr_task_sequence_iter = curr_task_sequence_iter
            epochs = self.params['epochs'][curr_task_sequence_iter]

            if self.params['learn_tasks_sequentially']:
                self.params['models_to_merge'] = models_to_merge_original[:curr_task_sequence_iter + 2]
                self.params['datasets_to_eval'] = datasets_to_eval_original[:curr_task_sequence_iter + 2]
                self.losses_lists = {}

            U_output, U_output_layer = self.merge_models(epochs)

            dict_loss = self.losses_lists.get('full_val_epoch', None)
            if dict_loss is not None:
                self.params['last_val_loss'] = dict_loss['full-loss'][-1]

            # Create plots
            if epochs > 0:
                self.use_merge_training_plots(layer_name='full', layer_num=None, with_all_plots=True,
                                              curr_task_sequence_iter=curr_task_sequence_iter)

            # Evaluate the model
            if with_eval and (eval_every_seq_iter or curr_task_sequence_iter == task_sequence_iters - 1):
                eval_text += self.eval_model_on_datasets(self.merged_model, U_output,
                                                        used_train_features=True, multi_head_eval=with_multi_head_eval)

            if self.params['learn_tasks_sequentially']:
                self.delete_directory(os.path.join(self.params['path_to_save'], 'merged_model')) # Delete the dir of the merged model layers
                self.save_merged_model(U_output, U_output_layer)
                self.extract_layers_from_merged_model()

        self.save_experiment_info(eval_text)

        # Save the model
        if with_save:
            self.save_merged_model(U_output, U_output_layer)
        else:
            self.delete_saved_models()
            self.delete_directory(os.path.join(self.params['path_to_save'], 'merged_model')) # Delete the dir of the merged model layers


    # Merge the models
    def merge_models(self, epochs):
        # Creating the merger
        merger = self.create_merger()

        # Count parameters
        self.params['num_parameters'] = sum(p.numel() for p in merger.parameters() if p.requires_grad)
        print("Number of parameters in the merged model: {}".format(self.params['num_parameters']))

        if epochs > 0:
            # Getting the features
            normalize_features = self.params['normalize_scale'] != 0.0
            dataset = self.create_features_dataset(train_shuffle=True, normalize_features=normalize_features)

            # Train the merger
            merger = self.train_merger(merger, dataset, epochs)

        # Creating the merged weights and inject them
        merged_weights_dict = merger.fold(device='cuda:0')
        self.merged_model = self.create_dummy_model()
        self.replace_layers(model=self.merged_model, layer_name='full', merged_weights_dict=merged_weights_dict)

        # Return the current U
        if epochs > 0:
            scales = dataset.scales if normalize_features else None
        else:
            scales = None

        with torch.no_grad():
            U_output = merger.get_U(device='cuda:0', scales=scales, norm_U_scale=self.params['norm_U_scale'])
            U_output_layer = merger.output_merger.get_U_layer(device='cuda:0') if self.params['learn_tasks_sequentially'] else None

        #self.verify_merged_model_after_merge(merger=merger, model=self.merged_model, U=U_output)
        return U_output, U_output_layer


    def create_merger(self):
        pre_processing_merger = self.create_merge_layer_pre_processing()

        transformer_merger_layers = []
        for layer_num in range(self.params['num_layers']):
            attention_merger = self.create_merge_layer_attention(layer_num=layer_num)
            transformer_merger_layers.append(attention_merger)

            mlp_merger = self.create_merge_layer_mlp(layer_num=layer_num)
            transformer_merger_layers.append(mlp_merger)

        output_merger = self.create_merge_layer_output()

        if self.params['model_type'] == 'bert':
            merger = BERTMergerFull(embeddings_merger=pre_processing_merger,
                                    transformer_merger_layers=transformer_merger_layers,
                                    pooler_merger=output_merger)

        else:
            merger = VITMergerFull(pre_processing_merger=pre_processing_merger,
                                   transformer_merger_layers=transformer_merger_layers,
                                   output_merger=output_merger,
                                   loss_type=self.params['loss_type'],
                                   loss_layer_num=self.params['loss_layer_num'], )


        return merger


    def create_merge_layer_pre_processing(self):
        if self.params['model_type'] == 'bert':
            word_embeddings = self.load_models_layers(layer_name="embeddings_word_embeddings_weight")
            position_embeddings = self.load_models_layers(layer_name="embeddings_position_embeddings_weight")
            token_type_embeddings = self.load_models_layers(layer_name="embeddings_token_type_embeddings_weight")
            if self.params['MU_init_method'] == 'average':
                LN_state_dict = self.get_average_layer_norm(layer_name='embeddings_LayerNorm')
            elif self.params['MU_init_method'] == 'first':
                LN_state_dict = self.get_average_layer_norm(layer_name='embeddings_LayerNorm', first=True)
            else:
                LN_state_dict = None

            return BERTEmbeddingsMerger(word_embeddings=word_embeddings,
                                        position_embeddings=position_embeddings,
                                        token_type_embeddings=token_type_embeddings,
                                        LN_state_dict=LN_state_dict,
                                        number_of_models=2 if self.params['learn_tasks_sequentially'] else len(self.params['models_to_merge']),
                                        # in learn_tasks_sequentially, we merge two models at a time
                                        comp_factor=self.params['comp_factor'],
                                        num_heads=self.params['num_heads'],
                                        rank=self.params['rank'],
                                        MU_type='diagonal_and_low_rank' if self.params['MU_type']=='att_heads' else self.params['MU_type'],
                                        MU_init_method=self.params['MU_init_method'],
                                        freeze_LN=self.params['freeze_LN'],
                                        config=self.params['bert_config'],
                                        use_dropout=self.params['use_dropout'])

        else:
            conv_weights = self.load_models_layers(layer_name="conv")
            class_embeddings = self.load_models_layers(layer_name="class_embedding")
            positional_embeddings = self.load_models_layers(layer_name="pos_embedding")
            if self.params['MU_init_method'] == 'average':
                LN_in_state_dict = self.get_average_layer_norm(layer_name='ln_pre_state_dict')
            elif self.params['MU_init_method'] == 'first':
                LN_in_state_dict = self.get_average_layer_norm(layer_name='ln_pre_state_dict', first=True)
            else:
                LN_in_state_dict = None

            return VITPreProcessingMergerFull(conv_weights=conv_weights,
                                              class_embeddings=class_embeddings,
                                              positional_embeddings=positional_embeddings,
                                              LN_in_state_dict=LN_in_state_dict,
                                              number_of_models=2 if self.params['learn_tasks_sequentially'] else len(self.params['models_to_merge']),
                                              # in learn_tasks_sequentially, we merge two models at a time
                                              comp_factor=self.params['comp_factor'],
                                              rank=self.params['rank'],
                                              MU_type=self.params['MU_type'],
                                              MU_init_method=self.params['MU_init_method'],
                                              freeze_LN=self.params['freeze_LN'])


    def create_merge_layer_attention(self, layer_num, learn_LN_U=True, U_prev=None, LN_dict_prev=None):
        if self.params['model_type'] == 'bert':
            key_weight_list = self.load_models_layers(layer_name="encoder_layer_{}_attention_self_key_weight", layer_num=layer_num)
            key_bias_list = self.load_models_layers(layer_name="encoder_layer_{}_attention_self_key_bias", layer_num=layer_num)
            query_weight_list = self.load_models_layers(layer_name="encoder_layer_{}_attention_self_query_weight", layer_num=layer_num)
            query_bias_list = self.load_models_layers(layer_name="encoder_layer_{}_attention_self_query_bias", layer_num=layer_num)
            value_weight_list = self.load_models_layers(layer_name="encoder_layer_{}_attention_self_value_weight", layer_num=layer_num)
            value_bias_list = self.load_models_layers(layer_name="encoder_layer_{}_attention_self_value_bias", layer_num=layer_num)
            linear_weight_list = self.load_models_layers(layer_name="encoder_layer_{}_attention_output_dense_weight", layer_num=layer_num)
            linear_bias_list = self.load_models_layers(layer_name="encoder_layer_{}_attention_output_dense_bias", layer_num=layer_num)

            if self.params['MU_init_method'] == 'average':
                LN_att_state_dict = self.get_average_layer_norm(layer_name=f'encoder_layer_{layer_num}_attention_output_LayerNorm')
            elif self.params['MU_init_method'] == 'first':
                LN_att_state_dict = self.get_average_layer_norm(layer_name=f'encoder_layer_{layer_num}_attention_output_LayerNorm', first=True)
            else:
                LN_att_state_dict = None

            return BERTAttentionMerger(key_weight_list=key_weight_list,
                                       key_bias_list=key_bias_list,
                                       query_weight_list=query_weight_list,
                                       query_bias_list=query_bias_list,
                                       value_weight_list=value_weight_list,
                                       value_bias_list=value_bias_list,
                                       linear_weight_list=linear_weight_list,
                                       linear_bias_list=linear_bias_list,
                                       LN_att_state_dict=LN_att_state_dict,
                                       number_of_models=2 if self.params['learn_tasks_sequentially'] else len(self.params['models_to_merge']),
                                       # in learn_tasks_sequentially, we merge two models at a time
                                       config=self.params['bert_config'],
                                       num_heads=self.params['num_heads'],
                                       comp_factor=self.params['comp_factor'],
                                       rank=self.params['rank'],
                                       MU_type=self.params['MU_type'],
                                       MU_init_method=self.params['MU_init_method'],
                                       freeze_LN=self.params['freeze_LN'],
                                       use_dropout=self.params['use_dropout'])

        else:
            in_proj_weight_list = self.load_models_layers(layer_name="att_in_proj_weight", layer_num=layer_num)
            in_proj_bias_list = self.load_models_layers(layer_name="att_in_proj_bias", layer_num=layer_num)
            out_proj_weight_list = self.load_models_layers(layer_name="att_out_proj_weight", layer_num=layer_num)
            out_proj_bias_list = self.load_models_layers(layer_name="att_out_proj_bias", layer_num=layer_num)
            if self.params['MU_init_method'] == 'average':
                LN_att_state_dict = self.get_average_layer_norm(layer_name='ln_1_state_dict_{}'.format(layer_num))
            elif self.params['MU_init_method'] == 'first':
                LN_att_state_dict = self.get_average_layer_norm(layer_name='ln_1_state_dict_{}'.format(layer_num), first=True)
            else:
                LN_att_state_dict = None

            return MergeAttentionSubBlockFull(in_proj_weight_list=in_proj_weight_list,
                                              in_proj_bias_list=in_proj_bias_list,
                                              out_proj_weight_list=out_proj_weight_list,
                                              out_proj_bias_list=out_proj_bias_list,
                                              LN_att_state_dict=LN_att_state_dict,
                                              number_of_models=2 if self.params['learn_tasks_sequentially'] else len(self.params['models_to_merge']),
                                              # in learn_tasks_sequentially, we merge two models at a time
                                              num_heads=self.params['num_heads'],
                                              comp_factor=self.params['comp_factor'],
                                              rank=self.params['rank'],
                                              MU_type=self.params['MU_type'],
                                              MU_init_method=self.params['MU_init_method'],
                                              learn_LN_U=learn_LN_U,
                                              U_prev=U_prev,
                                              LN_dict_prev=LN_dict_prev,
                                              freeze_LN=self.params['freeze_LN'])



    def create_merge_layer_mlp(self, layer_num):
        if self.params['model_type'] == 'bert':
            linear1_weight_list = self.load_models_layers(layer_name="encoder_layer_{}_intermediate_dense_weight", layer_num=layer_num)
            linear1_bias_list = self.load_models_layers(layer_name="encoder_layer_{}_intermediate_dense_bias", layer_num=layer_num)
            linear2_weight_list = self.load_models_layers(layer_name="encoder_layer_{}_output_dense_weight", layer_num=layer_num)
            linear2_bias_list = self.load_models_layers(layer_name="encoder_layer_{}_output_dense_bias", layer_num=layer_num)
            if self.params['MU_init_method'] == 'average':
                LN_mlp_state_dict = self.get_average_layer_norm(layer_name='encoder_layer_{}_output_LayerNorm'.format(layer_num))
            elif self.params['MU_init_method'] == 'first':
                LN_mlp_state_dict = self.get_average_layer_norm(layer_name='encoder_layer_{}_output_LayerNorm'.format(layer_num),
                                                                first=True)
            else:
                LN_mlp_state_dict = None

            return BERTMLPMerger(linear1_weight_list=linear1_weight_list,
                                 linear1_bias_list=linear1_bias_list,
                                 linear2_weight_list=linear2_weight_list,
                                 linear2_bias_list=linear2_bias_list,
                                 LN_mlp_state_dict=LN_mlp_state_dict,
                                 config=self.params['bert_config'],
                                 number_of_models=2 if self.params['learn_tasks_sequentially'] else len(self.params['models_to_merge']),
                                 # in learn_tasks_sequentially, we merge two models at a time
                                 comp_factor=self.params['comp_factor'],
                                 rank=self.params['rank'],
                                 num_heads=self.params['num_heads'],
                                 MU_type='diagonal_and_low_rank' if self.params['MU_type'] == 'att_heads' else self.params['MU_type'],
                                 MU_init_method=self.params['MU_init_method'],
                                 mlp_activation=self.params['mlp_activation'],
                                 freeze_LN=self.params['freeze_LN'],
                                 use_dropout=self.params['use_dropout'])

        else:
            linear1_weight_list = self.load_models_layers(layer_name="fc_1_weight", layer_num=layer_num)
            linear1_bias_list = self.load_models_layers(layer_name="fc_1_bias", layer_num=layer_num)
            linear2_weight_list = self.load_models_layers(layer_name="fc_2_weight", layer_num=layer_num)
            linear2_bias_list = self.load_models_layers(layer_name="fc_2_bias", layer_num=layer_num)
            if self.params['MU_init_method'] == 'average':
                LN_mlp_state_dict = self.get_average_layer_norm(layer_name='ln_2_state_dict_{}'.format(layer_num))
            elif self.params['MU_init_method'] == 'first':
                LN_mlp_state_dict = self.get_average_layer_norm(layer_name='ln_2_state_dict_{}'.format(layer_num), first=True)
            else:
                LN_mlp_state_dict = None

            return MergeMLPSubBlockFull(linear1_weight_list=linear1_weight_list,
                                        linear1_bias_list=linear1_bias_list,
                                        linear2_weight_list=linear2_weight_list,
                                        linear2_bias_list=linear2_bias_list,
                                        LN_mlp_state_dict=LN_mlp_state_dict,
                                        number_of_models=2 if self.params['learn_tasks_sequentially'] else len(self.params['models_to_merge']),# in learn_tasks_sequentially, we merge two models at a time
                                        comp_factor=self.params['comp_factor'],
                                        rank=self.params['rank'],
                                        MU_type=self.params['MU_type'],
                                        MU_init_method=self.params['MU_init_method'],
                                        mlp_activation=self.params['mlp_activation'],
                                        freeze_LN=self.params['freeze_LN'])


    def create_merge_layer_output(self):
        if self.params['model_type'] == 'bert':
            linear_weight_list = self.load_models_layers(layer_name="pooler_dense_weight")
            linear_bias_list = self.load_models_layers(layer_name="pooler_dense_bias")
            return BERTPoolerMerger(linear_weight_list=linear_weight_list,
                                    linear_bias_list=linear_bias_list,
                                    activation=self.params['pooled_activation'],
                                    number_of_models=self.curr_task_sequence_iter if self.params['learn_tasks_sequentially'] else len(self.params['models_to_merge']),
                                    # in learn_tasks_sequentially, we merge two models at a time, but the output is always according to the last model
                                    comp_factor=self.params['comp_factor'],
                                    rank=self.params['rank'],
                                    num_heads=self.params['num_heads'],
                                    MU_type='diagonal_and_low_rank' if self.params['MU_type'] == 'att_heads' else self.params['MU_type'],
                                    MU_init_method=self.params['MU_init_method'])

        else:
            linear_weight_list = self.load_models_layers(layer_name="out_proj")
            if self.params['MU_init_method'] == 'average':
                LN_output_dict = self.get_average_layer_norm(layer_name="ln_post_state_dict")
            elif self.params['MU_init_method'] == 'first':
                LN_output_dict = self.get_average_layer_norm(layer_name="ln_post_state_dict", first=True)
            else:
                LN_output_dict = None

            prev_merged_model_U = None
            if self.params['learn_tasks_sequentially'] and self.curr_task_sequence_iter > 0: # use the U layer of the previous model
                prev_merged_model_U = torch.load(os.path.join(self.params['path_to_save'], 'U_output_layer.pt'))

            return MergeVITOutputFull(linear_weight_list=linear_weight_list,
                                      LN_output_dict=LN_output_dict,
                                      num_models=2 if self.params['learn_tasks_sequentially'] else len(self.params['models_to_merge']),
                                      comp_factor=self.params['comp_factor'],
                                      rank=self.params['rank'],
                                      MU_type=self.params['MU_type'],
                                      MU_init_method=self.params['MU_init_method'],
                                      freeze_LN=self.params['freeze_LN'],
                                      learn_tasks_sequentially=self.params['learn_tasks_sequentially'],
                                      curr_task_sequence_iter=self.curr_task_sequence_iter,
                                      prev_merged_model_U=prev_merged_model_U,)


    def train_merger(self, merger, dataset, epochs):
        # Train the merger
        self.args_for_MU_training.lr = self.params['lr']
        self.args_for_MU_training.lr_diag = self.params['lr_diag']
        trainer = self.get_trainer()

        merger = trainer.train_model(model=merger, epochs=epochs, dataset=dataset)

        merger = merger.module
        self.save_trainer_statistics(trainer, layer_name='full')
        # self.print_merger_layers(merger)
        return merger

    def get_trainer(self):
        what_is_trained = 'bert_merge_layer' if self.params['model_type'] == 'bert' else 'merge_layer'
        return Trainer(args=self.args_for_MU_training, loss_fn=self.loss_func, epoch_per_eval = 180,
                    clip_grad_norm=self.params['clip_grad_norm'], out_dim=self.params['out_dim'],
                    print_per_epoch=self.params['print_per_epoch'], with_eval=True, eval_type='loss_test',
                    loss_type=self.params['loss_type'], with_early_stopping=self.params['with_early_stopping'])


    # Helper function to update or initialize a list within the losses_lists dictionary
    def save_trainer_statistics(self, trainer, layer_name, layer_num=None):
        def update_list_or_dict(key, new_data):
            if isinstance(new_data, dict):
                # If the new data is a dictionary, handle accordingly
                if key not in self.losses_lists:
                    self.losses_lists[key] = {k: [] for k in new_data.keys()}
                for sub_key, sub_list in new_data.items():
                    if sub_key in self.losses_lists[key]:
                        self.losses_lists[key][sub_key].extend(sub_list)
                    else:
                        self.losses_lists[key][sub_key] = sub_list
            else:
                # If the new data is a list (or a single value), handle as before
                if key in self.losses_lists and isinstance(self.losses_lists[key], list):
                    self.losses_lists[key].extend(new_data if isinstance(new_data, list) else [new_data])
                else:
                    self.losses_lists[key] = new_data if isinstance(new_data, list) else [new_data]

        # Update or initialize each key with the appropriate data from the trainer
        layer_string = layer_name if layer_num is None else '{}_{}'.format(layer_name, layer_num)
        update_list_or_dict('{}_train'.format(layer_string), trainer.train_loss)
        update_list_or_dict('{}_train_epoch'.format(layer_string), [None])  # Assuming you want to keep it as a list
        update_list_or_dict('{}_val_epoch'.format(layer_string), trainer.val_loss_each_epoch)
        update_list_or_dict('{}_end_of_epoch_step_num'.format(layer_string), trainer.end_of_epoch_step_num)  # Assuming single value, wrapped in a list
        update_list_or_dict('{}_lr'.format(layer_string), trainer.lr_list)
        if self.params['model_type'] != 'clip':
            update_list_or_dict('{}_var_val_epoch'.format(layer_string), trainer.val_var_each_epoch)
            update_list_or_dict('{}_entropy_val_epoch'.format(layer_string), trainer.val_ent_each_epoch)
            self.losses_lists['{}_features_scale'.format(layer_string)] = trainer.features_scale

        # Save the inner losses
        if trainer.train_inner_loss:
            self.train_inner_loss = trainer.train_inner_loss

        if trainer.val_inner_loss_each_epoch:
            self.val_inner_loss_each_epoch = trainer.val_inner_loss_each_epoch



    def use_merge_training_plots(self, layer_name, layer_num=None, with_all_plots=False, curr_task_sequence_iter=None):
        # Plot loss
        layer_full_name = '{}_{}'.format(layer_name, layer_num) if layer_num is not None else layer_name
        sequence_task_str = f"_seq_iter_{curr_task_sequence_iter}" if self.params['learn_tasks_sequentially'] else ""

        # Plot loss
        if self.params['model_type'] == 'clip':
            mul_training_plots(train_loss=self.losses_lists['train'],
                               val_loss_each_epoch=self.losses_lists['val_epoch'],
                               end_of_epoch_step_num=self.losses_lists['end_of_epoch_step_num'],
                               title='Distillation training - loss graph',
                               save_path=os.path.join(self.params['plots_path'], "loss_graph"))
        else:
            merge_training_plots(train_loss=self.losses_lists['{}_train'.format(layer_full_name)],
                                 train_loss_each_epoch=None,
                                 train_inner_loss=None,
                                 val_loss_each_epoch=self.losses_lists['{}_val_epoch'.format(layer_full_name)],
                                 val_inner_loss_each_epoch=None,
                                 end_of_epoch_step_num=self.losses_lists['{}_end_of_epoch_step_num'.format(layer_full_name)],
                                 horizontal_line=self.merged_model_loss_dict.get(layer_full_name,None),
                                 features_scales=self.losses_lists['{}_features_scale'.format(layer_full_name)],
                                 title='{} merger training {}'.format(layer_full_name, sequence_task_str),
                                 save_path=os.path.join(self.params['plots_path'], "{}_{}".format(layer_full_name, sequence_task_str)))

        if self.params['loss_type'].lower() in ['rec_with_inner_att', 'rec_with_inner_mlp', 'rec_with_inner_att_ids', 'rec_with_inner_mlp_ids']:

            merge_training_plots(train_loss=self.losses_lists['{}_train'.format(layer_full_name)],
                                 train_loss_each_epoch=None,
                                 train_inner_loss=self.train_inner_loss,
                                 val_loss_each_epoch=self.losses_lists['{}_val_epoch'.format(layer_full_name)],
                                 val_inner_loss_each_epoch=self.val_inner_loss_each_epoch,
                                 end_of_epoch_step_num=self.losses_lists['{}_end_of_epoch_step_num'.format(layer_full_name)],
                                 horizontal_line=self.merged_model_loss_dict.get(layer_full_name, None),
                                 features_scales=self.losses_lists['{}_features_scale'.format(layer_full_name)],
                                 title='{} merger training {}'.format(layer_full_name, sequence_task_str),
                                 save_path=os.path.join(self.params['plots_path'], "all_{}_{}".format(layer_full_name, sequence_task_str)))




    # Replace a specific layers in a given model, according to our stage in the merging process
    # model - the model to replace the layers in
    # model_num - the number of the model in the list of models to merge
    # layer_name - indicates our stage in the merging process
    # merged_weights_dict - the weights to put in the model
    # merged_weights_dict - the weights to put in the model
    def replace_layers(self, model, layer_name: str, merged_weights_dict: dict[str, torch.Tensor]):
        # emb_size = model.model.visual.ln_pre.bias.shape[0]
        """
        for key in merged_weights_dict:
            print(key, merged_weights_dict[key].shape)
        """

        with torch.no_grad():
            if self.params['model_type'] == 'bert':
                # Embeddings
                model.embeddings.word_embeddings.weight = nn.Parameter(merged_weights_dict['word_embeddings'])
                model.embeddings.position_embeddings.weight = nn.Parameter(merged_weights_dict['position_embeddings'])
                model.embeddings.token_type_embeddings.weight = nn.Parameter(merged_weights_dict['token_type_embeddings'])
                model.embeddings.LayerNorm.weight = nn.Parameter(merged_weights_dict['LN_emb_weight'])
                model.embeddings.LayerNorm.bias = nn.Parameter(merged_weights_dict['LN_emb_bias'])

                for layer_num in range(self.params['num_layers']):
                    # Attention
                    model.encoder.layer[layer_num].attention.self.query.weight = nn.Parameter(merged_weights_dict[f'query_weight_{layer_num}'])
                    model.encoder.layer[layer_num].attention.self.query.bias = nn.Parameter(merged_weights_dict[f'query_bias_{layer_num}'])
                    model.encoder.layer[layer_num].attention.self.key.weight = nn.Parameter(merged_weights_dict[f'key_weight_{layer_num}'])
                    model.encoder.layer[layer_num].attention.self.key.bias = nn.Parameter(merged_weights_dict[f'key_bias_{layer_num}'])
                    model.encoder.layer[layer_num].attention.self.value.weight = nn.Parameter(merged_weights_dict[f'value_weight_{layer_num}'])
                    model.encoder.layer[layer_num].attention.self.value.bias = nn.Parameter(merged_weights_dict[f'value_bias_{layer_num}'])
                    model.encoder.layer[layer_num].attention.output.dense.weight = nn.Parameter(merged_weights_dict[f'att_linear_weight_{layer_num}'])
                    model.encoder.layer[layer_num].attention.output.dense.bias = nn.Parameter(merged_weights_dict[f'att_linear_bias_{layer_num}'])
                    model.encoder.layer[layer_num].attention.output.LayerNorm.weight = nn.Parameter(merged_weights_dict[f'LN_att_weight_{layer_num}'])
                    model.encoder.layer[layer_num].attention.output.LayerNorm.bias = nn.Parameter(merged_weights_dict[f'LN_att_bias_{layer_num}'])

                    # MLP
                    model.encoder.layer[layer_num].intermediate.dense.weight = nn.Parameter(merged_weights_dict[f'linear1_weight_{layer_num}'])
                    model.encoder.layer[layer_num].intermediate.dense.bias = nn.Parameter(merged_weights_dict[f'linear1_bias_{layer_num}'])
                    model.encoder.layer[layer_num].output.dense.weight = nn.Parameter(merged_weights_dict[f'linear2_weight_{layer_num}'])
                    model.encoder.layer[layer_num].output.dense.bias = nn.Parameter(merged_weights_dict[f'linear2_bias_{layer_num}'])
                    model.encoder.layer[layer_num].output.LayerNorm.weight = nn.Parameter(merged_weights_dict[f'LN_mlp_weight_{layer_num}'])
                    model.encoder.layer[layer_num].output.LayerNorm.bias = nn.Parameter(merged_weights_dict[f'LN_mlp_bias_{layer_num}'])

                # Pooler
                model.pooler.dense.weight = nn.Parameter(merged_weights_dict['pooler_linear_weight'])
                model.pooler.dense.bias = nn.Parameter(merged_weights_dict['pooler_linear_bias'])

            else:
                # Pre-processing
                model.model.visual.conv1.weight = nn.Parameter(merged_weights_dict['conv_weights'])
                model.model.visual.class_embedding = nn.Parameter(merged_weights_dict['cls'])
                model.model.visual.positional_embedding = nn.Parameter(merged_weights_dict['pos_emb'])
                model.model.visual.ln_pre.weight = nn.Parameter(merged_weights_dict['LN_in_weight'])
                model.model.visual.ln_pre.bias = nn.Parameter(merged_weights_dict['LN_in_bias'])

                for layer_num in range(self.params['num_layers']):
                    # Attention
                    model.model.visual.transformer.resblocks[layer_num].attn.in_proj_weight = nn.Parameter(
                        merged_weights_dict['in_proj_weight_{}'.format(layer_num)])
                    model.model.visual.transformer.resblocks[layer_num].attn.in_proj_bias = nn.Parameter(
                        merged_weights_dict['in_proj_b_{}'.format(layer_num)])
                    model.model.visual.transformer.resblocks[layer_num].attn.out_proj.weight = nn.Parameter(
                        merged_weights_dict['out_proj_weight_{}'.format(layer_num)])
                    model.model.visual.transformer.resblocks[layer_num].attn.out_proj.bias = nn.Parameter(
                        merged_weights_dict['out_proj_b_{}'.format(layer_num)])
                    model.model.visual.transformer.resblocks[layer_num].ln_1.weight = nn.Parameter(
                        merged_weights_dict['LN_att_weight_{}'.format(layer_num)])
                    model.model.visual.transformer.resblocks[layer_num].ln_1.bias = nn.Parameter(
                        merged_weights_dict['LN_att_bias_{}'.format(layer_num)])

                    # MLP
                    model.model.visual.transformer.resblocks[layer_num].mlp[0].weight = nn.Parameter(
                        merged_weights_dict['linear1_weight_{}'.format(layer_num)])
                    model.model.visual.transformer.resblocks[layer_num].mlp[0].bias = nn.Parameter(
                        merged_weights_dict['linear1_bias_{}'.format(layer_num)])
                    model.model.visual.transformer.resblocks[layer_num].mlp[3].weight = nn.Parameter(
                        merged_weights_dict['linear2_weight_{}'.format(layer_num)])
                    model.model.visual.transformer.resblocks[layer_num].mlp[3].bias = nn.Parameter(
                        merged_weights_dict['linear2_bias_{}'.format(layer_num)])
                    model.model.visual.transformer.resblocks[layer_num].ln_2.weight = nn.Parameter(
                        merged_weights_dict['LN_mlp_weight_{}'.format(layer_num)])
                    model.model.visual.transformer.resblocks[layer_num].ln_2.bias = nn.Parameter(
                        merged_weights_dict['LN_mlp_bias_{}'.format(layer_num)])

                # Output
                model.model.visual.ln_post.weight = nn.Parameter(merged_weights_dict['LN_out_weight'])
                model.model.visual.ln_post.bias = nn.Parameter(merged_weights_dict['LN_out_bias'])
                model.model.visual.proj = nn.Parameter(merged_weights_dict['linear_weight'])

        return model


    def create_features_dataset(self, train_shuffle, normalize_features):
        if self.params['model_type'] == 'clip':
            return self.create_features_dataset_clip()
        else:
            return self.create_features_dataset_vit(train_shuffle=train_shuffle, normalize_features=normalize_features)

    def create_features_dataset_vit(self, train_shuffle=True, batch_size=None, normalize_features=False):
        if self.params['loss_type'] not in ['rec', 'rec_with_ids', 'rec_with_inner_att_ids', 'rec_with_inner_mlp_ids']:
            raise ValueError("Not implemented for loss_type: {}".format(self.params['loss_type']))

        features_path = os.path.join(self.params['path_for_models'], 'features_{}'.format(self.params['num_features_train']))

        # 1.1. Load the inputs and the ids
        input_train_list, input_val_list, input_early_stopping_list, ids_train_list, ida_val_list, ids_early_stopping_list = [], [], [], [], [], []
        for model_num, model_name in enumerate(self.params['models_to_merge']):
            model_name = model_name.replace('finetuned_', '')

            input_train_path, input_val_path, input_aug_train_path, input_early_stopping_path = \
                self.build_path_for_specific_features(features_path, model_name, model_num, what_to_load='input')

            input_train_curr = self.load_feature_dataset(dataset_path=input_train_path,
                                                         dataset_size=self.params['num_features_train'])
            input_train_list.append(input_train_curr)
            ids_train_list.append(torch.full((self.params['num_features_train'],), model_num, dtype=torch.int64))

            input_val_curr = self.load_feature_dataset(dataset_path=input_val_path,
                                                       dataset_size=self.params['num_features_test'])
            input_val_list.append(input_val_curr)
            ida_val_list.append(torch.full((self.params['num_features_test'],), model_num, dtype=torch.int64))

            # 1.2. Load the augmented inputs
            if self.params['num_features_aug_train'] > 0 and self.params['model_type'] != 'bert':
                input_aug_train_curr = self.load_feature_dataset(dataset_path=input_aug_train_path,
                                                                 dataset_size=self.params['num_features_aug_train'])
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
        ids_val = torch.cat(ida_val_list, dim=0)
        input_early_stopping = None
        ids_early_stopping = None
        if self.params['with_early_stopping'] and self.params['model_type'] == 'bert':
            input_early_stopping = torch.cat(input_early_stopping_list, dim=0)
            ids_early_stopping = torch.cat(ids_early_stopping_list, dim=0)

        # 2. Load the targets from each original model.
        # target_train_list is a list of elements, the i-th element is a tensor of all the targets created by the i-th model
        target_train_list, target_val_list, target_early_stopping_list = [], [], []
        for model_num, model_name in enumerate(self.params['models_to_merge']): # forwards on the features created by each finetuned model
            model_name = model_name.replace('finetuned_', '')

            for data_name in self.params['models_to_merge']: # forwards on the features created by each dataset
                data_name = data_name.replace('finetuned_', '')

                if (model_name != data_name) and (self.params['loss_type'] in ['rec', 'rec_with_ids', 'rec_with_inner_att_ids','rec_with_inner_mlp_ids']):
                    # We won't use the features of model X created from dataset Y
                    continue

                target_train_curr, target_val_curr, target_early_stopping_curr = \
                    self.load_relevant_features(features_path=features_path, model_name=model_name, model_num=model_num,
                                                data_name=data_name, layer_name='output')

                target_train_list.append(target_train_curr)
                target_val_list.append(target_val_curr)
                if self.params['with_early_stopping'] and self.params['model_type'] == 'bert':
                    target_early_stopping_list.append(target_early_stopping_curr)

        # 3. Load inner features for training.
        # The keys are the layer numbers, the values are lists of tensors,
        # each tensor is the inner features of a model, with shape (N, seq_len, emb_dim)
        inner_target_train_dict, inner_target_val_dict = None, None
        if self.params['loss_type'] in ['rec_with_inner_att', 'rec_with_inner_mlp', 'rec_with_inner_att_ids', 'rec_with_inner_mlp_ids']:

            inner_target_train_dict, inner_target_val_dict = {}, {}
            for layer_num in self.params['loss_layer_num']: # The layer number of the inner target
                inner_target_train_dict[layer_num] = []
                inner_target_val_dict[layer_num] = []
                layer_name = 'attn-{}'.format(layer_num) if 'att' in self.params['loss_type'] else 'fc2-{}'.format(layer_num)

                for model_name in self.params['models_to_merge']:  # forwards on the features created by each finetuned model
                    model_name = model_name.replace('finetuned_', '')

                    for data_name in self.params['models_to_merge']:  # forwards on the features created by each dataset
                        data_name = data_name.replace('finetuned_', '')

                        if (model_name != data_name) and (self.params['loss_type'] in ['rec_with_ids',
                                                          'rec_with_inner_att_ids', 'rec_with_inner_mlp_ids']):
                            # We won't use the features of model X created from dataset Y
                            continue

                        inner_train_curr, inner_val_curr = \
                            self.load_relevant_features(features_path=features_path, model_name=model_name, model_num=model_num,
                                                        data_name=data_name, layer_name=layer_name)

                        inner_target_train_dict[layer_num].append(inner_train_curr)
                        inner_target_val_dict[layer_num].append(inner_val_curr)

        # 4. Load attention_mask and token_type_ids for BERT models
        attention_mask_train, attention_mask_val, attention_mask_early_stopping, token_type_train, token_type_val,\
            token_type_early_stopping = None, None, None, None, None, None
        if self.params['model_type'] == 'bert':
            attention_mask_train_list, attention_mask_val_list, attention_mask_early_stopping_list,\
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
                if self.params['with_early_stopping']:
                    attention_mask_early_stopping = torch.cat(attention_mask_early_stopping_list, dim=0)
                    token_type_early_stopping = torch.cat(token_type_early_stopping_list, dim=0)

            attention_mask_train = torch.cat(attention_mask_train_list, dim=0)
            attention_mask_val = torch.cat(attention_mask_val_list, dim=0)
            token_type_train = torch.cat(token_type_train_list, dim=0)
            token_type_val = torch.cat(token_type_val_list, dim=0)

        # 5. Create the dataset
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
              train_shuffle=train_shuffle, batch_size=batch_size)

        return dataset



    # Load the target using the features_path, data_name, model_name and layer_name.
    # Including the train, val and maybe augmented train.
    def load_relevant_features(self, features_path, model_name, data_name, model_num, layer_name):
        if self.params['model_type'] != 'bert':
            curr_feature_path = os.path.join(features_path, f"dataset_{data_name}",
                                             f"model_finetuned_{model_name}", f"{layer_name}" + '_{}')
            curr_feature_path_val = os.path.join(features_path, f"dataset_{data_name}",
                                             f"model_finetuned_{model_name}", f"{layer_name}" + '_val')
        else:

            if self.params['model_type'] != 'bert':
                curr_feature_path = os.path.join(self.params['path_for_models'], model_name,
                                                 f"seed_{self.params['models_indexes'][model_num]}",
                                                 f"features_{self.params['num_features_train']}",
                                                 f"{layer_name}_train")
                curr_feature_path_val = os.path.join(self.params['path_for_models'], model_name,
                                                     f"seed_{self.params['models_indexes'][model_num]}",
                                                     f"features_{self.params['num_features_train']}",
                                                     f"{layer_name}_eval")
                curr_feature_path_early_stopping = os.path.join(self.params['path_for_models'], model_name,
                                                                f"seed_{self.params['models_indexes'][model_num]}",
                                                                f"features_{self.params['num_features_train']}",
                                                                f"{layer_name}_early_stopping")
            else:
                curr_feature_path = os.path.join(self.params['path_for_models'], model_name, f"seed_{self.params['models_indexes'][model_num]}",
                                         f"features_{self.params['num_features_train']}", f"{layer_name}_train")
                curr_feature_path_val = os.path.join(self.params['path_for_models'], model_name, f"seed_{self.params['models_indexes'][model_num]}",
                                                 f"features_{self.params['num_features_train']}", f"{layer_name}_eval")
                curr_feature_path_early_stopping = os.path.join(self.params['path_for_models'], model_name, f"seed_{self.params['models_indexes'][model_num]}",
                                                    f"features_{self.params['num_features_train']}", f"{layer_name}_early_stopping")

        target_train = self.load_feature_dataset(dataset_path=curr_feature_path.format('train'),
                                                 dataset_size=self.params['num_features_train'])
        target_val = self.load_feature_dataset(dataset_path=curr_feature_path_val,
                                               dataset_size=self.params['num_features_test'])

        if self.params['with_early_stopping'] and self.params['model_type'] == 'bert':
            target_early_stopping = self.load_feature_dataset(dataset_path=curr_feature_path_early_stopping, dataset_size=200)
        else:
            target_early_stopping = None

        # Load the augmented targets
        if self.params['num_features_aug_train'] > 0 and self.params['model_type'] != 'bert':
            target_aug_train = self.load_feature_dataset(dataset_path=curr_feature_path.format('augmented_train'),
                                                         dataset_size=self.params['num_features_aug_train'])
            target_train = torch.cat([target_train, target_aug_train], axis=0)

        return target_train, target_val, target_early_stopping



