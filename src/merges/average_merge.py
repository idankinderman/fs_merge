from typing import List, Dict

import numpy as np
import torch

from merges.general_merge import GeneralMerge
from modeling import ImageEncoder, ImageClassifier, ModuleWrapper, ClassificationHead
from merges.slerp_merge import flatten_weights, assign_weights


class AverageMerge(GeneralMerge):
    """
    Merging models by averaging their weights.
    model_type: The type of the model to merge.
    experiment_name: The name of the current merge experiment.
    experiment_dir: The directory to save the experiment.
    path_for_models: The path for the models to merge.
    models_to_merge: The names of the models to merge.
    datasets_to_eval: The vision_datasets to evaluate the merged model on.
    normalize_merged_norm: Whether to normalize the merged model's weights to the same norm as the original models.
    coefficients: The coefficients for the models to merge. If none, then will use simple average.
    train_classification_head: Whether to train a classification head on the merged model.
    """
    def __init__(self,
                 model_type: str,
                 experiment_name: str,
                 experiment_dir: str,
                 path_for_models: str,
                 models_to_merge: List[str],
                 models_indexes: List[int] | None = None,
                 datasets_to_eval: List[str] | None = None,
                 normalize_merged_norm: bool = False,
                 coefficients: Dict[str, float] | None = None,
                 train_classification_head: bool = False,
                 descriptor: str = None):

        super(AverageMerge, self).__init__(model_type=model_type,
                                          experiment_name=experiment_name,
                                          experiment_dir=experiment_dir,
                                          path_for_models=path_for_models,
                                          models_to_merge=models_to_merge,
                                          models_indexes=models_indexes,
                                          datasets_to_eval=datasets_to_eval,
                                          train_classification_head=train_classification_head,
                                          descriptor=descriptor)

        # Update the params
        self.params['normalize_merged_norm'] = normalize_merged_norm
        self.params['merge_type'] = 'average'
        self.create_merge_dir()
        if coefficients is None:
            self.coefficients = {model_name: 1 / len(models_to_merge) for model_name in models_to_merge}
        else:
            assert len(coefficients) == len(models_to_merge)
            self.coefficients = coefficients



    # Merge the models, evaluate, and save the experiment
    def merge(self,
              with_eval: bool = True,
              with_multi_head_eval: bool = False,
              with_save: bool = False):

        self.merge_models()
        eval_text = ""
        if with_eval:
            eval_text = self.eval_model_on_datasets(self.merged_model, multi_head_eval=with_multi_head_eval)
        self.save_experiment_info(eval_text)

        if with_save:
            self.save_merged_model()

    def merge_models(self):
        self.params['model_norms'] = {}
        weight_vec = 0
        sum_norms = 0
        with torch.no_grad():
            # Loading the models and flatting their weights
            for i, model_name in enumerate(self.params['models_to_merge']):
                model = self.load_model(model_name=model_name, model_number=i)
                curr_vec = flatten_weights(model=model)
                curr_vec_norm = np.linalg.norm(curr_vec)
                sum_norms += curr_vec_norm
                self.params['model_norms'][model_name] = curr_vec_norm
                print("norm of the curr model vec: ", curr_vec_norm)
                weight_vec = weight_vec + curr_vec * self.coefficients[model_name]
                del model

            merged_model_norm = np.linalg.norm(weight_vec)
            self.params['model_norms']['merged'] = merged_model_norm
            print("norm of the new model vec: ", merged_model_norm)

            if self.params['normalize_merged_norm']:
                mean_norms = sum_norms / len(self.params['models_to_merge'])
                weight_vec = weight_vec * mean_norms / merged_model_norm
                merged_model_norm_new = np.linalg.norm(weight_vec)
                self.params['model_norms']['merged_normalized'] = merged_model_norm_new
                print("norm of the new model vec, after normalization: ", merged_model_norm_new)

            # Assigning the weights to the merged model
            if self.params['model_type'] == 'bert':
                self.merged_model = self.load_model(model_name=self.params['models_to_merge'][0], model_number=0)

            else:
                pretrained = 'openai'
                if self.params['model_type'] == 'ViT-L-14':
                    pretrained = 'laion400m_e32'
                self.merged_model = ImageEncoder(self.loaded_args, keep_lang=False, pretrained=pretrained)

            self.merged_model = assign_weights(model=self.merged_model, weights=weight_vec)




