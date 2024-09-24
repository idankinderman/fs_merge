import os

from typing import List, Dict

import math
import numpy as np
import torch
import torch.nn as nn

from merges.general_merge import GeneralMerge
from modeling import ImageEncoder, ImageClassifier, ModuleWrapper, ClassificationHead


def flatten_weights(model):
    """
    From CL-Gym (https://openaccess.thecvf.com/content/CVPR2021W/CLVision/papers/Mirzadeh_CL-Gym_Full-Featured_PyTorch_Library_for_Continual_Learning_CVPRW_2021_paper.pdf).
    Flattens a PyTorch model. i.e., concat all parameters as a single, large vector.
    :param model: PyTorch model
    :return: the flattened (vectorized) model parameters either Torch tensor
    """
    with torch.no_grad():
        all_params = []
        for param in model.parameters():
            all_params.append(param.view(-1))
        return torch.cat(all_params)

def assign_weights(model, weights):
    """
    From CL-Gym (https://openaccess.thecvf.com/content/CVPR2021W/CLVision/papers/Mirzadeh_CL-Gym_Full-Featured_PyTorch_Library_for_Continual_Learning_CVPRW_2021_paper.pdf).
    Manually assigns `weights` of a Pytorch `model`.
    Note that weights is of vector form (i.e., 1D array or tensor).
    Usage: For implementation of Mode Connectivity SGD algorithm.
    :param model: Pytorch model.
    :param weights: A flattened (i.e., 1D) weight vector.
    :return: The `model` updated with `weights`.
    """
    state_dict = model.state_dict(keep_vars=True)
    # The index keeps track of location of current weights that is being un-flattened.
    index = 0
    # just for safety, no grads should be transferred.
    with torch.no_grad():
        for param in state_dict.keys():
            # ignore batchnorm params
            if 'running_mean' in param or 'running_var' in param or 'num_batches_tracked' in param:
                continue
            param_count = state_dict[param].numel()
            param_shape = state_dict[param].shape
            state_dict[param] = nn.Parameter(weights[index:index+param_count].reshape(param_shape))
            index += param_count
    model.load_state_dict(state_dict)
    return model


class SLERPMerge(GeneralMerge):
    """
    Merging models by averaging their weights using SLERP (spherical linear interpolation).
    model_type: The type of the model to merge.
    experiment_name: The name of the current merge experiment.
    experiment_dir: The directory to save the experiment.
    path_for_models: The path for the models to merge.
    models_to_merge: The names of the models to merge.
    datasets_to_eval: The vision_datasets to evaluate the merged model on.
    coefficient: The coefficients for the models to merge. If none, then will use simple average.
    center: The center of the sphere. Can be 'center' (the zero vector), 'mean' (the mean of the models' weights) or
    pre-trained (the weights of a pre-trained model).
    batch_size: The batch size for the evaluation.
    train_classification_head: Whether to train a classification head on the merged model.
    normalize_task_vectors: Whether to normalize the task vectors to the same norm before merging.
    """
    def __init__(self,
                 model_type: str,
                 experiment_name: str,
                 experiment_dir: str,
                 path_for_models: str,
                 models_to_merge: List[str],
                 models_indexes: List[int] | None = None,
                 datasets_to_eval: List[str] | None = None,
                 coefficient: float | None = None,
                 descriptor: str = None,
                 center : str = "center",
                 batch_size : int = None,
                 normalize_task_vectors : bool = False,
                 slerp_type : str = "regular",
                 train_classification_head : bool = False,):

        super(SLERPMerge, self).__init__(model_type=model_type,
                                         experiment_name=experiment_name,
                                         experiment_dir=experiment_dir,
                                         path_for_models=path_for_models,
                                         models_to_merge=models_to_merge,
                                         models_indexes=models_indexes,
                                         datasets_to_eval=datasets_to_eval,
                                         batch_size=batch_size,
                                         train_classification_head=train_classification_head,
                                         descriptor=descriptor)

        # Update the params
        self.params['merge_type'] = 'SLERP'
        self.params['center'] = center
        self.params['normalize_task_vectors'] = normalize_task_vectors
        self.params['slerp_type'] = slerp_type
        self.create_merge_dir()
        if coefficient is None:
            self.coefficient = 0.5
        else:
            self.coefficient = coefficient



    # Merge the models, evaluate, and save the experiment
    def merge(self,
              with_eval: bool = True,
              with_multi_head_eval: bool = False,
              with_save: bool = False):

        self.merge_models()
        if with_eval:
            eval_text = self.eval_model_on_datasets(self.merged_model, multi_head_eval=with_multi_head_eval)
            self.save_experiment_info(eval_text)

        if with_save:
            self.save_merged_model()


    def merge_models(self):
        with torch.no_grad():
            # Loading the models and flatting their weights
            model_a = self.load_model(model_name=self.params['models_to_merge'][0], model_number=0)
            model_a_vec = flatten_weights(model=model_a)
            del model_a

            model_b = self.load_model(model_name=self.params['models_to_merge'][1], model_number=1)
            model_b_vec = flatten_weights(model=model_b)
            del model_b

            model_a_norm = np.linalg.norm(model_a_vec)
            model_b_norm = np.linalg.norm(model_b_vec)
            self.params['model_norms'] = {self.params['models_to_merge'][0]: model_a_norm,
                                          self.params['models_to_merge'][1]: model_b_norm}

            print("original norms: ", model_a_norm, model_b_norm)

            # Defining the center of the sphere
            if self.params['center'].lower() in ['center', 'origin', 'zero']:
                center_vec = torch.zeros_like(model_a_vec)
            elif self.params['center'].lower() in ['mean', 'average']:
                center_vec = (model_a_vec + model_b_vec) / 2
            elif self.params['center'].lower() in ['pre-trained', 'pretrained']:
                directory_path = os.path.join(self.params['path_for_models'], 'checkpoints')
                zero_shot_model = torch.load(os.path.join(directory_path, 'zero_shot.pt'))
                center_vec = flatten_weights(model=zero_shot_model)
                del zero_shot_model
            else:
                raise ValueError("Invalid center for the sphere.")

            center_norm = np.linalg.norm(center_vec)
            self.params['model_norms']['center'] = center_norm
            print("norm of the center_vec: ", center_norm)

            # Centerizing and normalizing the vectors
            model_a_vec = model_a_vec - center_vec
            model_b_vec = model_b_vec - center_vec

            print("norms after centerizing: ", np.linalg.norm(model_a_vec), np.linalg.norm(model_b_vec))

            if self.params['normalize_task_vectors']:
                model_a_norm = np.linalg.norm(model_a_vec)
                model_b_norm = np.linalg.norm(model_b_vec)
                new_norm = (model_a_norm + model_b_norm) / 2
                model_a_vec = model_a_vec * (new_norm / model_a_norm)
                model_b_vec = model_b_vec * (new_norm / model_b_norm)

                print("norms after normalizing: ", np.linalg.norm(model_a_vec), np.linalg.norm(model_b_vec))

            # Perform the SLERP
            if self.params['slerp_type'].lower() == "regular":
                angle = math.acos(np.dot(model_a_vec, model_b_vec) / (np.linalg.norm(model_a_vec) * np.linalg.norm(model_b_vec)))
                if abs(angle - 0) < 1e-5 or abs(angle - math.pi) < 1e-5 :
                    print("angle is 0 or pi")
                    slerp_coef_a, slerp_coef_b = 0.5, 0.5
                else:
                    slerp_coef_a = (math.sin(self.coefficient * angle)) / (math.sin(angle) + 1e-6)
                    slerp_coef_b = (math.sin((1-self.coefficient) * angle)) / (math.sin(angle) + 1e-6)
                new_vec = slerp_coef_a * model_a_vec + slerp_coef_b * model_b_vec

            elif self.params['slerp_type'].lower() == "separate":
                model_a_norm = np.linalg.norm(model_a_vec)
                model_b_norm = np.linalg.norm(model_b_vec)
                model_new_norm = (model_a_norm + model_b_norm) / 2

                model_a_unit = model_a_vec / model_a_norm
                model_b_unit = model_b_vec / model_b_norm

                print("inside separate, unit vectors: ", np.linalg.norm(model_a_unit), np.linalg.norm(model_b_unit))

                angle = math.acos(np.dot(model_a_unit, model_b_unit))
                if abs(angle - 0) < 1e-5 or abs(angle - math.pi) < 1e-5 :
                    print("angle is 0 or pi")
                    slerp_coef_a, slerp_coef_b = 0.5, 0.5
                else:
                    slerp_coef_a = (math.sin(self.coefficient * angle)) / (math.sin(angle) + 1e-6)
                    slerp_coef_b = (math.sin((1-self.coefficient) * angle)) / (math.sin(angle) + 1e-6)
                new_vec = model_new_norm * (slerp_coef_a * model_a_unit + slerp_coef_b * model_b_unit)

            else:
                raise ValueError("Invalid slerp_type.")

            print("angle:", angle)
            print("slerp_coef_a:", slerp_coef_a)
            print("slerp_coef_b:", slerp_coef_b)
            print("norm of the new model without center vec: ", np.linalg.norm(new_vec))

            new_vec = new_vec + center_vec
            merged_model_norm = np.linalg.norm(new_vec)
            self.params['model_norms']['merged'] = merged_model_norm
            print("norm of the new model with the center vec: ", merged_model_norm)

            del model_a_vec, model_b_vec

            # Assigning the weights to the merged model
            if self.params['model_type'] == 'bert':
                self.merged_model = self.load_model(model_name=self.params['models_to_merge'][0], model_number=0)

            else:
                pretrained = 'openai'
                if self.params['model_type'] == 'ViT-L-14':
                    pretrained = 'laion400m_e32'
                self.merged_model = ImageEncoder(self.loaded_args, keep_lang=False, pretrained=pretrained)

            self.merged_model = assign_weights(model=self.merged_model, weights=new_vec)



