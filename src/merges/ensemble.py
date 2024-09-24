from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn

from merges.general_merge import GeneralMerge


def abs_mean(tensor: torch.Tensor):
    return torch.mean(torch.abs(tensor))

class GeneralEnsemble(GeneralMerge):
    """
    Ensemble of few models. The output will be the average of the model's outputs.
    model_type: The type of the model to merge.
    experiment_name: The name of the current merge experiment.
    experiment_dir: The directory to save the experiment.
    path_for_models: The path for the models to merge. Don't need when merging BERT models.
    models_to_merge: The names of the models to merge.
    models_indexes: The indexes of the models to merge. Needed only in BERT case.
    datasets_to_eval: The vision_datasets to evaluate the merged model on.
    coefficients: The coefficients for output average. If none, then will use simple average.
    normalize_output: Whether to normalize the output of the models before averaging.
    """
    def __init__(self,
                 model_type: str,
                 experiment_name: str,
                 experiment_dir: str,
                 path_for_models: str,
                 models_to_merge: List[str],
                 models_indexes: List[int] | None = None,
                 datasets_to_eval: List[str] | None = None,
                 coefficients: List[float] | None = None,
                 normalize_output: bool = False,
                 descriptor: str = None):

        super(GeneralEnsemble, self).__init__(model_type=model_type,
                                              experiment_name=experiment_name,
                                              experiment_dir=experiment_dir,
                                              path_for_models=path_for_models,
                                              models_to_merge=models_to_merge,
                                              models_indexes=models_indexes,
                                              datasets_to_eval=datasets_to_eval,
                                              descriptor=descriptor)

        # Update the params
        self.params['normalize_output'] = normalize_output
        self.params['merge_type'] = 'ensemble'
        self.create_merge_dir()
        if coefficients is None:
            self.coefficients = [1 / len(models_to_merge) for _ in models_to_merge]
        else:
            assert len(coefficients) == len(models_to_merge)
            self.coefficients = coefficients



    # Merge the models, evaluate, and save the experiment
    def merge(self,
              with_eval: bool = True,
              with_multi_head_eval: bool = False,
              with_save: bool = False):

        models_list = []
        for i, model_name in enumerate(self.params['models_to_merge']):
            models_list.append(self.load_model(model_name=model_name, model_number=i))

        if self.params['model_type'] == 'bert':
            self.merged_model = EnsembleModelText(models_list,
                                                  self.coefficients,
                                                  self.params['normalize_output'])
        else:
            self.merged_model = EnsembleModel(models_list,
                                              self.coefficients,
                                              self.params['normalize_output'])

        if with_eval:
            eval_text = self.eval_model_on_datasets(self.merged_model, multi_head_eval=with_multi_head_eval)
            self.save_experiment_info(eval_text)

        if with_save:
            self.save_merged_model()


class EnsembleModel(nn.Module):
    def __init__(self, models_list, coefficients, normalize_output=False):

        super(EnsembleModel, self).__init__()
        self.models_list = torch.nn.ModuleList(models_list)
        self.coefficients = coefficients
        self.normalize_output = normalize_output
        self.train_preprocess = self.models_list[0].train_preprocess
        self.val_preprocess = self.models_list[0].val_preprocess

    def forward(self, x):
        y = 0
        for i, model in enumerate(self.models_list):
            curr_out = model(x)
            if self.normalize_output:
                curr_out = curr_out / abs_mean(curr_out)
            y += curr_out * self.coefficients[i]
        return y


class EnsembleModelText(nn.Module):
    def __init__(self, models_list, coefficients, normalize_output=False):

        super(EnsembleModelText, self).__init__()
        self.models_list = torch.nn.ModuleList(models_list)
        self.coefficients = coefficients
        self.normalize_output = normalize_output

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):

        y = 0
        for i, model in enumerate(self.models_list):
            curr_out = model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds,
                             output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states,
                             return_dict=return_dict)[1]

            if self.normalize_output:
                curr_out = curr_out / abs_mean(curr_out)
            y += curr_out * self.coefficients[i]

        return None, y