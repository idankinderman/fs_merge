import os
import argparse

import torch

from utils import set_seed
from merges.ensemble import GeneralEnsemble
from merges.average_merge import AverageMerge
from merges.distillation_merge import DistillationMerge
from merges.fs_merge import FSMerge
from merges.regmean_merge import RegMeanMerge
from merges.slerp_merge import SLERPMerge

if __name__ == '__main__':
    ############## General ##############
    set_seed(seed=42)
    model_type = 'ViT-B-16'
    experiment_dir = os.path.join('results')
    datasets_to_eval = ['Cars', 'CIFAR10']
    models_to_merge = [f"finetuned_{data}" for data in datasets_to_eval]

    if model_type == 'ViT-B-16':
        exp_name = '4_1_24_diff_pretrained_finetune'
    elif model_type == 'ViT-L-14':
        exp_name = '9_3_24_diff_pretrained_finetuned'

    path_for_models = os.path.join('..', 'experiments', model_type, exp_name)
    descriptor = 'Merging 2 ViT models with {}'

    ############## Experiments ##############

    num_features_test = 64
    scheduler_type = 'warmup'
    batch_size = 128
    init = 'first'
    wd = 0.001

    ###############################################################################################
    # Ensemble

    ensemble = GeneralEnsemble(model_type=model_type,
                               experiment_name="ensemble",
                               experiment_dir=experiment_dir,
                               path_for_models=path_for_models,
                               models_to_merge=models_to_merge,
                               datasets_to_eval=datasets_to_eval,
                               coefficients=None,
                               normalize_output=True,
                               descriptor=descriptor.format('ensemble'))

    ensemble.merge(with_multi_head_eval=True)

    ###############################################################################################
    # Average merge

    average_merger = AverageMerge(model_type=model_type,
                                  experiment_name="average",
                                  experiment_dir=experiment_dir,
                                  path_for_models=path_for_models,
                                  models_to_merge=models_to_merge,
                                  datasets_to_eval=datasets_to_eval,
                                  coefficients=None,
                                  descriptor=descriptor.format('average'))
    average_merger.merge(with_eval=True, with_save=False, with_multi_head_eval=True)

    ###############################################################################################
    # SLERP merge

    slerp_merger = SLERPMerge(model_type=model_type,
                              experiment_name="SLERP",
                              experiment_dir=experiment_dir,
                              path_for_models=path_for_models,
                              models_to_merge=models_to_merge,
                              datasets_to_eval=datasets_to_eval,
                              coefficient=None,
                              descriptor=descriptor.format('SLERP'),
                              center='zero',
                              normalize_task_vectors=False,
                              slerp_type='regular')

    slerp_merger.merge(with_eval=True, with_save=False, with_multi_head_eval=True)
    
    ###############################################################################################
    # regMean merge
    regmean_merge = RegMeanMerge(
        model_type=model_type,
        experiment_name=f'Regmean',
        experiment_dir=experiment_dir,
        path_for_models=path_for_models,
        models_to_merge=models_to_merge,
        datasets_to_eval=datasets_to_eval,
        num_features_train=10,
        num_features_aug_train=10,
        reg_coef=0.7,
        descriptor=descriptor.format('regmean'),
        init='average')

    regmean_merge.merge(with_eval=True, with_save=False, with_multi_head_eval=True)
    
    ###############################################################################################
    # Distillation merge
    distillation_merge = DistillationMerge(
        model_type=model_type,
        experiment_name=f"Distillation",
        experiment_dir=experiment_dir,
        path_for_models=path_for_models,
        models_to_merge=models_to_merge,
        datasets_to_eval=datasets_to_eval,
        num_features_train=100,
        num_features_test=num_features_test,
        num_features_aug_train=100,
        descriptor=descriptor.format('distillation'),
        epochs=100,
        batch_size=batch_size,
        lr=0.0001,
        wd=wd,
        init=init,
        scheduler_type=scheduler_type)

    distillation_merge.merge(with_eval=True, with_save=False, with_multi_head_eval=True)

    ###############################################################################################
    # FS-Merge low rank
    fs_merge = FSMerge(
        model_type=model_type,
        experiment_name=f"FS_Merge_rank_12",
        experiment_dir=experiment_dir,
        path_for_models=path_for_models,
        models_to_merge=models_to_merge,
        datasets_to_eval=datasets_to_eval,
        num_features_train=100,
        num_features_test=num_features_test,
        num_features_aug_train=100,
        descriptor=descriptor.format('fs_merge_low_rank'),
        epochs=100,
        batch_size=batch_size,
        lr=0.0001,
        wd=wd,
        scheduler_type=scheduler_type,
        MU_init_method=init,
        MU_type='diagonal_and_low_rank',
        rank=12)

    fs_merge.merge(with_eval=True, with_save=False, with_multi_head_eval=True)

    ###############################################################################################
    # FS-Merge diagonal
    fs_merge = FSMerge(
        model_type=model_type,
        experiment_name=f"FS_Merge_diagonal",
        experiment_dir=experiment_dir,
        path_for_models=path_for_models,
        models_to_merge=models_to_merge,
        datasets_to_eval=datasets_to_eval,
        num_features_train=100,
        num_features_test=num_features_test,
        num_features_aug_train=100,
        descriptor=descriptor.format('fs_merge_diagonal'),
        epochs=100,
        batch_size=batch_size,
        lr=0.001,
        wd=wd,
        scheduler_type=scheduler_type,
        MU_init_method=init,
        MU_type='diagonal',
        rank=666)

    fs_merge.merge(with_eval=True, with_save=False, with_multi_head_eval=True)

    ###############################################################################################

    print("\n\nDone")


