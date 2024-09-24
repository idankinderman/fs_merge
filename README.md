<h1 align="center"> Foldable SuperNets: Scalable Merging of Transformers with Different Initializations and Tasks </h1> 

<p align="center">
    <a href="https://www.linkedin.com/in/edan-kinderman-1320611b8/"> Edan Kinderman </a> •
    <a href="https://www.linkedin.com/in/itay-hubara-57739b29/?originalSubdomain=il"> Itay Hubara </a> •
    <a href="https://haggaim.github.io/"> Haggai Maron </a> •
    <a href="https://soudry.github.io/"> Daniel Soudry </a>
</p>

<p align="center">
<img src="figures/setting_overview.png" alt="scatter" width="90%"/>
</p>

## Abstract
Many recent methods aim to merge neural networks (NNs) with identical architectures trained on different tasks to obtain a single multi-task model. Most existing works tackle the simpler setup of merging NNs initialized from a common pre-trained network, where simple heuristics like weight averaging work well. This work targets a more challenging goal: merging large transformers trained on different tasks from distinct initializations. First, we demonstrate that traditional merging methods fail catastrophically in this setup. To overcome this challenge, we propose Foldable SuperNet Merge (FS-Merge), a method that optimizes a SuperNet to fuse the original models using a feature reconstruction loss. FS-Merge is simple, data-efficient, and capable of merging models of varying widths. We test FS-Merge against existing methods, including knowledge distillation, on MLPs and transformers across various settings, sizes, tasks, and modalities. FS-Merge consistently outperformed them, achieving SOTA results, particularly in limited data scenarios.

<p align="center">
<img src="figures/local_global_fs_merge.png" alt="scatter" width="90%"/>
</p>

### Section 1.1 Installation
Create a virtual environment and install the dependencies:
```bash
conda create -n zipit python=3.7
conda activate zipit
pip install torch torchvision torchaudio
pip install -r requirements.txt
```
TODO

## Usage

### 1. Load the data and the models
To begin, load the datasets, the classification heads and the models you want to merge. The fine-tuned ViT-B-16 models referenced in our paper can be found here (link coming soon).
Alternatively, you can fine-tune your own models using the provided [src/finetune.py](src/finetune.py) script, which is based on [task vectors](https://github.com/mlfoundations/task_vectors) [[1]](#ref1) work.
You need to organize your files with a parent directory containing both the classification heads and the models. Inside this parent directory, create a folder named `heads` for storing the classification heads and another folder named `checkpoints` for storing the models.

### 2. Extract layers
Run the following command to extract and save the layers of the models you plan to merge.
```bash
python extract.py --extract_type layers --model_type ViT-B-16 --path_to_models <PATH>
```


### 3. Extract features
Use this command to extract and save the features from the models. These features will be used in the merging process.
To extract inner features required for methods like RegMean, `set extract_type = 'all'`.
The parameter `num_features_per_dataset` defines the number of images to be taken from each training dataset.
The `aug_factor` multiplies this number by applying data augmentations, effectively increasing the number of images used.
```bash
python extract.py --extract_type none --model_type ViT-B-16 --path_to_models <PATH> --num_features_per_dataset <NUM> --aug_factor <AUG> --datasets_for_features <DATA1> <DATA2>
```

### 4. Merge
Here is an example of merging a pair of models using FS-Merge with a low rank of `12`, `100` training images per dataset, and `800` augmented images per dataset. 
To use FS-Merge seq., set `learn_tasks_sequentially=True`. If you want to calculate the joint accuracy of the merged model, set `with_multi_head_eval=True`. To save the merged model, set `with_save=True`.
```python
pass
```

## Citation

If you use FS-Merge or this codebase in your work, please cite: TODO
```
```

## References and credits
* <a id="ref1">[[1]](https://arxiv.org/abs/2212.04089)</a> "Editing Models with Task Arithmetic", Ilharco, Gabriel and Ribeiro, Marco Tulio and Wortsman, Mitchell and Gururangan, Suchin and Schmidt, Ludwig and Hajishirzi, Hannaneh and Farhadi, Ali. The International Conference on Learning Representations, 2023.
