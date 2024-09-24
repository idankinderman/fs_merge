import os
import json
import pickle

from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np
import torch
import datasets
from datasets import load_dataset, load_metric, Dataset

from transformers import BertTokenizer
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertForMaskedLM,
    BertTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from modeling import TextClassifier

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
}

task_to_outputs = {
    'cola': 2,
    'mnli': 3,
    'mrpc': 2,
    'qnli': 2,
    'qqp': 2,
    'rte': 2,
    'sst2': 2,
    'stsb': 1,
}

create_validation_set = {
    "cola": False,
    "mnli": True,
    "mrpc": False,
    "qnli": False,
    "qqp": True,
    "rte": False,
    "sst2": True,
    "stsb": False,
    "wnli": False,
}

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    not_state_dict: bool = field(
        default=False, metadata={"help": "is path statedict or not"
        }
    )
    merged: bool = field(
        default=False, metadata={"help": "finetune from a merged model"
        }
    )
    model1_path: Optional[str] = field(
        default=None, metadata={"help": "statedict path 1, if needing to ft from merged model"
        }
    )
    model2_path: Optional[str] = field(
        default=None, metadata={"help": "statedict path 2, if needing to ft from merged model"
        }
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    head_only: bool = field(
        default=False,
        metadata={"help": "finetune only classification head params"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    index: str = field(
        default="0",
        metadata={"help": "The index of the model to use (for multiberts)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def create_eval_dataset(task, tokenizer, preprocess_function, max_seq_length, cache_dir, data_type):
    # Preprocessing the raw_datasets
    if task == 'mnli':
        split_name = 'validation_matched' # This is actually the test set
    else:
        split_name = 'validation' # This is actually the test set

    if data_type == 'val':
        split_name = 'train' # We will create a validation set from the training set

    raw_datasets = load_dataset("glue", task, cache_dir=cache_dir, split=split_name)

    if data_type == 'val' and not create_validation_set[task]:
        raise ValueError("No validation set for this task")

    # Padding strategy
    if max_seq_length > tokenizer.model_max_length:
        print(
            f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on dataset",
    )

    if data_type == 'val':
        raw_datasets = raw_datasets.train_test_split(test_size=0.15, seed=42, stratify_by_column="label")
        raw_datasets = raw_datasets["test"]

    eval_dataset = raw_datasets

    return eval_dataset

def get_eval_dataset(task, tokenizer, preprocess_function, max_seq_length, cache_dir, data_type):
    base_path = "/home/edank/task-vectors/data/text_data/"
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    data_path = os.path.join(base_path, f"{task}_{data_type}")
    data_exist = os.path.exists(data_path)

    if data_exist:
        print(f"Loading {task} {data_type} dataset from disk")
        data = datasets.load_from_disk(data_path)
    else:
        print(f"Creating {task} {data_type} dataset")
        data = create_eval_dataset(task, tokenizer, preprocess_function, max_seq_length, cache_dir, data_type=data_type)
        data.save_to_disk(data_path)

    return data


def get_metrics(task, model, model_args, cache_dir, data_type):
    # Get the tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Get dataset
    is_regression = (task == "stsb")
    padding = 'max_length'
    max_seq_length = 128
    max_seq_length = min(max_seq_length, tokenizer.model_max_length)
    sentence1_key, sentence2_key = task_to_keys[task]

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        return result

    eval_dataset = get_eval_dataset(task=task, tokenizer=tokenizer, preprocess_function=preprocess_function,
                                    max_seq_length=max_seq_length, cache_dir=cache_dir, data_type=data_type)

    # Get the metric function
    metric = load_metric("glue", task)

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        # print(Counter(preds))
        # print(Counter(p.label_ids))
        if task is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if padding == 'max_length':
        data_collator = default_data_collator

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        train_dataset=None,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Evaluation
    print('evaluate')

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [task]
    eval_datasets = [eval_dataset]
    if task == "mnli":
        tasks.append("mnli-mm")
        mismatched_eval = load_dataset("glue", task, cache_dir=cache_dir, split="validation_mismatched")
        eval_datasets.append(mismatched_eval.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on dataset",
        ))

    metric_list = []
    for eval_dataset, task in zip(eval_datasets, tasks):
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        metrics["eval_samples"] = len(eval_dataset)
        metric_list.append(metrics)
        # trainer.save_metrics("eval", metrics)

    return metric_list

# Evaluate the model on a specific text dataset.
# The task is classification, so a classification head is needed.
# dataset - the dataset to evaluate on. If None, will load the dataset according to dataset_name.
def eval_single_text_dataset(text_encoder, dataset_name, model_dir, U_output=None, data_type='test'):
    # Prepare the model
    is_regression = (dataset_name == "stsb")
    if not is_regression:
        num_labels = task_to_outputs[dataset_name]
    else:
        num_labels = 1

    classification_head = torch.load(os.path.join(model_dir, "layers", "classification_head"))
    config_file = open(os.path.join(model_dir, "config.json"))
    config = json.load(config_file)
    model = TextClassifier(text_encoder=text_encoder, classification_head=classification_head, config=config,
                           num_labels=num_labels, U_output=U_output)
    #model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model.to('cuda')
    model.eval()

    # Load the arguments
    def load_from_json(cls, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        # Create a new instance of cls, unpacking dictionary items as arguments
        return cls(**{f.name: data.get(f.name, None) for f in fields(cls)})

    data_args = load_from_json(DataTrainingArguments, os.path.join(model_dir, 'data_args.json'))
    model_args = load_from_json(ModelArguments, os.path.join(model_dir, 'model_args.json'))
    #training_args = load_from_json(TrainingArguments, os.path.join(model_dir, 'training_args.json'))

    print("\ndata_args:\n", data_args)
    print("\nmodel_args:\n", model_args)
    #print("\ntraining_args:\n", training_args)

    metrics = get_metrics(dataset_name, model, model_args=model_args, cache_dir=False, data_type=data_type)
    return metrics


def evaluate_text_datasets(text_encoder, eval_datasets, model_dirs, U_output=None):
    metric_dict = {}
    for dataset_num, dataset_name in enumerate(eval_datasets):
        if U_output is not None:
            # U_output is a matrix of shape [out_dim, num_models * out_dim]
            # Need to take the part of U_output that corresponds to the current dataset
            out_dim = U_output.shape[0]
            U_curr = U_output[:, dataset_num * out_dim: (dataset_num + 1) * out_dim]
        else:
            U_curr = None

        data_type_list = ['val', 'test']
        if not create_validation_set[dataset_name] and 'val' in data_type_list: data_type_list.remove('val')
        for data_type in data_type_list:
            results = eval_single_text_dataset(text_encoder=text_encoder, dataset_name=dataset_name,
                                               model_dir=model_dirs[dataset_num], U_output=U_curr, data_type=data_type)
            results = results[0]
            metric_dict['{}_{}_acc'.format(dataset_name, data_type)] = results['eval_accuracy']
            metric_dict['{}_{}_loss'.format(dataset_name, data_type)] = results['eval_loss']

    return None, metric_dict