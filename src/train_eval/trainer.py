import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from modeling import ModuleWrapper
from train_eval.eval import eval_single_dataset, eval_single_dataset_loss
from vision_datasets.registry import get_dataset
from vision_datasets.common import get_dataloader, maybe_dictionarize
from utils import cosine_lr
from losses.loss_wrapper import get_loss_wrapper

from merges.merge_layers_bert import compare_bert_merger

class Trainer():
    """
    Base class for Trainer component.
    """

    def __init__(self, args, loss_fn, clip_grad_norm=True, with_eval=True, eval_type='acc',
                 what_is_trained='VIT', models_to_merge=None, loss_type=None, out_dim=None,
                 with_early_stopping=False, early_stopping_patience=5, epoch_per_eval=1, epoch_per_print=10):
        self.args = args
        self.dataset_name = args.train_dataset
        self.loss_fn = loss_fn
        self.clip_grad_norm = clip_grad_norm # If True, will clip the gradient norm to 1.0
        self.with_eval = with_eval # Perform evaluation during training
        self.eval_type = eval_type # Evaluation type - 'acc' or 'loss'
        self.what_is_trained = what_is_trained # What is trained - 'VIT' or 'merge_layer'
        self.models_to_merge = models_to_merge # List of models the merge layer is merging
        self.loss_type = loss_type
        self.epoch_per_eval = epoch_per_eval
        self.epoch_per_print = epoch_per_print
        if self.models_to_merge is not None:
            self.num_models = len(self.models_to_merge)
        else:
            self.num_models = 1
        self.out_dim = out_dim # The output dimension of the model

            # Early stopping
        self.with_early_stopping = with_early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_counter = 0
        self.best_early_stopping_loss = 1e20

        if self.what_is_trained.lower() not in ['vit', 'vit_distillation', 'bert_distillation', 'merge_layer', 'bert_merge_layer']:
            raise Exception('what_is_trained must be in [VIT, VIT_distillation, merge_layer]')

        self.compute_loss = get_loss_wrapper(loss_type=self.loss_type,
                                             loss_fn=self.loss_fn,
                                             num_models=self.num_models,
                                             what_is_trained=self.what_is_trained)

        # Metrics
        # Loss
        self.train_loss = [] # All the losses during training
        self.val_loss_each_epoch = {} # The val loss on all the dataset per epoch
        self.val_loss_each_epoch['full-loss'] = []

        self.train_inner_loss = {}  # Dict to store the inner losses of the model
        self.val_inner_loss_each_epoch = {} # Dict to store the inner losses of the model

        # Variance
        self.val_var_each_epoch = {} # The val variance on all the dataset per epoch
        self.val_var_each_epoch['full-var'] = []

        # Entropy
        self.val_ent_each_epoch = {} # The val entropy on all the dataset per epoch
        self.val_ent_each_epoch['full-entropy'] = []

        # Accuracy
        self.train_acc = [] # The train accuracy per epoch
        self.val_acc = [] # The val accuracy per epoch
        self.test_acc = [] # The test accuracy per epoch

        # Else
        self.end_of_epoch_step_num = [] # The step number at the end of each epoch
        self.lr_list = [] # The learning rate per epoch
        self.features_scale = {}

        if eval_type not in ['acc', 'loss', 'loss_test']:
            raise Exception('eval_type must be in [acc, loss, loss_test]')


    def get_optimizer(self, model, lr, wd, lr_diag=None):
        if lr_diag == None:
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

        else:
            # Separate parameters based on the condition
            params_diag = []
            params = []

            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name.endswith("_diag"):
                        params_diag.append(param)
                    else:
                        params.append(param)

            # Create parameter groups with different learning rates
            param_groups = [
                {'params': params, 'lr': lr},
                {'params': params_diag, 'lr': lr_diag}
            ]

            optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)

        return optimizer, params


    def get_scheduler(self, optimizer, num_batches, epochs):
        if epochs is None:
            epochs = self.args.epochs
        if self.args.scheduler_type is None:
            return None
        elif self.args.scheduler_type == 'cosine_lr':
            return cosine_lr(optimizer, self.args.lr, self.args.warmup_length, epochs * num_batches)

        elif self.args.scheduler_type == 'steplr':
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.StepLR_step_size,
                                                   gamma=self.args.StepLR_gamma)
        elif self.args.scheduler_type == 'warmup':
            return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.args.lr, steps_per_epoch=num_batches,
                                                       epochs=epochs, anneal_strategy='linear')
        elif self.args.scheduler_type == 'lambda':
            lambda1 = lambda epoch: 0.9 ** epoch
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        else:
            raise Exception('scheduler_type must be in [None, cosine_lr, steplr, warmup]')


    def perform_scheduler_step(self, scheduler, step, end_of_epoch=False):
        if self.args.scheduler_type is None:
            pass
        elif self.args.scheduler_type == 'cosine_lr':
            if not end_of_epoch:
                scheduler(step)
        elif self.args.scheduler_type == 'steplr':
            if end_of_epoch:
                scheduler.step()
        elif self.args.scheduler_type == 'lambda':
            if end_of_epoch:
                scheduler.step()
        else:
            if not end_of_epoch:
                scheduler.step()

    # Evaluate the model on the train and test sets
    def perform_eval(self, model, dataset, step, with_all_plots):
        self.end_of_epoch_step_num.append(step)
        metric_dict = {}
        model.eval()
        if self.eval_type == 'acc':
            train_metric = eval_single_dataset(image_encoder=model.module.image_encoder,
                                               classification_head=model.module.classification_head,
                                               dataset_name=self.dataset_name,
                                               args=self.args,
                                               data_type='train',
                                               dataset=dataset,
                                               with_prints=False)
            self.train_acc.append(train_metric['top1-train'])
            metric_dict['train_acc'] = train_metric['top1-train']

            val_metric = eval_single_dataset(image_encoder=model.module.image_encoder,
                                             classification_head=model.module.classification_head,
                                             dataset_name=self.dataset_name,
                                             args=self.args,
                                             data_type='val',
                                             dataset=dataset,
                                             with_prints=False)
            self.val_acc.append(val_metric['top1-val'])
            metric_dict['val_acc'] = val_metric['top1-val']

            test_metric = eval_single_dataset(image_encoder=model.module.image_encoder,
                                             classification_head=model.module.classification_head,
                                             dataset_name=self.dataset_name,
                                             args=self.args,
                                             data_type='test',
                                             dataset=dataset,
                                             with_prints=False)
            self.test_acc.append(test_metric['top1-test'])
            metric_dict['test_acc'] = test_metric['top1-test']

        elif 'loss' in self.eval_type:
            val_metric = eval_single_dataset_loss(model=model,
                                                  data_type='val',
                                                  dataset=dataset,
                                                  loss_type=self.loss_type,
                                                  what_is_trained=self.what_is_trained,
                                                  num_of_models=len(self.models_to_merge),
                                                  with_all_plots=with_all_plots,
                                                  compute_loss=self.compute_loss)

            self.val_loss_each_epoch['full-loss'].append(val_metric['val-loss'])
            self.val_var_each_epoch['full-var'].append(val_metric['val-var'])
            self.val_ent_each_epoch['full-entropy'].append(val_metric['val-entropy'])
            for key in val_metric:
                if 'inner' in key:
                    # Add the inner loss to the list
                    if "val-{}".format(key) not in self.val_inner_loss_each_epoch:
                        self.val_inner_loss_each_epoch["val-{}".format(key)] = []
                    self.val_inner_loss_each_epoch["val-{}".format(key)].append(val_metric[key])


            # In case we evaluate on each task separately, we want to save the loss and var of each task
            if 'val-loss-split' in val_metric:
                for i, model_name in enumerate(self.models_to_merge):

                    if f'{model_name}-loss' not in self.val_loss_each_epoch:
                        self.val_loss_each_epoch[f'{model_name}-loss'] = []

                    if f'{model_name}-var' not in self.val_var_each_epoch:
                        self.val_var_each_epoch[f'{model_name}-var'] = []

                    if f'{model_name}-entropy' not in self.val_ent_each_epoch:
                        self.val_ent_each_epoch[f'{model_name}-entropy'] = []

                    self.val_loss_each_epoch[f'{model_name}-loss'].append(val_metric['val-loss-split'][i])
                    self.val_var_each_epoch[f'{model_name}-var'].append(val_metric['val-var-split'][i])
                    self.val_ent_each_epoch[f'{model_name}-entropy'].append(val_metric['val-entropy-split'][i])

            metric_dict['val_loss'] = val_metric['val-loss']

        return metric_dict

    def perform_early_stopping(self, model, dataset):
        print(f"early stopping with {len(dataset.early_stopping_dataset)} features, counter: {self.early_stopping_counter}, best loss: {self.best_early_stopping_loss}")
        val_metric = eval_single_dataset_loss(model=model,
                                              data_type='early_stopping',
                                              dataset=dataset,
                                              loss_type=self.loss_type,
                                              what_is_trained=self.what_is_trained,
                                              num_of_models=len(self.models_to_merge),
                                              compute_loss=self.compute_loss)

        loss = val_metric['early_stopping-loss']
        if loss < self.best_early_stopping_loss:
            self.best_early_stopping_loss = loss
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

        if self.early_stopping_counter >= self.early_stopping_patience:
            # Stop the training
            return True

        return False


    def create_distributed_model(self, model, gpu_id=None):
        if self.args.distributed == None:
            model = ModuleWrapper(model)
        elif self.args.distributed.lower() == 'data_parallel':
            model = torch.nn.DataParallel(model, device_ids=self.args.devices)
        else:
            raise Exception('distributed must be in [None, data_parallel, distributed_data_parallel]')

        return model


    def start_of_train_prints(self, metric_dict):
        if self.eval_type == 'acc':
            print(
                f"Start of epoch: 0 \t"
                f"Train acc : {metric_dict['train_acc']:.2f}%"
                f" | Val acc : {metric_dict['val_acc']:.2f}%",
                flush=True)

        elif self.eval_type == 'loss':
            print(
                f"Start of epoch: 0 \t"
                f"Train loss : {metric_dict['train_loss']:.4f}"
                f" | Val loss : {metric_dict['val_loss']:.4f}",
                flush=True
            )

        elif self.eval_type == 'loss_test':
            print(
                f"Start of epoch: 0 \t"
                f" | Val loss : {metric_dict['val_loss']:.4f}",
                flush=True
            )

    def end_of_epoch_prints(self, epoch, metric_dict, train_loss):
        if self.eval_type == 'acc':
            print(
                f"End of epoch: {epoch} \t"
                f"Train acc : {metric_dict['train_acc']:.2f}%"
                f" | Val acc : {metric_dict['val_acc']:.2f}%",
                flush=True
            )

        elif self.eval_type == 'loss':
            print(
                f"End of epoch: {epoch} \t"
                f"Train loss : {metric_dict['train_loss']:.4f}"
                f" | Val loss : {metric_dict['val_loss']:.4f}",
                flush=True
            )

        elif self.eval_type == 'loss_test':
            print(
                f"End of epoch: {epoch} \t"
                f"Train loss : {train_loss:.4f}"
                f" | Val loss : {metric_dict['val_loss']:.4f}",
                flush=True
            )



    def end_of_training(self, dataset):
        # If one of the metrics lists is empty, set it to None
        self.train_loss = None if self.train_loss == [] else self.train_loss
        self.val_loss_each_epoch = None if self.val_loss_each_epoch == [] else self.val_loss_each_epoch
        self.val_var_each_epoch = None if self.val_var_each_epoch == [] else self.val_var_each_epoch
        self.val_ent_each_epoch = None if self.val_ent_each_epoch == [] else self.val_ent_each_epoch

        self.train_acc = None if self.train_acc == [] else self.train_acc
        self.val_acc = None if self.val_acc == [] else self.val_acc
        self.test_acc = None if self.test_acc == [] else self.test_acc
        self.end_of_epoch_step_num = None if self.end_of_epoch_step_num == [] else self.end_of_epoch_step_num

        # Take labels statistics
        if self.what_is_trained in ['merge_layer', 'vit_distillation'] and len(self.models_to_merge) > 1:
            data_loader = get_dataloader(dataset, is_train=True, args=self.args, image_encoder=None)
            labels = None
            for batch in data_loader:
                labels = batch['labels'].to('cuda:0')
                break

            # Checking the scales of each task features
            labels_split = torch.chunk(labels, len(self.models_to_merge), dim=-1)

            for i, labels_curr in enumerate(labels_split):
                self.features_scale[self.models_to_merge[i]] = self.abs_mean(labels_curr)


    def abs_mean(self, tensor: torch.Tensor):
        return torch.mean(torch.abs(tensor))



    def train_model(self, model, epochs, dataset=None, print_outputs=False, with_all_plots=False, check_loss_no_train=False):
        """
        Train the given model on the given dataset for the given number of epochs.

        :param model: The model to train.
        :param epochs: The number of epochs to train the model for.
        :param dataset: The dataset to train the model on.
        :param print_outputs: Whether to print the outputs of the model during training.
        :param with_all_plots: Whether to plot all the plots during training.
        :param check_loss_no_train: Whether to check the loss of the model on the val set without training it.
        """

        model = self.create_distributed_model(model)

        if dataset is None: # Fetching the dataset if it isn't given
            preprocess_fn = model.module.train_preprocess
            dataset = get_dataset(
                self.dataset_name,
                preprocess_fn,
                location=self.args.data_location,
                batch_size=self.args.batch_size
            )
            batch_size = self.args.batch_size
        else:
            batch_size = dataset.batch_size

        num_batches = len(dataset.train_loader)
        print(f'There are {num_batches} batches in the training set.')

        #model = self.move_model_to_cuda(model) # Edan: I moved it to before get_scheduler
        model = model.cuda()
        print("lr: {} | epochs: {}".format(self.args.lr, epochs))
        optimizer, params = self.get_optimizer(model=model, lr=self.args.lr, lr_diag=self.args.lr_diag, wd=self.args.wd)
        scheduler = self.get_scheduler(optimizer, num_batches, epochs)

        step = 0
        if self.with_eval:
            metric_dict = self.perform_eval(model, dataset, step, with_all_plots)
            self.start_of_train_prints(metric_dict)
            if check_loss_no_train:
                self.end_of_training(dataset)
                return model

        # Training loop
        for epoch in range(epochs):
            model.train()
            self.lr_list.append(optimizer.param_groups[0]['lr'])
            data_loader = get_dataloader(dataset, is_train=True, args=self.args, image_encoder=None)
            train_loss = 0

            for i, batch in enumerate(data_loader):
                step += 1

                optimizer.zero_grad()
                inner_target_scales = getattr(dataset, "inner_target_scales", None)
                loss, loss_inner_dict = self.compute_loss(model, batch, inner_target_scales=inner_target_scales)
                loss.backward()

                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)

                optimizer.step()
                self.perform_scheduler_step(scheduler, step)

                # Save this step loss
                self.train_loss.append(loss.item())
                train_loss += loss.item()
                if loss_inner_dict is not None:
                    for key in loss_inner_dict:
                        # Add the inner loss to the list
                        if "train-inner-{}".format(key) not in self.train_inner_loss:
                            self.train_inner_loss["train-inner-{}".format(key)] = []
                        self.train_inner_loss["train-inner-{}".format(key)].append(loss_inner_dict[key])

            train_loss /= len(data_loader)
            self.perform_scheduler_step(scheduler, step, end_of_epoch=True)

            if self.with_eval and epoch % self.epoch_per_eval == 0:
                metric_dict = self.perform_eval(model, dataset, step-1, with_all_plots)
                if epoch % self.epoch_per_print == 0:
                    self.end_of_epoch_prints(epoch, metric_dict, train_loss)
            elif epoch % self.epoch_per_print == 0:
                print(f"Epoch: {epoch} \t Train loss: {train_loss:.4f} \t LR: {optimizer.param_groups[0]['lr']:.8f}", flush=True)

            if self.with_early_stopping and epoch % 8 == 0:
                early_stopping = self.perform_early_stopping(model, dataset)
                if early_stopping:
                    print(f"Early stopping at epoch {epoch} out of {epochs}")
                    break

        self.end_of_training(dataset)

        return model



