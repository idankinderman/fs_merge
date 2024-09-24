import torch
from vision_datasets.common import maybe_dictionarize
from utils import batch_index_slicing, batch_seq_index_slicing, scale_and_multiply


def get_loss_wrapper(loss_type, loss_fn, num_models, what_is_trained):
    print("what_is_trained:", what_is_trained, "loss_type:", loss_type)
    if what_is_trained == 'merge_layer':
        return get_loss_wrapper_merger(loss_type, loss_fn, num_models)
    elif what_is_trained == 'bert_merge_layer':
        print("\n!!!!! Using bert merger loss!!!!!  \n")
        return get_loss_wrapper_bert_merger(loss_type, loss_fn, num_models)
    elif what_is_trained == 'bert_distillation':
        return get_loss_wrapper_bert_distillation(loss_fn)
    else:
        return get_loss_wrapper_general(loss_type, loss_fn)


def get_loss_wrapper_merger(loss_type, loss_fn, num_models):
    if loss_type == "rec_with_merged_features":
        def loss_wrapper(model, batch, return_output=False, inner_target_scales=None):
            inputs = batch['images'].cuda()
            labels = batch['labels'].cuda()
            merged_features = batch['merged_features'].to('cuda:0')
            logits_full, logits_merged = model(inputs, two_outputs=True)
            loss = loss_fn(full_output=logits_full, full_target=labels,
                                merged_output=logits_merged, merged_target=merged_features)

            if not return_output:
                return loss, None
            return loss, logits_full, labels, None


    elif loss_type == "rec_with_ids":
        # If the images are from task X, compute the loss only according to the features of task X.
        #print("Will use mixed precision for the loss!!!")
        def loss_wrapper(model, batch, return_output=False, inner_target_scales=None):
            batch = maybe_dictionarize(batch)
            inputs = batch['images'].cuda()
            labels = batch['labels'].cuda()
            ids = batch['ids'].cuda()
            logits = model(inputs)
            # Selecting the relevant logits and labels
            selected_logits = batch_index_slicing(logits, ids=ids, num_models=num_models)
            loss = loss_fn(selected_logits, labels)

            if not return_output:
                return loss, None
            return loss, selected_logits, labels, None

    elif loss_type in ['rec_with_inner_att', 'rec_with_inner_mlp', 'rec_with_inner_att_ids', 'rec_with_inner_mlp_ids']:
        # The loss also takes into account the inner features

        def loss_wrapper(model, batch, return_output=False, inner_target_scales=None):
            batch = maybe_dictionarize(batch)
            for key in batch:
                batch[key] = batch[key].cuda()

            # Forward pass
            logits, inner_features = model(batch['images'])

            # Selecting the relevant labels and inner features, according to the ids
            selected_logits = batch_index_slicing(logits, ids=batch['ids'], num_models=num_models)

            for key in inner_features:
                inner_features[key] = batch_seq_index_slicing(inner_features[key], ids=batch['ids'],
                                                              num_models=num_models)

            loss, loss_dict = loss_fn(logits=selected_logits, inner_features=inner_features, batch=batch,
                                           inner_target_scales=inner_target_scales)

            if not return_output:
                return loss, loss_dict
            return loss, logits, batch['labels'], loss_dict

    else:
        def loss_wrapper(model, batch, return_output=False, inner_target_scales=None):
            batch = maybe_dictionarize(batch)
            inputs = batch['images'].cuda()
            labels = batch['labels'].cuda()
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            if not return_output:
                return loss, None
            return loss, logits, labels, None

    return loss_wrapper

def get_loss_wrapper_bert_merger(loss_type, loss_fn, num_models):
    def loss_wrapper(model, batch, return_output=False, inner_target_scales=None):
        batch = maybe_dictionarize(batch)
        input_ids = batch['input_ids'].cuda()
        targets = batch['targets'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        token_type_ids = batch['token_type_ids'].cuda()
        ids = batch['ids'].cuda()

        output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        #emb = targets.shape[1] // num_models

        # Selecting the relevant logits and labels
        selected_output = batch_index_slicing(output, ids=ids, num_models=num_models)

        loss = loss_fn(selected_output, targets)
        if not return_output:
            return loss, None
        return loss, selected_output, targets, None

    return loss_wrapper

def get_loss_wrapper_general(loss_type, loss_fn):
    if loss_type in ['rec_with_inner_att', 'rec_with_inner_mlp', 'rec_with_inner_att_ids', 'rec_with_inner_mlp_ids']:
        print("Using inner features loss for dist")
        def loss_wrapper(model, batch, return_output=False, inner_target_scales=None):
            batch = maybe_dictionarize(batch)
            for key in batch:
                batch[key] = batch[key].cuda()

            # Forward pass
            logits, inner_features = model(batch['images'])
            loss, loss_dict = loss_fn(logits=logits, inner_features=inner_features, batch=batch)

            if not return_output:
                return loss, loss_dict
            return loss, logits, batch['labels'], loss_dict


    else:
        def loss_wrapper(model, batch, return_output=False, inner_target_scales=None):
            batch = maybe_dictionarize(batch)
            inputs = batch['images'].cuda()
            labels = batch['labels'].cuda()
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            if not return_output:
                return loss, None
            return loss, logits, labels, None

    return loss_wrapper

def get_loss_wrapper_bert_distillation(loss_fn):
    def loss_wrapper(model, batch, return_output=False, inner_target_scales=None):
        batch = maybe_dictionarize(batch)
        input_ids = batch['input_ids'].cuda()
        targets = batch['targets'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        token_type_ids = batch['token_type_ids'].cuda()

        output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[1]

        loss = loss_fn(output, targets)
        if not return_output:
            return loss, None
        return loss, output, targets, None

    return loss_wrapper

