from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import open_clip
from transformers.modeling_outputs import SequenceClassifierOutput

import utils

class ArgsWrapper:
    def __init__(self, model_type):
        self.model = model_type
        self.openclip_cachedir = None
        self.cache_dir = None


class ImageEncoder(torch.nn.Module):
    def __init__(self, args, pretrained, keep_lang=False, random_init=False):
        super().__init__()

        print(f'Loading {args.model} pre-trained weights.')
        if '__pretrained__' in args.model:
            name, _ = args.model.split('__pretrained__')
        else:
            name = args.model

        self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained)

        if random_init:
            self.init_params()
        
        self.cache_dir = args.cache_dir

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')
        if keep_lang and hasattr(self.model, 'transformer'):
            for param in self.model.transformer.parameters():
                param.requires_grad = False


    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)
    
    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, model_name, filename):
        print(f'Loading image encoder from {filename}')
        state_dict = torch.load(filename)
        return cls.load(model_name, state_dict)

    @classmethod
    def load_from_state_dict(cls, model_name, state_dict):
        self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, cache_dir=args.openclip_cachedir)
        self.model.load_from_state_dict(state_dict)
        

    def init_params(self):
        # Before the transformer
        nn.init.normal_(self.model.visual.conv1.weight, std=0.02)
        nn.init.normal_(self.model.visual.class_embedding, std=0.02)
        nn.init.normal_(self.model.visual.positional_embedding, std=0.01)
        self.model.visual.ln_pre = nn.LayerNorm(self.model.visual.ln_pre.normalized_shape)

        # Inside the transformer
        proj_std = (self.model.visual.transformer.width ** -0.5) * ((2 * self.model.visual.transformer.layers) ** -0.5)
        attn_std = self.model.visual.transformer.width ** -0.5
        fc_std = (2 * self.model.visual.transformer.width) ** -0.5
        for block in self.model.visual.transformer.resblocks:
            block.ln_1 = nn.LayerNorm(block.ln_1.normalized_shape)
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.in_proj_bias, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.attn.out_proj.bias, std=proj_std)

            block.ln_2 = nn.LayerNorm(block.ln_2.normalized_shape)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_fc.bias, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_proj.bias, std=proj_std)

        # After the transformer
        self.model.visual.ln_post = nn.LayerNorm(self.model.visual.ln_post.normalized_shape)
        nn.init.normal_(self.model.visual.proj, std=0.01)


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        # In case the inputs are a merge of few models, U_output is the matrix that reconstruct the original inputs
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
            self.weight.requires_grad_(False)
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))
        self.bias.requires_grad_(False)

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        return utils.torch_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head, U_output=None):
        super().__init__()
        self.image_encoder = image_encoder
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

        self.classification_head = classification_head
        if classification_head is not None:
            self.classification_head.weight.requires_grad_(False)
            self.classification_head.bias.requires_grad_(False)

        if U_output is None:
            self.U_output = None
        else:
            self.U_output = nn.Parameter(U_output)
            self.U_output.requires_grad_(False)


    def forward(self, inputs):
        features = self.image_encoder(inputs)
        if self.U_output is not None:
            features = features @ self.U_output
        outputs = self.classification_head(features)
        return outputs

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)

    def unfreeze_head(self):
        self.classification_head.weight.requires_grad_(True)
        self.classification_head.bias.requires_grad_(True)

    def freeze_image_encoder(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def freeze_U(self):
        if self.U_output is not None:
            self.U_output.requires_grad_(False)


class TextClassifier(torch.nn.Module):
    def __init__(self, text_encoder, classification_head, config, num_labels, U_output=None):
        super().__init__()
        self.config = config
        self.text_encoder = text_encoder
        self.num_labels = num_labels

        classifier_dropout = (
            config['classifier_dropout'] if config['classifier_dropout'] is not None else config['hidden_dropout_prob']
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = classification_head

        if U_output is None:
            self.U_output = None
        else:
            self.U_output = nn.Parameter(U_output)
            self.U_output.requires_grad_(False)

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
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        outputs = self.text_encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        if self.U_output is not None:
            pooled_output = pooled_output @ self.U_output
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config['problem_type'] is None:
                if self.num_labels == 1:
                    self.config['problem_type'] = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config['problem_type'] = "single_label_classification"
                else:
                    self.config['problem_type'] = "multi_label_classification"

            if self.config['problem_type'] == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config['problem_type'] == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config['problem_type'] == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class MultiHeadImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_heads, U_outputs=None):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_heads = torch.nn.ModuleList(classification_heads)
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

        if U_outputs is None:
            self.U_outputs = None
        else:
            self.U_outputs = torch.nn.ParameterList(U_outputs)
            self.U_outputs.requires_grad_(False)

    def freeze_head(self):
        for idx in range(len(self.classification_heads)):
            self.classification_heads[idx].weight.requires_grad_(False)
            self.classification_heads[idx].bias.requires_grad_(False)

    def forward_id(self, inputs, head_idx):
        features = self.image_encoder(inputs)
        if self.U_outputs is not None:
            features = features @ self.U_outputs[head_idx]
        outputs = self.classification_heads[head_idx](features)
        return outputs

    def forward(self, inputs):
        features = self.image_encoder(inputs)
        outputs = []
        for head_idx in range(len(self.classification_heads)):
            if self.U_outputs is not None:
                features_u = features @ self.U_outputs[head_idx]
                outputs.append(self.classification_heads[head_idx](features_u))
            else:
                outputs.append(self.classification_heads[head_idx](features))

        return torch.cat(outputs, dim=-1)

    def get_head_place(self, head_idx):
        place = 0
        for idx in range(head_idx):
            place += self.classification_heads[idx].weight.shape[0]
        return place

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)


class CLIPSharedTransformer(torch.nn.Module):
    """
    CLIP model with a single transformer shared between text and image encoders.
    """
    def __init__(self, transformer, visual_weights, text_weights):
        super().__init__()

        self.transformer = transformer

        # Visual weights
        self.d, c, patch_size, _  = visual_weights['conv'].shape # d, 3, patch_size, patch_size
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.d, kernel_size=patch_size, stride=patch_size, bias=False)
        self.conv1.weight = torch.nn.Parameter(visual_weights['conv'])

        self.class_embedding = nn.Parameter(visual_weights['class_embedding']) # d

        self.visual_positional_embedding = nn.Parameter(visual_weights['pos_embedding']) # T, d

        self.visual_ln_pre = torch.nn.LayerNorm(visual_weights['ln_pre']['weight'].shape[0]) # d
        self.visual_ln_pre.load_state_dict(visual_weights['ln_pre'])

        self.visual_ln_post = torch.nn.LayerNorm(visual_weights['ln_post']['weight'].shape[0])  # d
        self.visual_ln_post.load_state_dict(visual_weights['ln_post'])

        self.visual_proj = nn.Parameter(visual_weights['proj']) # d, out_dim

        # Text weights
        self.context_length_text, self.d_text = text_weights['pos_embedding'].shape
        pos_embedding = text_weights['pos_embedding']
        pos_embedding = torch.nn.functional.pad(pos_embedding, (0, self.d - self.d_text), "constant", 0)  # make sure the embedding is the same size as the visual transformer width
        self.text_positional_embedding = nn.Parameter(pos_embedding) # T_text, d_text

        token_embedding = text_weights['token_embedding']  # dict_size, d_text
        token_embedding = torch.nn.functional.pad(token_embedding, (0, self.d - self.d_text), "constant", 0) # make sure the embedding is the same size as the visual transformer width
        self.token_embedding = nn.Embedding(token_embedding.size(0), self.d)
        self.token_embedding.weight = torch.nn.Parameter(token_embedding, requires_grad=False)

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.text_ln_final = torch.nn.LayerNorm(text_weights['ln_final']['weight'].shape[0])
        self.text_ln_final.load_state_dict(text_weights['ln_final'])

        self.text_projection = nn.Parameter(text_weights['text_projection']) # d_text, out_dim


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length_text, self.context_length_text)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def freeze_non_transformer(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.transformer.parameters():
            param.requires_grad = True

    def forward(self, inputs, modality):
        if modality == 'image':
            return self.encode_image(inputs)
        elif modality == 'text':
            return self.encode_text(inputs)
        else:
            raise ValueError(f'Unknown modality: {modality}')

    def encode_image(self, images):
        x = self.conv1(images)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                    device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual_positional_embedding.to(x.dtype)
        x = self.visual_ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.visual_ln_post(x[:, 0, :])

        if self.visual_proj is not None:
            x = x @ self.visual_proj

        return x

    def encode_text(self, text):
        y = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        y = y + self.text_positional_embedding
        y = y.permute(1, 0, 2)  # NLD -> LND
        y = self.transformer(y, attn_mask=self.attn_mask)
        y = y[:,:,:self.d_text]
        y = y.permute(1, 0, 2)  # LND -> NLD
        y = self.text_ln_final(y)
        # y.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        y = y[torch.arange(y.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return y

class ModuleWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model

    def forward(self, inputs, **kwargs):
        return self.module(inputs, **kwargs)
