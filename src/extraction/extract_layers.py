import os
from pathlib import Path
import pickle

from datetime import datetime
import tqdm

import torch
import open_clip


from args import parse_arguments


class LayersExtractorVit:
    """
    Used in order to extract all the layers of a ViT model.
    """
    def __init__(self,
                 dir_path: str,
                 models_path: str = None,):

        self.dir_path = dir_path
        if models_path is not None:
            self.models_path = models_path
        else:
            self.models_path = os.path.join(self.dir_path, 'checkpoints')

        self.layers_dir = self.dir_path
        Path(self.layers_dir).mkdir(parents=True, exist_ok=True)


    # Load a specific model
    def load_model(self, model_name):
        if model_name == 'zero_shot':
            return self.load_zero_shot_model()

        model_full_name = "{}.pt".format(model_name)
        file_path = os.path.join(self.models_path, model_full_name)
        model = torch.load(file_path)
        return model

    def load_zero_shot_model(self):
        file_path = os.path.join(self.models_path, 'zero_shot.pt')
        model = torch.load(file_path)
        return model


    def save_layer(self, parameters, model_name, layer_name, layer_num=None):
        layer_dir = os.path.join(self.layers_dir, 'layers', model_name)
        Path(layer_dir).mkdir(parents=True, exist_ok=True)
        layer_path = os.path.join(layer_dir, layer_name)
        if layer_num is not None:
            layer_path += "_{}".format(layer_num)
        parameters.requires_grad = False
        pickle.dump(parameters, open(layer_path, 'wb'))


    def extract_layers(self, model_name):
        model = self.load_model(model_name)

        # Layers before and after the transformer
        self.extract_vit_special_blocks(model.model.visual, model_name)
        self.extract_transformer(model.model.visual.transformer, model_name)

    def extract_vit_special_blocks(self, visual, model_name):
        # Layers before the transformer
        parameters = visual.conv1.weight
        self.save_layer(parameters, model_name=model_name, layer_name="conv")

        parameters = visual.class_embedding
        self.save_layer(parameters, model_name=model_name, layer_name="class_embedding")

        parameters = visual.positional_embedding
        self.save_layer(parameters, model_name=model_name, layer_name="pos_embedding")

        parameters = visual.ln_pre
        self.save_layer(parameters, model_name=model_name, layer_name="ln_pre")

        parameters = visual.ln_pre.state_dict()
        self.save_layer(parameters, model_name=model_name, layer_name="ln_pre_state_dict")

        # Layers after the transformer
        parameters = visual.ln_post
        self.save_layer(parameters, model_name=model_name, layer_name="ln_post")

        parameters = visual.ln_post.state_dict()
        self.save_layer(parameters, model_name=model_name, layer_name="ln_post_state_dict")

        parameters = visual.proj
        self.save_layer(parameters, model_name=model_name, layer_name="out_proj")


    def extract_text_special_blocks(self, clip, model_name):
        parameters = clip.token_embedding.weight
        self.save_layer(parameters, model_name=model_name, layer_name="token_embedding")

        parameters = clip.positional_embedding
        self.save_layer(parameters, model_name=model_name, layer_name="pos_embedding")

        parameters = clip.ln_final
        self.save_layer(parameters, model_name=model_name, layer_name="ln_final")

        parameters = clip.ln_final.state_dict()
        self.save_layer(parameters, model_name=model_name, layer_name="ln_final_state_dict")

        parameters = clip.text_projection
        self.save_layer(parameters, model_name=model_name, layer_name="text_projection")

    def extract_transformer(self, transformer, model_name):
        num_layers = len(transformer.resblocks)

        for layer_num in range(num_layers):
            # Layers inside the transformer, MultiheadAttention sub-block
            parameters = transformer.resblocks[layer_num].ln_1
            self.save_layer(parameters, model_name=model_name, layer_name="ln_1", layer_num=layer_num)

            parameters = transformer.resblocks[layer_num].ln_1.state_dict()
            self.save_layer(parameters, model_name=model_name, layer_name="ln_1_state_dict", layer_num=layer_num)

            parameters = transformer.resblocks[layer_num].attn.in_proj_weight
            self.save_layer(parameters, model_name=model_name, layer_name="att_in_proj_weight", layer_num=layer_num)

            parameters = transformer.resblocks[layer_num].attn.in_proj_bias
            self.save_layer(parameters, model_name=model_name, layer_name="att_in_proj_bias", layer_num=layer_num)

            parameters = transformer.resblocks[layer_num].attn.out_proj.weight
            self.save_layer(parameters, model_name=model_name, layer_name="att_out_proj_weight", layer_num=layer_num)

            parameters = transformer.resblocks[layer_num].attn.out_proj.bias
            self.save_layer(parameters, model_name=model_name, layer_name="att_out_proj_bias", layer_num=layer_num)

            # Layers inside the transformer, MLP sub-block
            parameters = transformer.resblocks[layer_num].ln_2
            self.save_layer(parameters, model_name=model_name, layer_name="ln_2", layer_num=layer_num)

            parameters = transformer.resblocks[layer_num].ln_2.state_dict()
            self.save_layer(parameters, model_name=model_name, layer_name="ln_2_state_dict", layer_num=layer_num)

            parameters = transformer.resblocks[layer_num].mlp[0].weight
            self.save_layer(parameters, model_name=model_name, layer_name="fc_1_weight", layer_num=layer_num)

            parameters = transformer.resblocks[layer_num].mlp[0].bias
            self.save_layer(parameters, model_name=model_name, layer_name="fc_1_bias", layer_num=layer_num)

            parameters = transformer.resblocks[layer_num].mlp[3].weight
            self.save_layer(parameters, model_name=model_name, layer_name="fc_2_weight", layer_num=layer_num)

            parameters = transformer.resblocks[layer_num].mlp[3].bias
            self.save_layer(parameters, model_name=model_name, layer_name="fc_2_bias", layer_num=layer_num)


def extract_layers_from_model(model_type):
    """
    This used in order to extract layers from a number of VITs.
    The 'model_names' determine the models from which the features will be extracted.
    The layers are saved in the 'layers' directory, inside the 'dir_path' directory.
    """

    if model_type == 'ViT-B-16':
        exp_name = '4_1_24_diff_pretrained_finetune'
    elif model_type == 'ViT-L-14':
        exp_name = '9_3_24_diff_pretrained_finetuned'


    if model_type in ['ViT-B-16', 'ViT-L-14']:
        dir_path = os.path.join('..', 'experiments', model_type, exp_name)
        model_names = ['finetuned_Cars']

        layers_extractor = LayersExtractorVit(dir_path)

        for model_name in tqdm.tqdm(model_names, desc='Extracting layers'):
            print("\nExtracting layers from dataset {}".format(model_name))
            layers_extractor.extract_layers(model_name=model_name)

if __name__ == '__main__':
    extract_layers_from_model(model_type='ViT-B-16')

