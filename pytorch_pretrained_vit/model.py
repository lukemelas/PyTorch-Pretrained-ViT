"""model.py - Model and module class for ViT.
   They are built to mirror those in the official Jax implementation.
"""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings

import einops

from .transformer import Transformer
from .utils import load_pretrained_weights, as_tuple, maybe_print, resize_positional_embedding_
from .configs import PRETRAINED_CONFIGS

class LearnedPositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))
    
    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.pos_embedding


class ViT(nn.Module):
    """
    Args:
        name (str): Model name, e.g. 'B_16'
        pretrained (bool): Load pretrained weights
        in_channels (int): Number of channels in input data
        num_classes (int): Number of classes, default 1000

    References:
        [1] https://openreview.net/forum?id=YicbFdNTTy
    """

    def __init__(
        self, config,
        name: str = None, 
        pretrained: bool = False,
        load_fc_layer: bool = True,
        load_repr_layer: bool = False,
        ret_attn_scores: bool = False,
        conv_patching: bool = False,
        ret_image_patchified: bool = False,
        ret_interm_repr: bool = False,
        multimodal: bool = False,
    ):
        super().__init__()
        config.calc_pre_dims()
        self.config = config
        self.ret_attn_scores = ret_attn_scores
        self.ret_image_patchified = ret_image_patchified
        #self.config = deepcopy(config)
        
        # Patch embedding
        if conv_patching == False:
            self.patch_embedding = nn.Conv2d(
                in_channels=self.config.num_channels, out_channels=self.config.hidden_size, 
                kernel_size=(self.config.fh, self.config.fw), stride=(self.config.fh, self.config.fw))
        else:
            self.patch_embedding = ConvPatchingStem(config)

        # Class token
        if self.config.classifier == 'token':
            self.class_token = nn.Parameter(torch.zeros(1, 1, self.config.hidden_size))
            self.config.seq_len += 1
            
        # Positional embedding
        if self.config.pos_embedding_type == 'learned':
            self.positional_embedding = LearnedPositionalEmbedding1D(self.config.seq_len, self.config.hidden_size)
        
        # Text embedding
        if multimodal:
            bert_config = BertConfig(
                # vocab_size is default for bert_base 30,522
                vocab_size=self.config.vocab_size,
                hidden_size=self.config.hidden_size,
                max_position_embeddings=self.config.max_text_seq_len,
                layer_norm_eps=self.config.layer_norm_eps,
                hidden_dropout_prob=self.config.hidden_dropout_prob
            )
            self.text_embeddings = BertEmbeddings(bert_config)
            self.config.seq_len += self.config.max_text_seq_len

        # Transformer
        self.transformer = Transformer(num_layers=self.config.num_hidden_layers, dim=self.config.hidden_size, 
            num_heads=self.config.num_attention_heads, ff_dim=self.config.intermediate_size, 
            hidden_dropout_prob=self.config.hidden_dropout_prob, attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
            layer_norm_eps=self.config.layer_norm_eps, ret_attn_scores=ret_attn_scores, ret_interm_repr=ret_interm_repr)
        
        # Representation layer
        if self.config.representation_size and load_repr_layer:
            self.pre_logits = nn.Linear(self.config.hidden_size, self.config.representation_size)
            pre_logits_size = self.config.representation_size
        else:
            pre_logits_size = self.config.hidden_size
        
        # Classifier head
        if load_fc_layer:
            self.norm = nn.LayerNorm(pre_logits_size, eps=self.config.layer_norm_eps)
            self.fc = nn.Linear(pre_logits_size, self.config.num_classes)

        # Initialize weights
        self.init_weights()
        
        # Load pretrained model
        if pretrained:
            assert name in PRETRAINED_CONFIGS.keys(), \
                'name should be in: ' + ', '.join(PRETRAINED_CONFIGS.keys())
            
            pretrained_num_channels = self.config.pretrained_num_channels #PRETRAINED_CONFIGS[name]['config']['pretrained_num_channels']
            pretrained_num_classes = self.config.pretrained_num_classes #PRETRAINED_CONFIGS[name]['config']['pretrained_num_classes']
            pretrained_image_size = self.config.pretrained_image_size #PRETRAINED_CONFIGS[name]['config']['pretrained_image_size']
            
            load_pretrained_weights(
                self, name, 
                load_first_conv=(self.config.num_channels == pretrained_num_channels),
                load_fc=(self.config.num_classes == pretrained_num_classes),
                load_repr_layer=load_repr_layer,
                resize_positional_embedding=(self.config.image_size != pretrained_image_size),
            )
        
        self.ret_interm_repr = ret_interm_repr
        
    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
        self.apply(_init)
        if hasattr(self, 'fc'):
            nn.init.constant_(self.fc.weight, 0)
            nn.init.constant_(self.fc.bias, 0)
        nn.init.normal_(self.positional_embedding.pos_embedding, std=0.02)  # _trunc_normal(self.positional_embedding.pos_embedding, std=0.02)
        nn.init.constant_(self.class_token, 0)

    def forward(self, images, text=None, mask=None):
        """Breaks image into patches, applies transformer, applies MLP head.
        Args:
            images (tensor): `b,c,fh,fw`
            text (tensor): b, max_text_seq_len
            mask (bool tensor): (B(batch_size) x S(seq_len))
        """
        b, c, fh, fw = images.shape
        images = self.patch_embedding(images)  # b,d,gh,gw
        image_patchified = images.flatten(2).transpose(1, 2)  # b,gh*gw,d
        #image_patchified = einops.rearrange(x, 'b d gh gw -> b (gh gw) d')
        
        if hasattr(self, 'class_token'):
            x = torch.cat((self.class_token.expand(b, -1, -1), image_patchified), dim=1)  # b,gh*gw+1,d
        if hasattr(self, 'positional_embedding'): 
            x = self.positional_embedding(x)  # b,gh*gw+1,d

        # concatenate text to images
        if hasattr(self, 'text_embeddings'):
            text = self.text_embeddings(text) #b, max_text_seq_len > b, max_text_seq_len, d
            x = torch.cat((x, text), dim=1) #b, gh*gw+1+max_text_seq_len,d
        
        if self.ret_interm_repr and self.ret_attn_scores:
            x, interm_repr, scores = self.transformer(x, mask)
        elif self.ret_interm_repr:
            x, interm_repr = self.transformer(x, mask)
        elif self.ret_attn_scores:
            x, scores = self.transformer(x, mask)  # b,gh*gw+1,d
        else:
            x = self.transformer(x, mask)
        
        if hasattr(self, 'pre_logits'):
            x = self.pre_logits(x) # b,d
            x = torch.tanh(x) # b,d
        
        if hasattr(self, 'fc'):
            x = self.norm(x)[:, 0]  # b,d
            x = self.fc(x)  # b,num_classes
        
        if self.ret_image_patchified and self.ret_interm_repr and self.ret_attn_scores:
            return x, interm_repr, scores, image_patchified
        
        elif self.ret_interm_repr and self.ret_attn_scores:
            return x, interm_repr, scores
        elif self.ret_interm_repr and self.ret_image_patchified:
            return x, interm_repr, image_patchified
        elif self.ret_image_patchified and self.ret_attn_scores:
            return x, scores, image_patchified
        
        elif self.ret_interm_repr:
            return x, interm_repr
        elif self.ret_image_patchified:
            return x, image_patchified
        elif self.ret_attn_scores:
            return x, scores
        
        else:
            return x

    def load_partial(self, weights_path, pretrained_image_size, pretrained_mode, verbose=True):
            
            state_dict = torch.load(weights_path)
            
            if pretrained_mode == 'full_tokenizer':
                components_list = ['patch_embedding.weight', 'patch_embedding.bias', 
                                            'positional_embedding.pos_embedding', 'class_token']
            elif pretrained_mode == 'patchprojection':
                components_list = ['patch_embedding.weight', 'patch_embedding.bias']
            elif pretrained_mode == 'posembeddings':
                components_list = ['positional_embedding.pos_embedding']
            elif pretrained_mode == 'clstoken':
                components_list = ['class_token']
            elif pretrained_mode == 'patchandposembeddings':
                components_list = ['patch_embedding.weight', 'patch_embedding.bias', 
                                            'positional_embedding.pos_embedding']   
            elif pretrained_mode == 'patchandclstoken':
                components_list = ['patch_embedding.weight', 'patch_embedding.bias', 
                                            'class_token']    
            elif pretrained_mode == 'posembeddingsandclstoken':
                components_list = ['positional_embedding.pos_embedding', 'class_token']
            else:
                maybe_print('Pretrained mode component not in available list. No pretrained weights loaded.', verbose)
                return None

            not_load_keys = []
            for param_tensor in state_dict:
                if (param_tensor not in components_list):
                    not_load_keys.append(param_tensor)
            for not_keys in not_load_keys:
                state_dict.pop(not_keys)

            # Change size of positional embeddings
            if ('positional_embedding.pos_embedding' in components_list):
                if (self.config.image_size != pretrained_image_size): 
                    posemb = state_dict['positional_embedding.pos_embedding']
                    posemb_new = self.state_dict()['positional_embedding.pos_embedding']
                    state_dict['positional_embedding.pos_embedding'] = resize_positional_embedding_(
                        posemb=posemb, posemb_new=posemb_new, 
                        has_class_token=hasattr(self, 'class_token'))
                    maybe_print('Resized positional embeddings from {} to {}'.format(
                        posemb.shape, posemb_new.shape), True)
                    
            self.init_weights()
            ret = self.load_state_dict(state_dict, strict=False)
            maybe_print('Missing keys when loading pretrained weights: {}'.format(ret.missing_keys), verbose)
            maybe_print('Unexpected keys when loading pretrained weights: {}'.format(ret.unexpected_keys), verbose)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, 
    kernel_size=3, stride=2, padding=1, activation='relu', norm='batchnorm'):
        super(ConvLayer, self).__init__()

        if in_channels == out_channels:
            stride = 1
            
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
            kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=(padding, padding))
        if activation == 'relu':
            self.activation = nn.ReLU()
        if norm == 'batchnorm':
            self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x) 
        if hasattr(self, 'activation'):
            x = self.activation(x)
        if hasattr(self, 'norm'):
            x = self.norm(x)
        return x


class ConvPatchingStem(nn.Module):
    def __init__(self, config):
        super(ConvPatchingStem, self).__init__()

        channels_in_list = [config.num_channels, 64, 128, 128, 256, 256]
        channels_out_list = [64, 128, 128, 256, 256, 512]

        self.conv3x3layers = nn.ModuleList([
            ConvLayer(channels_in, channels_out)
            for (channels_in, channels_out) in zip(channels_in_list, channels_out_list)
        ])

        self.conv1x1 = nn.Conv2d(512, config.hidden_size, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        #print(x.shape)
        for layer in self.conv3x3layers: 
            x = layer(x)
            #print(x.shape)
        x = self.conv1x1(x)
        #print(x.shape)
        return x
