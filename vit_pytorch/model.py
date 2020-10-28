"""model.py - Model and module class for ViT.
   They are built to mirror those in the official Jax implementation.
"""

import torch
from torch import nn
from torch.nn import functional as F
from .utils import load_pretrained_weights, as_tuple

from .configs import CONFIGS, WEIGHTS


class PositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__() 
        assert embedding_type == '1d', 'embedding type not implemented'
        self.pos_embedding = nn.Parameter(1, seq_len, dim)
    
    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.pos_embedding


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""
    def __init__(self, dim, mlp_dim, out_dim, dropout_rate):
        super().__init__()
        self
    
    def apply(self,
            inputs,
            mlp_dim,
            dtype=jnp.float32,
            out_dim=None,
            dropout_rate=0.1,
            deterministic=True,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6)):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if out_dim is None else out_dim
    return output


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
        self, 
        name: str = None, 
        pretrained: bool = False, 
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.1,
        representation_size: int = None,
        classifier: str = 'token',
        positional_embedding: str = '1d',
        image_size: int = 384,
        in_channels: int = 3, 
        num_classes: int = 1000,
    ):
        super().__init__()
        assert name or not pretrained, 'specify name of pretrained model'

        # Get pretrained config
        if name is not None:
            assert name in WEIGHTS.keys()), \
                'name should be in: ' + ', '.join(WEIGHTS.keys())
            config_name = model_name[:4]
            config = CONFIGS[config_name]
            dim = config['dim']
            ff_dim = config['ff_dim']
            num_heads = config['num_heads']
            num_layers = config['num_layers']
            attention_dropout_rate = config['attention_dropout_rate']
            dropout_rate = config['dropout_rate']
            representation_size = config['representation_size']
            classifier = config['classifier']

        # Image and patch sizes
        h, w = as_tuple(image_size)  # image sizes
        fh, fw = as_tuple(patches)  # patch sizes
        gh, gw = self.h // self.fh, self.w // self.fw  # number of patches
        seq_len = gh * gw

        # Patch embedding
        self.patch_embedding = nn.Conv2d(3, dim, kernel_size=(fh, fw), stride=(fh, fw))

        # Class token
        if classifier == 'token':
            self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
            seq_len += 1
        
        # Positional embedding
        self.positional_embedding = AddPositionEmbs(
            seq_len, dim, embedding_type=positional_embedding)
        
        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=num_heads, dim_feedforward=ff_dim, 
                dropout=dropout_rate, activation='gelu'), 
            num_layers=num_layers
        )
        
        # Classifier head
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.fc = nn.Linear(dim, 1000)
        
        # Load pretrained model
        if pretrained:
            load_pretrained_weights(model_name, num_classes=num_classes)
        
        # Modify model as specified. NOTE: We do not do this earlier because 
        # it's easier to load only part of a pretrained model in this manner.
        if in_channels != 3:
            self.embedding = nn.Conv2d(in_channels, patches, kernel_size=patches, stride=patches)
        if num_classes != 1000:
            self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        """Breaks image into patches, applies transformer, applies MLP head.

        Args:
            x (tensor): `b,c,fh,fw`
        """
        b, c, fh, fw = x.shape
        x = self.patch_embedding(x)  # b,d,gh,gw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,c
        if hasattr(self, 'class_token'):
            x = torch.cat((self.class_token.expand(b, -1, -1)))  # b,gh*gw,c
        if hasattr(self, 'position_embedding'): 
            x = self.positional_embedding(x)  # b,gh*gw,c 
        x = self.transformer(x)  # b,gh*gw,c
        if hasattr(self, fc):
            x = self.norm(x)[:, 0]
            x = self.fc(x)
        return x

