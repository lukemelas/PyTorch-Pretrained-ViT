"""configs.py - ViT model configurations, based on:
https://github.com/google-research/vision_transformer/blob/master/vit_jax/configs.py
# HuggingFace's Transformers ViTConfig as baseline for architecture configuration
# https://huggingface.co/transformers/_modules/transformers/models/vit/configuration_vit.html#ViTConfig
max_text_seq_len comes from experiments with daf:re where the median number of tokens per image is 32
vocab_size comes from bert's vocab size: 30,522
"""
from transformers import ViTConfig

def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)

class ViTConfigExtended(ViTConfig):
    def __init__(self,
        pos_embedding_type: str = 'learned',
        classifier: str = 'token',
        num_classes: int = 21843,
        representation_size: int = 768,
        pretrained_image_size: int = 224,  
        pretrained_num_channels: int = 3,
        pretrained_num_classes: int = 21843,
        max_text_seq_len: int = 32,
        vocab_size: int= 30522, 
        **kwargs):
        super().__init__(**kwargs)
        self.pos_embedding_type = pos_embedding_type
        self.classifier = classifier
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.pretrained_image_size = pretrained_image_size
        self.pretrained_num_channels = pretrained_num_channels
        self.pretrained_num_classes = pretrained_num_classes
        self.max_text_seq_len = max_text_seq_len
        self.vocab_size = vocab_size

    def calc_pre_dims(self):

        h, w = as_tuple(self.image_size)  # image sizes
        self.fh, self.fw = as_tuple(self.patch_size)  # patch sizes
        self.gh, self.gw = h // self.fh, w // self.fw  # number of patches
        self.seq_len = self.gh * self.gw # sequence length

def get_base_config():
    """Base ViT config ViT"""
    return dict(
      hidden_size=768,
      intermediate_size=3072,
      num_attention_heads=12,
      num_hidden_layers=12,
      attention_probs_dropout_prob=0.0,
      hidden_dropout_prob=0.1,
      representation_size=768,
      classifier='token',
      pretrained_num_classes=21843,
      pretrained_image_size=224,
      image_size=224,
      pretrained_num_channels=3,
      pos_embedding_type='learned',
      hidden_act="gelu",
      layer_norm_eps=1e-12,
    )

def get_sb16_config():
    """Returns the ViT-B/16 configuration."""
    config = get_base_config()
    config.update(dict(
        patch_size=(16, 16),
        num_hidden_layers=3
    ))
    return config

def get_ti16_config():
    """Returns the ViT-S/16 configuration."""
    config = get_base_config()
    config.update(dict(
        patch_size=(16, 16),
        hidden_size=192,
        intermediate_size=768,
        num_attention_heads=3,
        representation_size=192
    ))
    return config

def get_s16_config():
    """Returns the ViT-L/16 configuration."""
    config = get_base_config()
    config.update(dict(
        patch_size=(16, 16),
        hidden_size=384,
        intermediate_size=1536,
        num_attention_heads=6,
        representation_size=384
    ))
    return config

def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = get_base_config()
    config.update(dict(patch_size=(16, 16)))
    return config

def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.update(dict(patch_size=(32, 32)))
    return config

def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = get_base_config()
    config.update(dict(
        patch_size=(16, 16),
        hidden_size=1024,
        intermediate_size=4096,
        num_attention_heads=16,
        num_hidden_layers=24,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.1,
        representation_size=1024
    ))
    return config

def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.update(dict(patch_size=(32, 32)))
    return config

def get_h14_config():
    """Returns the ViT-H/14 configuration."""
    config = get_base_config()
    config.update(dict(
        patch_size=(14, 14),
        hidden_size=1280,
        intermediate_size=5120,
        num_attention_heads=16,
        num_hidden_layers=32,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.1,
        representation_size=1280
    ))
    return config

def drop_head_variant(config):
    config.update(dict(representation_size=None))
    return config

PRETRAINED_CONFIGS = {
    'sB_16': {
      'config': get_sb16_config(),
      'url': None,
      'url_og': None
    },
    'Ti_16': {
      'config': get_ti16_config(),
      'url': None,
      'url_og': None
    },
    'S_16': {
      'config': get_s16_config(),
      'url': None,
      'url_og': None
    },
    'B_16': {
      'config': get_b16_config(),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16.pth",
      'url_og': "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz"
    },
    'B_32': {
      'config': get_b32_config(),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_32.pth",
      'url_og': "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_32.npz"
    },
    'L_16': {
      'config': get_l16_config(),
      'url': None,
      'url_og': "https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz"
    },
    'L_32': {
      'config': get_l32_config(),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_32.pth",
      'url_og': "https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_32.npz"
    },
    'H_14': {
      'config': get_h14_config(),
      'url': None,
      'url_og': "https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz"
    }
}

'''
# For reference
# from transformers
https://huggingface.co/transformers/_modules/transformers/models/vit/configuration_vit.html#ViTConfig
class ViTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.ViTModel`. It is used to
    instantiate an ViT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the ViT `google/vit-base-patch16-224
    <https://huggingface.co/google/vit-base-patch16-224>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        image_size (:obj:`int`, `optional`, defaults to :obj:`224`):
            The size (resolution) of each image.
        patch_size (:obj:`int`, `optional`, defaults to :obj:`16`):
            The size (resolution) of each patch.
        num_channels (:obj:`int`, `optional`, defaults to :obj:`3`):
            The number of input channels.


    Example::

        >>> from transformers import ViTModel, ViTConfig

        >>> # Initializing a ViT vit-base-patch16-224 style configuration
        >>> configuration = ViTConfig()

        >>> # Initializing a model from the vit-base-patch16-224 style configuration
        >>> model = ViTModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "vit"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_encoder_decoder=False,
        image_size=224,
        patch_size=16,
        num_channels=3,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
'''