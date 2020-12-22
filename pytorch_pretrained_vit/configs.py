"""configs.py - ViT model configurations, based on:
https://github.com/google-research/vision_transformer/blob/master/vit_jax/configs.py
"""

def get_base_config():
    """Base ViT config ViT"""
    return dict(
      dim=768,
      ff_dim=3072,
      num_heads=12,
      num_layers=12,
      attention_dropout_rate=0.0,
      dropout_rate=0.1,
      representation_size=768,
      classifier='token'
    )

def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = get_base_config()
    config.update(dict(patches=(16, 16)))
    return config

def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.update(dict(patches=(32, 32)))
    return config

def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = get_base_config()
    config.update(dict(
        patches=(16, 16),
        dim=1024,
        ff_dim=4096,
        num_heads=16,
        num_layers=24,
        attention_dropout_rate=0.0,
        dropout_rate=0.1,
        representation_size=1024
    ))
    return config

def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.update(dict(patches=(32, 32)))
    return config

def get_h14_config():
    """Returns the ViT-H/14 configuration."""
    config = get_base_config()
    config.update(dict(
        patches=(14, 14),
        dim=1280,
        ff_dim=5120,
        num_heads=16,
        num_layers=32,
        attention_dropout_rate=0.0,
        dropout_rate=0.1,
        representation_size=1280
    ))
    return config

def drop_head_variant(config):
    config.update(dict(representation_size=None))
    return config

PRETRAINED_MODELS = {
    'B_16': {
      'config': get_b16_config(),
      'num_classes': 21843,
      'image_size': (224, 224),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16.pth",
      'url_og': "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz"
    },
    'B_32': {
      'config': get_b32_config(),
      'num_classes': 21843,
      'image_size': (224, 224),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_32.pth",
      'url_og': "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_32.npz"
    },
    'L_16': {
      'config': get_l16_config(),
      'num_classes': 21843,
      'image_size': (224, 224),
      'url': None,
      'url_og': "https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz"
    },
    'L_32': {
      'config': get_l32_config(),
      'num_classes': 21843,
      'image_size': (224, 224),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_32.pth",
      'url_og': "https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_32.npz"
    },
    'H_14': {
      'config': get_h14_config(),
      'num_classes': 21843,
      'image_size': (224, 224),
      'url': None,
      'url_og': "https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz"
    },
    'B_16_imagenet1k': {
      'config': drop_head_variant(get_b16_config()),
      'num_classes': 1000,
      'image_size': (384, 384),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth",
      'url_og': "https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_16.npz"
    },
    'B_32_imagenet1k': {
      'config': drop_head_variant(get_b32_config()),
      'num_classes': 1000,
      'image_size': (384, 384),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_32_imagenet1k.pth",
      'url_og': "https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_32.npz"
    },
    'L_16_imagenet1k': {
      'config': drop_head_variant(get_l16_config()),
      'num_classes': 1000,
      'image_size': (384, 384),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_16_imagenet1k.pth",
      'url_og': "https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-L_16.npz"
    },
    'L_32_imagenet1k': {
      'config': drop_head_variant(get_l32_config()),
      'num_classes': 1000,
      'image_size': (384, 384),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_32_imagenet1k.pth",
      'url_og': "https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-L_32.npz"
    },
}
