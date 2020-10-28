"""configs.py - ViT model configurations, based on:
https://github.com/google-research/vision_transformer/blob/master/vit_jax/configs.py
"""

WEIGHTS = {
    'B_16': None,
    'B_32': None,
    'L_16': None,
    'L_32': None,
    'B_16_imagenet1k': None,
    'B_32_imagenet1k': None,
    'L_16_imagenet1k': None,
    'L_32_imagenet1k': None,
}

CONFIGS = {
    'B_16': get_b16_config(),
    'B_32': get_b32_config(),
    'L_16': get_l16_config(),
    'L_32': get_l32_config(),
}


def get_base_config():
    """Base ViT config ViT"""
    return dict(
      dim=768,
      ff_dim=3072,
      num_heads=12,
      num_layers=12,
      attention_dropout_rate=0.0,
      dropout_rate=0.1,
      representation_size=None,
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
    hidden_size=1024,
    ff_dim=4096,
    num_heads=16,
    num_layers=24,
    attention_dropout_rate=0.0,
    dropout_rate=0.1,
  ))
  return config


def get_l32_config():
  """Returns the ViT-L/32 configuration."""
  config = get_l16_config()
  config.update(dict(patches=(32, 32)))
  return config