"""utils.py - Helper functions
"""

import torch
from torch.utils import model_zoo

from .configs import PRETRAINED_MODELS


def load_pretrained_weights(model, model_name, weights_path=None, load_fc=True, verbose=True):
    """Loads pretrained weights from weights path or download using url.

    Args:
        model (Module): The whole model of efficientnet.
        model_name (str): Model name of efficientnet.
        weights_path (None or str):
            str: path to pretrained weights file on the local disk.
            None: use pretrained weights downloaded from the Internet.
        load_fc (bool): Whether to load pretrained weights for fc layer at the end of the model.
        verbose (bool): Whether to print on completion
    """
    if weights_path is None:
        state_dict = model_zoo.load_url(PRETRAINED_MODELS[model_name]['url'])
    else:
        state_dict = torch.load(weights_path)
    
    if load_fc:
        ret = model.load_state_dict(state_dict, strict=False)
        assert not ret.missing_keys, 'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
    else:
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        ret = model.load_state_dict(state_dict, strict=False)
        assert set(ret.missing_keys) == set(
            ['fc.weight', 'fc.bias']), 'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
    assert not ret.unexpected_keys, 'Missing keys when loading pretrained weights: {}'.format(ret.unexpected_keys)
    
    if verbose:
        print('Loaded pretrained weights for {}'.format(model_name))


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)
