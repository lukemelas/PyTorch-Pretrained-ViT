# ViT PyTorch

Forked from [Luke Melas-Kyriazi repository](https://github.com/lukemelas/PyTorch-Pretrained-ViT). 

### Setup

```
git clone https://github.com/arkel23/PyTorch-Pretrained-ViT.git
cd PyTorch-Pretrained-ViT
pip install -e .
python download_convert_models.py # can modify to download different models, by default it downloads all 5 ViTs pretrained on ImageNet21k
```

### Usage

```
from pytorch_pretrained_vit import ViT, ViTConfigExtended, PRETRAINED_CONFIGS

model_name = 'B_16'
def_config = PRETRAINED_CONFIGS['{}'.format(model_name)]['config']
configuration = ViTConfigExtended(**def_config)
model = ViT(configuration, name=model_name, pretrained=True, load_repr_layer=False, ret_attn_scores=False)
```

### Changes compared to original

* Added support for 'H-14' and L'16' ViT models.
* Added support for downloading the models directly from Google's cloud storage.
* Corrected the Jax to Pytorch weights transformation. Previous methodology would lead to .pth state_dict files without the 'representation layer'. `ViT('load_repr_layer'=True)` would lead to an error. If only interested in inference the representation layer was unnecessary as discussed in the original paper for the Vision Transformer, but for other applications and experiments it may be useful so I added a `download_convert_models.py` to first download the required models, convert them with all the weights, and then you can completely tune the parameters.
* Added support for visualizing attention, by returning the scores values in the multi-head self-attention layers. The visualizing script was mostly taken from [jeonsworld/ViT-pytorch repository](https://github.com/jeonsworld/ViT-pytorch).
* Added examples for inference (single image), and fine-tuning/training (using CIFAR-10).
* Modified loading of models by using configurations similar to HuggingFace's Transformers.
```
# Change the default configuration by accessing individual attributes
configuration.image_size = 128
configuration.num_classes = 10
configuration.num_hidden_layers = 3
model = ViT_modified(config=configuration, name='B_16', pretrained=True)
# for another example see examples/configurations/load_configs.py
```
* Added support to partially load ViT
```
model = ViT(config=configuration, name='B_16')
pretrained_mode = 'full_tokenizer'
weights_path = "/hdd/edwin/support/torch/hub/checkpoints/B_16.pth"
model.load_partial(weights_path=weights_path, pretrained_image_size=configuration.pretrained_image_size, 
pretrained_mode=pretrained_mode, verbose=True)
for pretrained_mode in ['full_tokenizer', 'patchprojection', 'posembeddings', 'clstoken', 
        'patchandposembeddings', 'patchandclstoken', 'posembeddingsandclstoken']:
     model.load_partial(weights_path=weights_path, 
     pretrained_image_size=configuration.pretrained_image_size, pretrained_mode=pretrained_mode, verbose=True)
```

### About

This repository contains an op-for-op PyTorch reimplementation of the [Vision Transformer](https://openreview.net/forum?id=YicbFdNTTy) architecture from [Google](https://github.com/google-research/vision_transformer), along with pre-trained models and examples.


Visual Transformers (ViT) are a straightforward application of the [transformer architecture](https://arxiv.org/abs/1706.03762) to image classification. Even in computer vision, it seems, attention is all you need. 

The ViT architecture works as follows: (1) it considers an image as a 1-dimensional sequence of patches, (2) it prepends a classification token to the sequence, (3) it passes these patches through a transformer encoder (like [BERT](https://arxiv.org/abs/1810.04805)), (4) it passes the first token of the output of the transformer through a small MLP to obtain the classification logits. 
ViT is trained on a large-scale dataset (ImageNet-21k) with a huge amount of compute. 

<div style="text-align: center; padding: 10px">
    <img src="https://raw.githubusercontent.com/google-research/vision_transformer/master/figure1.png" width="100%" style="max-width: 300px; margin: auto"/>
</div>

#### Credit

Other great repositories with this model include: 
 - [Google Research's repo](https://github.com/google-research/vision_transformer)
 - [Ross Wightman's repo](https://github.com/rwightman/pytorch-image-models)
 - [Phil Wang's repo](https://github.com/lucidrains/vit-pytorch)
 - [Eunkwang Jeon's repo](https://github.com/jeonsworld/ViT-pytorch)
 - [Luke Melas-Kyriazi repo](https://github.com/lukemelas/PyTorch-Pretrained-ViT)

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.

I look forward to seeing what the community does with these models!
