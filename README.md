# ViT PyTorch

### Quickstart

Install with `pip install pytorch_pretrained_vit` and load a pretrained ViT with:
```python
from pytorch_pretrained_vit import ViT
model = ViT('B_16_imagenet1k', pretrained=True)
```

Or find a Google Colab example [here](https://colab.research.google.com/drive/1muZ4QFgVfwALgqmrfOkp7trAvqDemckO?usp=sharing).  

### Overview
This repository contains an op-for-op PyTorch reimplementation of the [Visual Transformer](https://openreview.net/forum?id=YicbFdNTTy) architecture from [Google](https://github.com/google-research/vision_transformer), along with pre-trained models and examples.

The goal of this implementation is to be simple, highly extensible, and easy to integrate into your own projects. 

At the moment, you can easily:
 * Load pretrained ViT models
 * Evaluate on ImageNet or your own data
 * Finetune ViT on your own dataset

_(Upcoming features)_ Coming soon: 
 * Train ViT from scratch on ImageNet (1K)
 * Export to ONNX for efficient inference

### Table of contents
1. [About ViT](#about-vit)
2. [About ViT-PyTorch](#about-vit-pytorch)
3. [Installation](#installation)
4. [Usage](#usage)
    * [Load pretrained models](#loading-pretrained-models)
    * [Example: Classify](#example-classification)
    <!-- * [Example: Extract features](#example-feature-extraction) -->
    <!-- * [Example: Export to ONNX](#example-export) -->
6. [Contributing](#contributing)

### About ViT

Visual Transformers (ViT) are a straightforward application of the [transformer architecture](https://arxiv.org/abs/1706.03762) to image classification. Even in computer vision, it seems, attention is all you need. 

The ViT architecture works as follows: (1) it considers an image as a 1-dimensional sequence of patches, (2) it prepends a classification token to the sequence, (3) it passes these patches through a transformer encoder (like [BERT](https://arxiv.org/abs/1810.04805)), (4) it passes the first token of the output of the transformer through a small MLP to obtain the classification logits. 
ViT is trained on a large-scale dataset (ImageNet-21k) with a huge amount of compute. 

<div style="text-align: center; padding: 10px">
    <img src="https://raw.githubusercontent.com/google-research/vision_transformer/master/figure1.png" width="100%" style="max-width: 300px; margin: auto"/>
</div>


### About ViT-PyTorch

ViT-PyTorch is a PyTorch re-implementation of ViT. It is consistent with the [original Jax implementation](https://github.com/google-research/vision_transformer), so that it's easy to load Jax-pretrained weights.

At the same time, we aim to make our PyTorch implementation as simple, flexible, and extensible as possible.

### Installation

Install with pip:
```bash
pip install pytorch_pretrained_vit
```

Or from source:
```bash
git clone https://github.com/lukemelas/ViT-PyTorch
cd ViT-Pytorch
pip install -e .
```

### Usage

#### Loading pretrained models

Loading a pretrained model is easy:
```python
from pytorch_pretrained_vit import ViT
model = ViT('B_16_imagenet1k', pretrained=True)
```

Details about the models are below:

|    *Name*         |* Pretrained on *|*Finetuned on*|*Available? *|
|:-----------------:|:---------------:|:------------:|:-----------:|
| `B_16`            |  ImageNet-21k   | -            |      ✓      |
| `B_32`            |  ImageNet-21k   | -            |      ✓      |
| `L_16`            |  ImageNet-21k   | -            |      -      |
| `L_32`            |  ImageNet-21k   | -            |      ✓      |
| `B_16_imagenet1k` |  ImageNet-21k   | ImageNet-1k  |      ✓      |
| `B_32_imagenet1k` |  ImageNet-21k   | ImageNet-1k  |      ✓      |
| `L_16_imagenet1k` |  ImageNet-21k   | ImageNet-1k  |      ✓      |
| `L_32_imagenet1k` |  ImageNet-21k   | ImageNet-1k  |      ✓      |

#### Custom ViT

Loading custom configurations is just as easy: 
```python
from pytorch_pretrained_vit import ViT
# The following is equivalent to ViT('B_16')
config = dict(hidden_size=512, num_heads=8, num_layers=6)
model = ViT.from_config(config)
```

#### Example: Classification

Below is a simple, complete example. It may also be found as a Jupyter notebook in `examples/simple` or as a [Colab Notebook]().  
<!-- TODO: new Colab -->

```python
import json
from PIL import Image
import torch
from torchvision import transforms

# Load ViT
from pytorch_pretrained_vit import ViT
model = ViT('B_16_imagenet1k', pretrained=True)
model.eval()

# Load image
# NOTE: Assumes an image `img.jpg` exists in the current directory
img = transforms.Compose([
    transforms.Resize((384, 384)), 
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])(Image.open('img.jpg')).unsqueeze(0)
print(img.shape) # torch.Size([1, 3, 384, 384])

# Classify
with torch.no_grad():
    outputs = model(img)
print(outputs.shape)  # (1, 1000)
```

<!-- #### Example: Feature Extraction

You can easily extract features with `model.extract_features`:
```python
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0')

# ... image preprocessing as in the classification example ...
print(img.shape) # torch.Size([1, 3, 384, 384])

features = model.extract_features(img)
print(features.shape) # torch.Size([1, 1280, 7, 7])
``` -->

<!-- #### Example: Export to ONNX

Exporting to ONNX for deploying to production is now simple:
```python
import torch
from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_pretrained('efficientnet-b1')
dummy_input = torch.randn(10, 3, 240, 240)

model.set_swish(memory_efficient=False)
torch.onnx.export(model, dummy_input, "test-b1.onnx", verbose=True)
```

[Here](https://colab.research.google.com/drive/1rOAEXeXHaA8uo3aG2YcFDHItlRJMV0VP) is a Colab example. -->


#### ImageNet

See `examples/imagenet` for details about evaluating on ImageNet.

#### Credit

Other great repositories with this model include: 
 - [Ross Wightman's repo](https://github.com/rwightman/pytorch-image-models)
 - [Phil Wang's repo](https://github.com/lucidrains/vit-pytorch)

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.

I look forward to seeing what the community does with these models!
