import json
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

import pytorch_pretrained_vit


npz_files = {
    'B_16': 'jax_weights/ViT-B_16.npz',
    'B_32': 'jax_weights/ViT-B_32.npz',
    # 'L_16': 'jax_weights/ViT-L_16.npz',  # <-- not available
    'L_32': 'jax_weights/ViT-L_32.npz',
    'B_16_imagenet1k': 'jax_weights/ViT-B_16_imagenet1k.npz',
    'B_32_imagenet1k': 'jax_weights/ViT-B_32_imagenet1k.npz',
    'L_16_imagenet1k': 'jax_weights/ViT-L_16_imagenet1k.npz',
    'L_32_imagenet1k': 'jax_weights/ViT-L_32_imagenet1k.npz',
}


def jax_to_pytorch(k):
    k = k.replace('Transformer/encoder_norm', 'norm')
    k = k.replace('LayerNorm_0', 'norm1')
    k = k.replace('LayerNorm_2', 'norm2')
    k = k.replace('MlpBlock_3/Dense_0', 'pwff.fc1')
    k = k.replace('MlpBlock_3/Dense_1', 'pwff.fc2')
    k = k.replace('MultiHeadDotProductAttention_1/out', 'proj')
    k = k.replace('MultiHeadDotProductAttention_1/query', 'attn.proj_q')
    k = k.replace('MultiHeadDotProductAttention_1/key', 'attn.proj_k')
    k = k.replace('MultiHeadDotProductAttention_1/value', 'attn.proj_v')
    k = k.replace('Transformer/posembed_input', 'positional_embedding')
    k = k.replace('encoderblock_', 'blocks.')
    k = 'patch_embedding.bias' if k == 'embedding/bias' else k
    k = 'patch_embedding.weight' if k == 'embedding/kernel' else k
    k = 'class_token' if k == 'cls' else k
    k = k.replace('head', 'fc')
    k = k.replace('kernel', 'weight')
    k = k.replace('scale', 'weight')
    k = k.replace('/', '.')
    k = k.lower()
    return k


def convert(npz, state_dict):
    new_state_dict = {}
    pytorch_k2v = {jax_to_pytorch(k): v for k, v in npz.items()}
    for pytorch_k, pytorch_v in state_dict.items():
        
        # Naming
        if 'self_attn.out_proj.weight' in pytorch_k:
            v = pytorch_k2v[pytorch_k]
            v = v.reshape(v.shape[0] * v.shape[1], v.shape[2])
        elif 'self_attn.in_proj_' in pytorch_k:
            v = np.stack((pytorch_k2v[pytorch_k + '*q'], 
                          pytorch_k2v[pytorch_k + '*k'], 
                          pytorch_k2v[pytorch_k + '*v']), axis=0)
        else:
            if pytorch_k not in pytorch_k2v:
                print(pytorch_k, list(pytorch_k2v.keys()))
                assert False
            v = pytorch_k2v[pytorch_k]
        v = torch.from_numpy(v)
        
        # Sizing
        if '.weight' in pytorch_k:
            if len(pytorch_v.shape) == 2:
                v = v.transpose(0, 1)
            if len(pytorch_v.shape) == 4:
                v = v.permute(3, 2, 0, 1)
        if ('proj.weight' in pytorch_k):
            v = v.transpose(0, 1)
            v = v.reshape(-1, v.shape[-1]).T
        if ('attn.proj_' in pytorch_k and 'weight' in pytorch_k):
            v = v.permute(0, 2, 1)
            v = v.reshape(-1, v.shape[-1])
        if 'attn.proj_' in pytorch_k and 'bias' in pytorch_k:
            v = v.reshape(-1)
        new_state_dict[pytorch_k] = v
    return new_state_dict


def check_model(model, name):
    model.eval()
    img = Image.open('../examples/simple/img.jpg')
    img = transforms.Compose([transforms.Resize(model.image_size), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])(img).unsqueeze(0)
    if 'imagenet1k' in name:
        labels_file = '../examples/simple/labels_map.txt' 
        labels_map = json.load(open(labels_file))
        labels_map = [labels_map[str(i)] for i in range(1000)]
        print('-----\nShould be index 388 (panda) w/ high probability:')
    else:
        print('~ not checked ~')
        return # labels_map = open('../examples/simple/labels_map_21k.txt').read().splitlines()
    with torch.no_grad():
        outputs = model(img).squeeze(0)
    for idx in torch.topk(outputs, k=3).indices.tolist():
        prob = torch.softmax(outputs, -1)[idx].item()
        print('[{idx}] {label:<75} ({p:.2f}%)'.format(idx=idx, label=labels_map[idx], p=prob*100))


for name, filename in npz_files.items():
    
    # Load Jax weights
    npz = np.load(filename)

    # Load PyTorch model
    model = pytorch_pretrained_vit.ViT(name=name, pretrained=False)

    # Convert weights
    new_state_dict = convert(npz, model.state_dict())

    # Load into model and test
    model.load_state_dict(new_state_dict)
    print(f'Checking: {name}')
    check_model(model, name)

    # Save weights
    new_filename = f'weights/{name}.pth'
    torch.save(new_state_dict, new_filename, _use_new_zipfile_serialization=False)
    print(f"Converted {filename} and saved to {new_filename}")

