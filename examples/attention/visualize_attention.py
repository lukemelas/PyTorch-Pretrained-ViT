import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from urllib.request import urlretrieve

import torch
from torchvision import transforms

from pytorch_pretrained_vit import ViT

models_list = ['B_16', 'B_32', 'L_32', 'B_16_imagenet1k', 'B_32_imagenet1k', 'L_16_imagenet1k', 'L_32_imagenet1k']
model_name = models_list[3]
model = ViT(model_name, pretrained=True, visualize=True)

# Test Image
os.makedirs("attention_data", exist_ok=True)
img_url = "https://images.mypetlife.co.kr/content/uploads/2019/04/09192811/welsh-corgi-1581119_960_720.jpg"
urlretrieve(img_url, "attention_data/img.jpg")

transform = transforms.Compose([
    transforms.Resize(model.image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
im = Image.open("attention_data/img.jpg")
im = Image.open('img.jpg')
x = transform(im)
x.size()

# Load class names
labels_map = json.load(open('labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]

# Classify
model.eval()
with torch.no_grad():
    outputs, att_mat = model(x.unsqueeze(0))

outputs = outputs.squeeze(0)
print(outputs.shape)
print(len(att_mat))
print(att_mat[0].shape)
#print(outputs, att_mat)
#print('logits_size and att_mat sizes: ', outputs.shape, att_mat.shape)

att_mat = torch.stack(att_mat).squeeze(1)
print(att_mat.shape)

# Average the attention weights across all heads.
att_mat = torch.mean(att_mat, dim=1)
print(att_mat.shape)

# To account for residual connections, we add an identity matrix to the
# attention matrix and re-normalize the weights.
residual_att = torch.eye(att_mat.size(1))
aug_att_mat = att_mat + residual_att
aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
print('residual_att and aug_att_mat sizes: ', residual_att.shape, aug_att_mat.shape)

# Recursively multiply the weight matrices
joint_attentions = torch.zeros(aug_att_mat.size())
joint_attentions[0] = aug_att_mat[0]

for n in range(1, aug_att_mat.size(0)):
    joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
    
# Attention from the output token to the input space.
v = joint_attentions[-1] # last layer output attention map
print('joint_attentions and last layer (v) sizes: ', joint_attentions.shape, v.shape)
grid_size = int(np.sqrt(aug_att_mat.size(-1)))
mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
print(mask.shape)
mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
print(mask.shape)
result = (mask * im).astype("uint8")

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

ax1.set_title('Original')
ax2.set_title('Attention Map')
_ = ax1.imshow(im)
_ = ax2.imshow(result)

print('-----')
for idx in torch.topk(outputs, k=3).indices.tolist():
    prob = torch.softmax(outputs, -1)[idx].item()
    print('[{idx}] {label:<75} ({p:.2f}%)'.format(idx=idx, label=labels_map[idx], p=prob*100))

for i, v in enumerate(joint_attentions):
    # Attention from the output token to the input space.
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
    result = (mask * im).astype("uint8")

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    ax1.set_title('Original')
    title = 'AttentionMap_Layer{}'.format(i+1)
    ax2.set_title(title)
    _ = ax1.imshow(im)
    _ = ax2.imshow(result)
    plt.savefig(os.path.join('attention_data', title))