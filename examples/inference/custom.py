from pytorch_pretrained_vit import ViT
# The following is equivalent to ViT('B_16')
config = dict(hidden_size=512, num_heads=8, num_layers=6)
model = ViT.from_config(config)
