from pytorch_pretrained_vit import ViT

models_list = ['B_16', 'B_32', 'L_16', 'L_32', 'H_14']
for model_name in models_list:
    model = ViT(model_name, pretrained=True, load_repr_layer=True)

