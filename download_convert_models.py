from pytorch_pretrained_vit import ViT

models_list = ['B_16', 'B_32', 'L_16', 'L_32', 'H_14', 'B_16_imagenet1k', 'B_32_imagenet1k', 'L_16_imagenet1k', 'L_32_imagenet1k']
for model_name in models_list:
    model = ViT(model_name, pretrained=True, load_repr_layer=True)

