from pytorch_pretrained_vit import ViT, ViTConfigExtended, PRETRAINED_CONFIGS

models_list = ['B_16', 'B_16_in1k']
for model_name in models_list:
    def_config = PRETRAINED_CONFIGS['{}'.format(model_name)]['config']
    configuration = ViTConfigExtended(**def_config)
    model = ViT(configuration, name=model_name, pretrained=True, load_repr_layer=True)

