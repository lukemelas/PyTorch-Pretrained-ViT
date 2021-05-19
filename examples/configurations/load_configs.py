import yaml
from pytorch_pretrained_vit import ViT, ViTConfigExtended, PRETRAINED_CONFIGS

configuration = ViTConfigExtended(hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=512,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            is_encoder_decoder=False,
            image_size=128,
            patch_size=16,
            num_channels=3,
            pos_encoding_type='learned',
            classifier='token',
            num_classes=10)

model = ViT(configuration)

with open('config_custom.yaml') as f:
    config_dic = yaml.safe_load(f)
configuration = ViTConfigExtended(**config_dic)

model = ViT(configuration)
