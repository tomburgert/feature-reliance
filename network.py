import timm


valid_model_names = [
    'convnext_tiny',
    'convnextv2_tiny',
    'densenet161',
    'efficientnet_b5',
    'efficientnetv2_rw_t',
    'inception_v3',
    'mobilenetv3_large_100',
    'regnetx_016',
    'regnety_040',
    'resnet50',
    'resnext50_32x4d',
    'xception41',
    'cait_xxs24_224',
    'convmixer_768_32',
    'swin_tiny_patch4_window7_224',
    'swin_base_patch4_window7_224',
    'deit_tiny_patch16_224',
    'deit_base_patch16_224',
    'pit_ti_224',
    'pvt_v2_b2',
    'vit_tiny_patch16_224',
    'vit_base_patch16_224',
    'resnet50_sota',
    'resnet50_clip_gap',
    'resnet50x4_clip_gap',
    'vit_base_patch16_clip_224',
    'vit_small_patch16_224.dino',
    'vit_base_patch16_224.dino',
    'vit_base_patch16_224.mae',
    'vit_base_patch16_siglip_224',
    'coat_lite_small',
    'maxvit_tiny_tf_512.in1k',
]


def get_network(arch_name, num_channels, num_classes, timm_pretrained=False):
    assert arch_name in valid_model_names
    if arch_name == 'resnet50_sota':
        network = timm.create_model(
            'resnet50',
            num_classes=num_classes,
            in_chans=num_channels,
            pretrained=timm_pretrained
        )
    else:
        network = timm.create_model(
            arch_name,
            num_classes=num_classes,
            in_chans=num_channels,
            pretrained=timm_pretrained
        )
    return network
