import torch
import torch.nn as nn

from timm.models.efficientnet import EfficientNet
from timm.models.efficientnet import decode_arch_def, round_channels, default_cfgs
from torch.nn import SiLU as Swish  # Use PyTorch's native Swish/SiLU for compatibility

from ._base import EncoderMixin


def get_efficientnet_kwargs(channel_multiplier=1.0, depth_multiplier=1.0, drop_rate=0.2):
    """Creates an EfficientNet model."""
    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'],
        ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'],
        ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_channels(1280, channel_multiplier, 8, None),
        stem_size=32,
        channel_multiplier=channel_multiplier,
        act_layer=Swish,
        norm_kwargs={},
        drop_rate=drop_rate,
        drop_path_rate=0.2,
    )
    return model_kwargs


def gen_efficientnet_lite_kwargs(channel_multiplier=1.0, depth_multiplier=1.0, drop_rate=0.2):
    """Creates an EfficientNet-Lite model."""
    arch_def = [
        ['ds_r1_k3_s1_e1_c16'],
        ['ir_r2_k3_s2_e6_c24'],
        ['ir_r2_k5_s2_e6_c40'],
        ['ir_r3_k3_s2_e6_c80'],
        ['ir_r3_k5_s1_e6_c112'],
        ['ir_r4_k5_s2_e6_c192'],
        ['ir_r1_k3_s1_e6_c320'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, fix_first_last=True),
        num_features=1280,
        stem_size=32,
        fix_stem=True,
        channel_multiplier=channel_multiplier,
        act_layer=nn.ReLU6,
        norm_kwargs={},
        drop_rate=drop_rate,
        drop_path_rate=0.2,
    )
    return model_kwargs


class EfficientNetBaseEncoder(EfficientNet, EncoderMixin):
    def __init__(self, stage_idxs, out_channels, depth=5, **kwargs):
        # Fix: Remove arguments that newer timm versions don't accept
        kwargs.pop('channel_multiplier', None)
        kwargs.pop('norm_kwargs', None)

        super().__init__(**kwargs)

        self._stage_idxs = stage_idxs
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

        # We delete the classifier because we don't need it for segmentation
        del self.classifier

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv_stem, self.bn1, self.act1),
            self.blocks[:self._stage_idxs[0]],
            self.blocks[self._stage_idxs[0]:self._stage_idxs[1]],
            self.blocks[self._stage_idxs[1]:self._stage_idxs[2]],
            self.blocks[self._stage_idxs[2]:],
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("classifier.bias", None)
        state_dict.pop("classifier.weight", None)
        super().load_state_dict(state_dict, **kwargs)


class EfficientNetEncoder(EfficientNetBaseEncoder):
    def __init__(self, stage_idxs, out_channels, depth=5, channel_multiplier=1.0, depth_multiplier=1.0, drop_rate=0.2):
        kwargs = get_efficientnet_kwargs(channel_multiplier, depth_multiplier, drop_rate)
        super().__init__(stage_idxs, out_channels, depth, **kwargs)


class EfficientNetLiteEncoder(EfficientNetBaseEncoder):
    def __init__(self, stage_idxs, out_channels, depth=5, channel_multiplier=1.0, depth_multiplier=1.0, drop_rate=0.2):
        kwargs = gen_efficientnet_lite_kwargs(channel_multiplier, depth_multiplier, drop_rate)
        super().__init__(stage_idxs, out_channels, depth, **kwargs)


def prepare_settings(settings):
    # 1. Try to unwrap 'DefaultCfg' (new timm), but catch crashes
    try:
        if hasattr(settings, 'default'):
            settings = settings.default
    except (IndexError, AttributeError):
        pass

    # 2. Extract values safely
    if isinstance(settings, dict):
        mean = settings.get('mean')
        std = settings.get('std')
        url = settings.get('url')
    else:
        mean = getattr(settings, 'mean', None)
        std = getattr(settings, 'std', None)
        url = getattr(settings, 'url', None)

        if mean is None and hasattr(settings, 'input_mean'): mean = settings.input_mean
        if std is None and hasattr(settings, 'input_std'): std = settings.input_std

    if mean is None: mean = [0.485, 0.456, 0.406]
    if std is None: std = [0.229, 0.224, 0.225]

    return {
        "mean": mean,
        "std": std,
        "url": url,
        "input_range": (0, 1),
        "input_space": "RGB",
    }


timm_efficientnet_encoders = {

    "timm-efficientnet-b0": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["tf_efficientnet_b0"]),
            "advprop": prepare_settings(default_cfgs["tf_efficientnet_b0_ap"]),
            "noisy-student": prepare_settings(default_cfgs["tf_efficientnet_b0_ns"]),
        },
        "params": {
            "out_channels": (3, 32, 24, 40, 112, 320),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 1.0,
            "depth_multiplier": 1.0,
            "drop_rate": 0.2,
        },
    },

    "timm-efficientnet-b1": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["tf_efficientnet_b1"]),
            "advprop": prepare_settings(default_cfgs["tf_efficientnet_b1_ap"]),
            "noisy-student": prepare_settings(default_cfgs["tf_efficientnet_b1_ns"]),
        },
        "params": {
            "out_channels": (3, 32, 24, 40, 112, 320),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 1.0,
            "depth_multiplier": 1.1,
            "drop_rate": 0.2,
        },
    },

    "timm-efficientnet-b2": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["tf_efficientnet_b2"]),
            "advprop": prepare_settings(default_cfgs["tf_efficientnet_b2_ap"]),
            "noisy-student": prepare_settings(default_cfgs["tf_efficientnet_b2_ns"]),
        },
        "params": {
            "out_channels": (3, 32, 24, 48, 120, 352),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 1.1,
            "depth_multiplier": 1.2,
            "drop_rate": 0.3,
        },
    },

    "timm-efficientnet-b3": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["tf_efficientnet_b3"]),
            "advprop": prepare_settings(default_cfgs["tf_efficientnet_b3_ap"]),
            "noisy-student": prepare_settings(default_cfgs["tf_efficientnet_b3_ns"]),
        },
        "params": {
            "out_channels": (3, 40, 32, 48, 136, 384),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 1.2,
            "depth_multiplier": 1.4,
            "drop_rate": 0.3,
        },
    },

    "timm-efficientnet-b4": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["tf_efficientnet_b4"]),
            "advprop": prepare_settings(default_cfgs["tf_efficientnet_b4_ap"]),
            "noisy-student": prepare_settings(default_cfgs["tf_efficientnet_b4_ns"]),
        },
        "params": {
            "out_channels": (3, 48, 32, 56, 160, 448),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 1.4,
            "depth_multiplier": 1.8,
            "drop_rate": 0.4,
        },
    },

    "timm-efficientnet-b5": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["tf_efficientnet_b5"]),
            "advprop": prepare_settings(default_cfgs["tf_efficientnet_b5_ap"]),
            "noisy-student": prepare_settings(default_cfgs["tf_efficientnet_b5_ns"]),
        },
        "params": {
            "out_channels": (3, 48, 40, 64, 176, 512),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 1.6,
            "depth_multiplier": 2.2,
            "drop_rate": 0.4,
        },
    },

    "timm-efficientnet-b6": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["tf_efficientnet_b6"]),
            "advprop": prepare_settings(default_cfgs["tf_efficientnet_b6_ap"]),
            "noisy-student": prepare_settings(default_cfgs["tf_efficientnet_b6_ns"]),
        },
        "params": {
            "out_channels": (3, 56, 40, 72, 200, 576),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 1.8,
            "depth_multiplier": 2.6,
            "drop_rate": 0.5,
        },
    },

    "timm-efficientnet-b7": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["tf_efficientnet_b7"]),
            "advprop": prepare_settings(default_cfgs["tf_efficientnet_b7_ap"]),
            "noisy-student": prepare_settings(default_cfgs["tf_efficientnet_b7_ns"]),
        },
        "params": {
            "out_channels": (3, 64, 48, 80, 224, 640),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 2.0,
            "depth_multiplier": 3.1,
            "drop_rate": 0.5,
        },
    },

    "timm-efficientnet-b8": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["tf_efficientnet_b8"]),
            "advprop": prepare_settings(default_cfgs["tf_efficientnet_b8_ap"]),
        },
        "params": {
            "out_channels": (3, 72, 56, 88, 248, 704),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 2.2,
            "depth_multiplier": 3.6,
            "drop_rate": 0.5,
        },
    },

    "timm-efficientnet-l2": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "noisy-student": prepare_settings(default_cfgs["tf_efficientnet_l2_ns"]),
        },
        "params": {
            "out_channels": (3, 136, 104, 176, 480, 1376),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 4.3,
            "depth_multiplier": 5.3,
            "drop_rate": 0.5,
        },
    },

    "timm-tf_efficientnet_lite0": {
        "encoder": EfficientNetLiteEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["tf_efficientnet_lite0"]),
        },
        "params": {
            "out_channels": (3, 32, 24, 40, 112, 320),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 1.0,
            "depth_multiplier": 1.0,
            "drop_rate": 0.2,
        },
    },

    "timm-tf_efficientnet_lite1": {
        "encoder": EfficientNetLiteEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["tf_efficientnet_lite1"]),
        },
        "params": {
            "out_channels": (3, 32, 24, 40, 112, 320),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 1.0,
            "depth_multiplier": 1.1,
            "drop_rate": 0.2,
        },
    },

    "timm-tf_efficientnet_lite2": {
        "encoder": EfficientNetLiteEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["tf_efficientnet_lite2"]),
        },
        "params": {
            "out_channels": (3, 32, 24, 48, 120, 352),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 1.1,
            "depth_multiplier": 1.2,
            "drop_rate": 0.3,
        },
    },

    "timm-tf_efficientnet_lite3": {
        "encoder": EfficientNetLiteEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["tf_efficientnet_lite3"]),
        },
        "params": {
            "out_channels": (3, 32, 32, 48, 136, 384),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 1.2,
            "depth_multiplier": 1.4,
            "drop_rate": 0.3,
        },
    },

    "timm-tf_efficientnet_lite4": {
        "encoder": EfficientNetLiteEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["tf_efficientnet_lite4"]),
        },
        "params": {
            "out_channels": (3, 32, 32, 56, 160, 448),
            "stage_idxs": (2, 3, 5),
            "channel_multiplier": 1.4,
            "depth_multiplier": 1.8,
            "drop_rate": 0.4,
        },
    },
}