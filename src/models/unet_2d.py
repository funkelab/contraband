import torch
import numpy as np
from funlib.learn.torch.models.conv4d import Conv4d
import funlib
from models.contrastive_volume_net import TrainingContrastiveVolumeNet, InferencingContrastiveVolumeNet


class unet_2d():

    def __init__(self):
        self.name = "unet_2d"
        self.pipeline = "standard_2d"

    def create_model(self, params, training):
        if training:
            return training_unet_2d(params)
        else:
            return inferencing_unet_2d(params)

class base_unet_2d(torch.nn.Module):

    def __init__(self, params):
        super().__init__()
        self.unet = funlib.learn.torch.models.UNet(
                    in_channels=1,
                    num_fmaps=12,
                    fmap_inc_factors=6,
                    downsample_factors=[(2, 2), (2, 2), (2, 2)],
                    kernel_size_down=[[(3, 3), (3, 3)]]*4,
                    kernel_size_up=[[(3, 3), (3, 3)]]*3,
                    constant_upsample=True)


class training_unet_2d(base_unet_2d):

    def __init__(self, params):
        super().__init__(params)
        self.model = TrainingContrastiveVolumeNet(self.unet, 20, 3)

    def forward(self, **kwargs):
        return self.model.forward(**kwargs)


class inferencing_unet_2d(base_unet_2d):

    def __init__(self, params):
        super().__init__(params)
        self.model = InferencingContrastiveVolumeNet(self.unet, 20, 2)

    def forward(self, **kwargs):
        return self.model.forward(**kwargs)
