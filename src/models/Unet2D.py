import torch
import funlib
from models.ContrastiveVolumeNet import SegmentationVolumeNet, ContrastiveVolumeNet
from src.utils import load_model


class Unet2D():
    def __init__(self):
        self.name = "Unet2D"
        self.pipeline = "Standard2D"

    def create_model(self, params, mode, checkpoint=None):
        if mode == 'contrastive':
            return Unet2DContrastive(params)
        else:
            return Unet2DSeg(params, checkpoint)


class BaseNet(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.unet = funlib.learn.torch.models.UNet(
            in_channels=1,
            num_fmaps=12,
            fmap_inc_factors=6,
            downsample_factors=[(2, 2), (2, 2), (2, 2)],
            kernel_size_down=[[(3, 3), (3, 3)]] * 4,
            kernel_size_up=[[(3, 3), (3, 3)]] * 3,
            constant_upsample=True)


class Unet2DContrastive(BaseNet):
    def __init__(self, params):
        super().__init__(params)
        self.model = ContrastiveVolumeNet(self.unet, 20, 3)

    def forward(self, **kwargs):
        return self.model.forward(**kwargs)


class Unet2DSeg(BaseNet):
    def __init__(self, params, checkpoint):
        super().__init__(params)
        unet = load_model(self.unet, "unet.", checkpoint)
        self.model = SegmentationVolumeNet(unet, checkpoint, 20, 2,
                                           params)

    def forward(self, **kwargs):
        return self.model.forward(**kwargs)
