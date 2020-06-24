import torch
import funlib
from contraband.models.ContrastiveVolumeNet import (
    SegmentationVolumeNet,
    ContrastiveVolumeNet)
from contraband.utils import load_model, get_output_shape


class Unet2D(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.name = "Unet2D"
        self.pipeline = "Standard2D"
        self.in_shape = (260, 260)

        self.unet = funlib.learn.torch.models.UNet(
            in_channels=1,
            num_fmaps=12,
            fmap_inc_factors=6,
            downsample_factors=[(2, 2), (2, 2), (2, 2)],
            kernel_size_down=[[(3, 3), (3, 3)]] * 4,
            kernel_size_up=[[(3, 3), (3, 3)]] * 3,
            constant_upsample=True)

        self.out_channels = self.unet.out_channels
        self.dims = self.unet.dims

        self.out_shape = get_output_shape(self.unet, 
                                          [1, 1, *self.in_shape])

    def forward(self, raw):
        return self.unet.forward(raw)
