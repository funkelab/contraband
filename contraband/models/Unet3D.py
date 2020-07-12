import torch
import funlib
from contraband.models.ContrastiveVolumeNet import (
    SegmentationVolumeNet,
    ContrastiveVolumeNet)
from contraband.utils import load_model, get_output_shape


class Unet3D(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.name = "Unet3D"
        self.pipeline = "Standard3D"
        self.z = 6 
        self.in_shape = (self.z, 260, 260)

    def make_model(self, h_channels):

        self.unet = funlib.learn.torch.models.UNet(
            in_channels=1,
            num_fmaps=h_channels,
            fmap_inc_factors=6,
            downsample_factors=[(1, 2, 2), (1, 2, 2), (1, 2, 2)],
            kernel_size_down=[[(1, 3, 3), (1, 3, 3)]] * 4,
            kernel_size_up=[[(1, 3, 3), (1, 3, 3)]] * 3,
            constant_upsample=True)

        self.out_channels = self.unet.out_channels
        self.dims = self.unet.dims

        self.out_shape = get_output_shape(self.unet, 
                                          [1, 1, *self.in_shape])

    def forward(self, raw):
        return self.unet.forward(raw)
