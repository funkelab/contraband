import torch
import funlib
from funlib.learn.torch.models.conv4d import Conv4d
from contraband.utils import get_output_shape


class Unet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.name = "Unet"

    def make_model(self, model_params):

        self.unet = funlib.learn.torch.models.UNet(
            in_channels=model_params['in_channels'],
            num_fmaps=model_params['num_fmaps'],
            fmap_inc_factors=model_params['fmap_inc_factors'],
            downsample_factors=model_params['downsample_factors'],
            kernel_size_down=model_params['kernel_size_down'],
            kernel_size_up=model_params['kernel_size_up'],
            constant_upsample=model_params['constant_upsample'])

        self.dims = self.unet.dims
        self.in_shape = model_params['in_shape']

        self.legacy_no_h = False
        if 'legacy_no_h' not in model_params:
            conv = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d, 4: Conv4d}[self.dims]
            self.h_layer = conv(model_params['num_fmaps'], model_params["h_channels"], (1,) * self.dims)
        else:
            self.legacy_no_h = True

        self.out_channels = model_params["h_channels"] 

        self.out_shape = get_output_shape(self, [1, 1, *self.in_shape])

    def forward(self, raw):
        unet_out = self.unet.forward(raw)
        if not self.legacy_no_h: 
            h = self.h_layer(unet_out)
        else:
            h = unet_out
        return h 
