import torch
from funlib.learn.torch.models.conv4d import Conv4d
from contraband.utils import load_model


class SimpleSegHead(torch.nn.Module):

    def __init__(self, base_encoder, h_channels, out_channels):
        super().__init__()

        self.name = "SimpleSegHead"

        # self.base_encoder = base_encoder # load_model(base_encoder, model_dir)
        self.in_channels = base_encoder.out_channels
        self.h_channels = h_channels
        self.out_channels = out_channels
        self.dims = base_encoder.dims

        conv = {
            2: torch.nn.Conv2d,
            3: torch.nn.Conv3d,
            4: Conv4d
        }[self.dims]

        self.segmentation_head = torch.nn.Sequential(
            conv(self.in_channels, h_channels, (1,) * self.dims),
            torch.nn.ReLU(),
            conv(h_channels, out_channels, (1,) * self.dims)
        )
        # add sigmoid

    def forward(self, h):

        # (b, c, dim_1, ..., dim_d)
        # h = self.base_encoder(raw)
        segmentation = self.segmentation_head(h)

        return segmentation

