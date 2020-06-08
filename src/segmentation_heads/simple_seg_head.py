import torch
import numpy as np
from funlib.learn.torch.models.conv4d import Conv4d
import funlib
from src.utils import load_model
import torch.nn.functional as F
from torch import nn
from models.contrastive_volume_net import InferencingContrastiveVolumeNet

class simple_seg_head(torch.nn.Module):

    def __init__(self, base_encoder, model_dir, h_channels, out_channels, params):
        super.__init__()

        self.name = "simple_seg_head"
        
        self.base_encoder = load_model(base_encoder, model_dir, params)
        self.in_channels = base_encoder.out_channels
        self.h_channels = h_channels
        self.out_channels = out_channels
        self.dims = base_encoder.dims

        conv = {
            2: torch.nn.Conv2d,
            3: torch.nn.Conv3d,
            4: Conv4d
        }[self.dims]

        self.projection_head = torch.nn.Sequential(
            conv(self.in_channels, h_channels, (1,)*self.dims),
            torch.nn.ReLU(),
            conv(h_channels, out_channels, (1,)*self.dims)
        )

    def forward(self, raw):

        # (b, c, dim_1, ..., dim_d)
        z = self.base_encoder(raw)
        segmentation = self.projection_head(h)

        return segmentation
