import torch
from funlib.learn.torch.models.conv4d import Conv4d
from contraband.utils import load_model, get_output_shape
from contraband import utils


"""
    This file holds the ContrastiveVolumeNet and the SegmentationVolumeNet.
    These models combine the base_encoder and the projection_head/seg_head.
"""


class ContrastiveVolumeNet(torch.nn.Module):
    """
        Takes a base_encoder and creates a simple projection_head on top.
        
        Args:

            base_encoder (`torch.nn.Module`):

                The base_encoder to use.

            h_channels (`int`):

                The number of h_channels the model should have. This is
                also probably the size of the embedding of each voxel.

            out_channels (`int`):
                
                The embedding size of each voxel.
    """
    def __init__(self, base_encoder, h_channels, out_channels):
        super().__init__()

        self.base_encoder = base_encoder

        in_channels = base_encoder.out_channels
        dims = base_encoder.dims

        conv = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d, 4: Conv4d}[dims]

        self.projection_head = torch.nn.Sequential(
            conv(in_channels, h_channels, (1, ) * dims),
            torch.nn.ReLU(), conv(h_channels, out_channels, (1, ) * dims))

        self.out_shape = get_output_shape(self.projection_head, 
                                          base_encoder.out_shape)
        self.in_shape = base_encoder.in_shape

    def forward(self, raw_0, raw_1):
        """
            Takes raw_0 and raw_1 and computes z, then normalizes them.
            Returns both the unnormalized and normalized version, but
            only the normalized verions should be used for training.
        """
        # (b, c, dim_1, ..., dim_d)
        h_0 = self.base_encoder(raw_0)
        z_0 = self.projection_head(h_0)
        z_0_norm = torch.nn.functional.normalize(z_0, dim=1)

        h_1 = self.base_encoder(raw_1)
        z_1 = self.projection_head(h_1)
        z_1_norm = torch.nn.functional.normalize(z_1, dim=1)

        return h_0, h_1, z_0_norm, z_1_norm


class SegmentationVolumeNet(torch.nn.Module):
    """
        Takes a base_encoder and a seg_head to compute segmentations.
        
        Args:

            base_encoder (`torch.nn.Module`):

                The base_encoder to use.

            seg_head (`torch.nn.Module`):
                
                The seg_head to use.
    """
    def __init__(self, base_encoder, seg_head):
        super().__init__()
        self.in_shape = base_encoder.in_shape
        self.out_shape = seg_head.out_shape

        self.base_encoder = base_encoder
        self.seg_head = seg_head

    def forward(self, raw=None, **kwargs):
        """
            A flexible forward function that can deal with normal base_encoders,
            and placeholder base_encoder.

            Args:

                raw (`numpy array`, optional):

                    This is the spatially shaped data that goes through the
                    model. It is optional because the forward function 
                    should be able to handel spatial data or points (and any
                    other addition inputs). 

                    If point data on embeddings is wanted, then the embs
                    will be passed as raw. Raw will still be passed through 
                    the base_encoder, but the base_encoder should be a 
                    placeholder and will just return the raw (which are 
                    the embeddings) as h. 

                    If training a sparse baseline, then the point locations
                    will be passed as a kwarg. 
        """
        h = self.base_encoder(raw)
        z = self.seg_head(h, **kwargs)

        # When we use just points we can't return the h emb
        if h is None:
            return z
        else: 
            return z, h

    def load_base_encoder(self, checkpoint_file):
        self.base_encoder = load_model(self.base_encoder,
                                       "base_encoder.",
                                       checkpoint_file,
                                       freeze_model=True)

    def load_seg_head(self, checkpoint_file):
        self.seg_head = load_model(self.seg_head,
                                   "seg_head.",
                                   checkpoint_file,
                                   freeze_model=True)
