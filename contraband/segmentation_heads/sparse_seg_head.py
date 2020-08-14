import torch
from funlib.learn.torch.models.conv4d import Conv4d
from contraband.utils import load_model, get_output_shape
import numpy as np


class SparseSegHead(torch.nn.Module):
    def __init__(self, base_encoder, h_channels, out_channels):
        """
            Segmentation head that can do sparse voxel wise trainig/prediction
            on points. Can be turned into a normal convolutional segmentation 
            head by calling 'eval'

            Args:
                
                base_encoder (:class: `torch.nn.Module`):

                    The base_encoder that this segmentation head will go on top
                    of. This is used to get the output shape and channels.

                h_channels (:class: `int`):
                    
                    How deep the embedding is.

                out_channels (:class: `int`):

                    How many channels the output should have.
        """
        super().__init__()

        self.name = "Sparse"

        self.in_channels = base_encoder.out_channels
        self.h_channels = h_channels
        self.out_channels = out_channels
        self.dims = base_encoder.dims
        self.in_shape = base_encoder.out_shape

        self.segmentation_head = torch.nn.Sequential(
            torch.nn.Linear(self.in_channels,
                            self.h_channels), torch.nn.ReLU(),
            torch.nn.Linear(self.h_channels, self.out_channels))

        self.out_shape = [out_channels]
        self.used = False

    def forward(self, h=None, points=None):
        """ 
            Performs segmentation voxel wise.
            During training if passed an embedding (h) and points it will 
            select the points from the embedding. If only given points it 
            will just evaluate on those points. If not training it will 
            evaulate on entire embedding.
            """

        if self.training and h is not None and points is not None:

            if not self.used:
                # (b, c, dim_1, ..., dim_d)
                self.b, self.c, *self.volume_shape = h.shape
                d = len(self.volume_shape)
                self.v = np.prod(self.volume_shape)

                self.ind_kernel = torch.Tensor([
                    np.prod(self.volume_shape[i + 1:]) for i in range(d)
                ]).float()
                self.used = True

            h = h.view(self.b, self.c, self.v)
            ind = torch.matmul(torch.floor(points), self.ind_kernel) \
                .long().squeeze(dim=0)

            ind = torch.cat([ind for i in range(self.c)], axis=0) \
                .view(h.shape[0], self.c, points.shape[1])

            h_p = h.gather(dim=2, index=ind).transpose(2, 1)

            segmentation = self.segmentation_head(h_p).view(
                -1, self.out_channels)

        elif self.training and h is None and points is not None:
            segmentation = self.segmentation_head(points).view(
                -1, self.out_channels)

        elif not self.training:
            segmentation = self.eval_head(h)

        else:
            raise Exception(
                """1By1 model is set to training but was not provided points.
                    Make sure you are providing points, or that you are calling
                    eval() if you want to evaluate an entire embedding"""
            )

        return segmentation

    def eval(self, in_shape=None):
        """ Overrides the default eval method to reshape the linear weights
        into convolutional layers. Still provides the required torch
        functionality of eval.
        """
        conv = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d, 4: Conv4d}[self.dims]

        self.eval_head = torch.nn.Sequential(
            conv(self.in_channels, self.h_channels, (1, ) * self.dims),
            torch.nn.ReLU(),
            conv(self.h_channels, self.out_channels, (1, ) * self.dims))

        # Turn the linear layers into convolutional layers
        # Bias will appear every other weight so use a toggle to
        # treat bias and normal weights differently
        is_bias = False
        with torch.no_grad():
            for eval_param, train_param in zip(
                    self.eval_head.parameters(),
                    self.segmentation_head.parameters()):
                if not is_bias:
                    linear_w = train_param.data
                    reshaped_linear = linear_w.view(
                        (*linear_w.shape, *((1, ) * self.dims)))
                    eval_param.data = reshaped_linear
                    assert torch.all(torch.eq(eval_param.data,
                                              reshaped_linear))
                else:
                    eval_param.data = train_param.data
                    assert torch.all(
                        torch.eq(eval_param.data, train_param.data))

                is_bias = not is_bias

        # If in_shape is none then we are not using a placaeholder model
        # and can get the output shape normally. If a new in_shape is
        # specified then we are probably using a placaeholder model
        # and we will use the given local in_shape
        if in_shape is None:
            new_out_shape = get_output_shape(self.eval_head, self.in_shape)
        else:
            new_out_shape = get_output_shape(self.eval_head, in_shape)

        # Modifiy the output shape IN PLACE
        # This must be done inplace to change the outshape
        # in the ContrastiveVolumeNet.
        self.out_shape.clear()
        for d in new_out_shape:
            self.out_shape.append(d)

        self.train(False)
