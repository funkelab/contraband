import torch
import numpy as np


class PointLoss(torch.nn.Module):

    def __init__(self, criterion):
        """
            Gets the loss from predicted points, and gt labels,
            either in point or spatial shape using the given criterion.

            Args:

                criterion (:class: `torch loss`):

                    The torch loss to evaluate the given predictions by.
        """
        super().__init__()

        self.used = False
        self.criterion = criterion

    def forward(self, pred, gt, points=None):
        if points is not None:
            if not self.used:
                # (b, c, dim_1, ..., dim_d)
                self.b, self.c, *self.volume_shape = gt.shape
                d = len(self.volume_shape)
                self.v = np.prod(self.volume_shape)

                self.ind_kernel = torch.Tensor([np.prod(self.volume_shape[i + 1:])
                                                for i in range(d)]).float()
                self.used = True

            gt = gt.reshape(self.b, self.c, self.v)

            ind = torch.matmul(torch.floor(points), self.ind_kernel).long().squeeze(dim=0)
            # We want to grab the same location for each channel
            # In order to do this using gather, we must stack the same
            # index c times.
            ind = torch.cat([ind for i in range(self.c)], axis=0).view(gt.shape[0], self.c, points.shape[1])

            # Pick the labels for each point
            gt_p = gt.gather(dim=2, index=ind).transpose(2, 1)
            gt_p[gt_p > 0] = 1
        else:
            gt_p = gt

        return self.criterion(pred, torch.squeeze(gt_p))

        
