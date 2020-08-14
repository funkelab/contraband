import torch
import numpy as np

from contraband.segmentation_heads.sparse_seg_head import SparseSegHead
from contraband.models.Unet import Unet
from contraband.models.ContrastiveVolumeNet import SegmentationVolumeNet
import contraband.param_mapping as mapping

emb = torch.rand((4, 6, 3, 3, 3))
p = torch.Tensor([[0, 1, 2], [0, 2, 0], [0, 2, 1]] * 4).view(
    4, 3, 3).transpose(2, 1)


def test_getting_points():
    print("p_shape:", p.shape)
    print("emb_shape:", emb.shape)
    print("p:", p)

    print(emb[0,:,0,0,0])
    print(emb[0,:,1,2,2])
    print(emb[0,:,2,0,1])

    b, c, *volume_shape = emb.shape
    d = len(volume_shape)
    v = np.prod(volume_shape)
    emb = emb.view(b, c, v)

    ind_kernel = torch.Tensor([np.prod(volume_shape[i + 1:])
                               for i in range(d)]).float()

    print(ind_kernel)
    print(torch.cat([ind_kernel, ind_kernel, ind_kernel], axis=0))

    ind = torch.matmul(torch.floor(p), ind_kernel).long().squeeze(dim=0)
    ind = torch.cat([ind for i in range(c)], axis=1).view(4, c, 3)
    print(ind)
    print(ind.shape)
    print(emb.shape) 
    emb_p = emb.gather(dim=2, index=ind).transpose(2, 1)
    print(emb_p)
    print(emb_p.shape)


def model_3d():

    model_params = {
        "model": "unet",
        "in_shape": [14, 260, 260],
        "in_channels": 1,
        "num_fmaps": 12,
        "fmap_inc_factors": 6,
        "downsample_factors": [[1, 2, 2], [1, 2, 2], [1, 2, 2]],
        "kernel_size_down": [[[1, 3, 3], [3, 3, 3]], [[1, 3, 3], [1, 3, 3]]],
        "kernel_size_down_repeated": 2,
        "kernel_size_up": [[[1, 3, 3], [3, 3, 3]], [[1, 3, 3], [1, 3, 3]], [[1, 3, 3], [3, 3, 3]]],
        "kernel_size_up_repeated": 1,
        "constant_upsample": True,
        "h_channels": 6,
        "contrastive_out_channels": 6
    }
    mapping.map_model_params(model_params)
    model = Unet()
    model.make_model(model_params)

    seg_head = SparseSegHead(model, h_channels=6, out_channels=1)
    cvn = SegmentationVolumeNet(model, seg_head)

    print("Model's state_dict:")
    for param_tensor in cvn.state_dict():
        print(param_tensor, "\t", cvn.state_dict()[param_tensor].size())

    out, h = cvn(torch.rand((4, 1, 14, 260, 260)), points=p)
    print("out: ", out.shape)

        
    cvn.eval()
    seg_head.eval()
    out, h = cvn(torch.rand((4, 1, 14, 260, 260)))
    print("out: ", out.shape)


def model_2d():

    p = torch.Tensor([[0, 1, 2], [0, 2, 0]] * 4).view(
        4, 2, 3).transpose(2, 1)

    model_params = {
      "model": "unet",
      "in_shape": [260, 260],
      "in_channels": 1,
      "num_fmaps": 12,
      "fmap_inc_factors": 6,
      "downsample_factors": [[2, 2], [2, 2], [2, 2]],
      "kernel_size_down": [[[3, 3], [3, 3]]],
      "kernel_size_down_repeated": 4,
      "kernel_size_up": [[[3, 3], [3, 3]]],
      "kernel_size_up_repeated": 3,
      "constant_upsample": True,
      "h_channels": 12,
      "contrastive_out_channels": 12
    }
    mapping.map_model_params(model_params)
    model = Unet()
    model.make_model(model_params)

    seg_head = SparseSegHead(model, h_channels=6, out_channels=1)
    cvn = SegmentationVolumeNet(model, seg_head)

    print("Model's state_dict:")
    for param_tensor in cvn.state_dict():
        print(param_tensor, "\t", cvn.state_dict()[param_tensor].size())

    out, h = cvn(torch.rand((4, 1, 260, 260)), points=p)
    print("out: ", out.shape)

    cvn.eval()
    seg_head.eval()
    out, h = cvn(torch.rand((4, 1, 260, 260)))
    print("out: ", out.shape)

model_2d()
model_3d()
