import torch
from funlib.learn.torch.models.conv4d import Conv4d


class BaseVolumeNet(torch.nn.Module):
    def __init__(self, base_encoder, h_channels, out_channels):

        super().__init__()

        self.base_encoder = base_encoder
        self.in_channels = base_encoder.out_channels
        self.h_channels = h_channels
        self.out_channels = out_channels
        self.dims = base_encoder.dims


class ContrastiveVolumeNet(BaseVolumeNet):
    def __init__(self, base_encoder, h_channels, out_channels):
        super().__init__(base_encoder, h_channels, out_channels)

        conv = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d, 4: Conv4d}[self.dims]

        self.projection_head = torch.nn.Sequential(
            conv(self.in_channels, h_channels, (1, ) * self.dims),
            torch.nn.ReLU(), conv(h_channels, out_channels, (1, ) * self.dims))

    def forward(self, raw_0, raw_1):

        # (b, c, dim_1, ..., dim_d)
        h_0 = self.base_encoder(raw_0)
        z_0 = self.projection_head(h_0)
        z_0_norm = torch.nn.functional.normalize(z_0, 1)

        h_1 = self.base_encoder(raw_1)
        z_1 = self.projection_head(h_1)
        z_1_norm = torch.nn.functional.normalize(z_1, 1)

        return h_0, h_1, z_0_norm, z_1_norm


class SegmentationVolumeNet(BaseVolumeNet):
    def __init__(self, base_encoder, model_dir, h_channels, out_channels,
                 params):
        super().__init__(base_encoder, h_channels, out_channels)
        self.seg_head = params['seg_head'](base_encoder, model_dir, h_channels,
                                           out_channels, params)

    def forward(self, raw):
        h = self.base_encoder(raw)
        z = self.seg_head(h)

        return z


if __name__ == "__main__":

    emb_0 = torch.randint(0, 10, (1, 3, 100, 20, 10)).float()
    emb_1 = torch.randint(0, 10, (1, 3, 100, 20, 10)).float()
    emb_0 = torch.nn.functional.normalize(emb_0, 2)
    emb_1 = torch.nn.functional.normalize(emb_1, 2)

    locations_0 = torch.Tensor([[[0, 0, 0], [1, 1, 1]]])
    locations_1 = torch.Tensor([[[2, 2, 2], [3, 3, 3]]])

    loss = contrastive_volume_loss(emb_0, emb_1, locations_0, locations_1, 1.0)

    print(f"a[0, 0, 0]: {emb_0[0, :, 0, 0, 0]}")
    print(f"b[2, 2, 2]: {emb_1[0, :, 2, 2, 2]}")
    print(loss)
