import torch
import numpy as np

def contrastive_volume_loss(emb_0, emb_1, locations_0, locations_1, t):
    '''Computes a contrastive learning loss for positive and negative pairs of
    points in two volumes a and b.
    This loss is known as NT-Xent for image classification [0].
    Here, the same loss is applied to pairs of points within the same volume,
    which is assumed to have been augmented in two different ways to yield
    embedding volumes ``emb_0`` and ``emb_1``.
    For the expected shapes, we use: b=batch, c=channels, d=spatial dimensions,
    and n=number of points.
    Args:
        emb_0, emb_1 (``torch.Tensor``):
            The embeddings extracted for two differently augmented versions of
            the same volume. Embeddings are assumed to be normalized already
            (i.e., magnitude is 1).
            Expected shape: ``(b, c, dim_1, ..., dim_d)``.
        locations_0, locations_1 (``torch.Tensor``):
            List of integer coordinates to be used as indices into ``a`` and
            ``b`, such that ``locations_0[i]`` is the same location as
            ``locations_1[i]`` in the original volume.
            Expected shape: ``(b, n, d)``.
        t (``float``):
            Temperature to be used in the NT-Xent loss.
    [0] http://arxiv.org/abs/2002.05709
    '''

    n = locations_0.shape[1]
    assert n == locations_1.shape[1], (
        f"Different number of points given in locations_0 ({n}) and "
        f"locations_1 ({locations_1.shape[1]})")

    # We expect the following shapes
    # (b=batch, c=channels, d=spatial dimensions, n=num points)

    # emb_0      : (b, c, dim_1, ..., dim_d)
    # emb_1      : (b, c, dim_1, ..., dim_d)
    # locations_0: (b, n, d)
    # locations_1: (b, n, d)

    assert emb_0.shape == emb_1.shape, \
        "Embedding tensors do not have the same shape"

    b, c, *volume_shape = emb_0.shape
    d = len(volume_shape)
    v = np.prod(volume_shape)

    assert b == 1, "Batch size > 1 not yet implemented"

    # flatten the embeddings, spatial index first
    # (b, v, c)
    emb_0 = emb_0.view(b, c, v).transpose(2, 1)
    emb_1 = emb_1.view(b, c, v).transpose(2, 1)

    ind_kernel = torch.Tensor([
        np.prod(volume_shape[i + 1:])
        for i in range(d)
    ]).float().to(locations_0.device)

    # turn point coordinates into indices
    # (n,) (TODO: concatenate batches here, do the same for embeddings)
    ind_0 = torch.matmul(torch.floor(locations_0), ind_kernel).long().squeeze(dim=0)
    ind_1 = torch.matmul(torch.floor(locations_1), ind_kernel).long().squeeze(dim=0)

    # get embeddings for each point
    # (b, n, c)
    emb_0 = emb_0.index_select(dim=1, index=ind_0)
    emb_1 = emb_1.index_select(dim=1, index=ind_1)

    # concatenate embeddings of all points, such that a pos pair is indexed by
    # i and i + n
    # (b, 2n, c)
    emb = torch.cat([emb_0, emb_1], dim=1)

    # get all pairwise similarities sim_uv between all points (including within
    # lists locations_0 and locations_1)
    # (b, 2n, 2n)
    sim = torch.matmul(emb, emb.transpose(2, 1))

    # precompute exp_uv = exp(s_uv/t)
    # (b, 2n, 2n)
    exp = torch.exp(sim/t)

    # get sum_u of all e_uw (with w≠u)
    # (b, 2n)
    sum_e = torch.sum(exp, dim=2) - np.exp(1.0/t)

    # for each point u in either list
    loss = 0
    for u in range(n):

        # print(f"computing loss for point {u} and {u + n}")

        # find corresponding point v in other list
        v = u + n
        # get e_uv (= e_vu)
        e_uv = exp[:, u, v]
        # l_uv = -log(e_uv/sum_u) - log(e_vu/sum_v)
        loss_uv = -torch.sum(
            torch.log(e_uv/sum_e[:, u]) +
            torch.log(e_uv/sum_e[:, v]))

        # print(f"sim_uv = {sim[:, u, v]}")
        # print(f"e_uv = {e_uv}")
        # print(f"sum_e[:, u] = {sum_e[:, u]}")
        # print(f"sum_e[:, v] = {sum_e[:, v]}")
        # print(f"loss_uv = {loss_uv}")

        loss += loss_uv

    loss /= 2*n

    return loss
