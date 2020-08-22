# Math is not actually unsed
import math
import torch
from contraband.pipelines.contrastive import Contrastive
from contraband.pipelines.segmentation import Segmentation
from contraband.pipelines.sparse_baseline import SparseBasline
from contraband.pipelines.sparse_sh import SparseSH
from contraband.segmentation_heads.simple_seg_head import SimpleSegHead
from contraband.segmentation_heads.sparse_seg_head import SparseSegHead
from contraband.models.unet import Unet
from itertools import product


def generate_param_grid(params):
    return [
        dict(zip(params.keys(), values))
        for values in product(*params.values())
    ]


def map_params(params):
    """
        Maps pipeline parameters to the objects they should
        represet.
        Ex: optimizer -> torch optimizer module
    """
    if params['optimizer'] == 'adam':
        kwargs = {}
        if 'lr' in params:
            kwargs['lr'] = params['lr']
        if 'clipvalue' in params:
            kwargs['clipvalue'] = params['clipvalue']
        elif 'clipnorm' in params:
            kwargs['clipnorm'] = params['clipnorm']
        if 'decay' in params:
            kwargs['decay'] = params['decay']
        params['optimizer'] = torch.optim.Adam
        params['optimizer_kwargs'] = kwargs

    if 'seg_head' in params:
        if params['seg_head'] == 'SimpleSegHead':
            params['seg_head'] = SimpleSegHead
        if params['seg_head'] == 'Sparse':
            params['seg_head'] = SparseSegHead

    if 'elastic_params' in params and "rotation_interval" in params['elastic_params']:
        for i, dim in enumerate(params['elastic_params']["rotation_interval"]):
            if isinstance(dim, str) and "pi" in dim:
                params['elastic_params']["rotation_interval"][i] = \
                    eval(dim.replace("pi", "math.pi"))


def map_model_params(params):
    """
        Maps the model parameters.
    """
    if params['model'] == "unet":
        params['model'] = Unet()

    # Make downsample_factors tuples, torch complains otherwise
    params['downsample_factors'] = [tuple(factor) for factor in params['downsample_factors']]

    params['kernel_size_down'] = params['kernel_size_down'] * params['kernel_size_down_repeated']
    params['kernel_size_up'] = params['kernel_size_up'] * params['kernel_size_up_repeated']


def map_pipeline(mode, pipeline):
    """
        Returns the correct pipelines based on pipeline
        asked for.
    """
    if pipeline == "standard":
        if mode == 'contrastive':
            return Contrastive
        else:
            return Segmentation
    elif pipeline == "sparse_sh":
        if mode == 'contrastive':
            return Contrastive
        else:
            return SparseSH
    elif pipeline == "baseline_sparse_sh":
        if mode == 'contrastive':
            return Contrastive
        else:
            return SparseBasline
    else:
        raise ValueError('Incorrect pipeline: ' + pipeline)
