import torch
from contraband.pipelines.Standard2DContrastive import Standard2DContrastive
from contraband.pipelines.Standard2DSeg import Standard2DSeg
from contraband.segmentation_heads.SimpleSegHead import SimpleSegHead
from itertools import product
import math


def generate_param_grid(params):
    return [
        dict(zip(params.keys(), values))
        for values in product(*params.values())
    ]


def map_params(params):
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

    if 'elastic_params' in params and "rotation_interval" in params['elastic_params']:
        for i, dim in enumerate(params['elastic_params']["rotation_interval"]):
            if isinstance(dim, str) and "pi" in dim:
                params['elastic_params']["rotation_interval"][i] = \
                    eval(dim.replace("pi", "math.pi"))


def map_pipeline(mode, pipeline):
    if pipeline == "Standard2D":
        if mode == 'contrastive':
            return Standard2DContrastive
        else:
            return Standard2DSeg
    else:
        raise ValueError('Incorrect pipeline: ' + pipeline)
