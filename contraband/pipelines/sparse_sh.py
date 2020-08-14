import gunpowder as gp
import logging
import os
import numpy as np
import torch
from contraband.pipelines.utils import Blur, InspectBatch, RemoveChannelDim, \
    PrepareBatch, AddSpatialDim, \
    SetDtype, AddChannelDim, RemoveSpatialDim, \
    RandomPointGenerator, RandomPointSource, \
    FillLocations, PointsLabelsSource
from contraband.pipelines.point_loss import PointLoss
import pandas as pd
import zarr

logger = logging.getLogger(__name__)


class SparseSHTrain():
    def __init__(self, params, logdir, log_every=1):
        """
            Trains the SparseSegHead on precalculated embeddings and pre chosen 
            points. Use the `emb` mode to generate the needed embeddings and 
            then run the `generate_points.py` script the create random points.
        """

        self.params = params
        self.logdir = logdir
        self.log_every = log_every

        split = logdir.split('/')
        contrastive_ckpt = split[-1]
        comb = split[3]
        exp = split[:3]
        emb_filename = os.path.join(*exp, comb, "embs", contrastive_ckpt,
                                    'validate/raw_embs.zarr')
        
        # Get embedding data
        dataset = "embs"
        embs = zarr.open(emb_filename, mode='r')[dataset]

        # Get precomputed points
        points = pd.read_csv(
            os.path.join(*exp, "validate_points.csv"),
            converters={
                "point": lambda x: [int(i) for i in x.strip("()").split(", ")]
            })
        
        # Split pos and neg points
        pos = points.loc[points['gt'] == 1]
        neg = points.loc[points['gt'] == 0]

        # Num points refers to number of total points, but when doing
        # FG/BG prediction half will be pos and half will be neg
        self.num_points = int(self.params['num_points'] / 2)

        self.pos_points = pos["point"][:self.num_points].tolist()
        self.pos_gt = pos["gt"][:self.num_points].tolist()
        self.neg_points = neg['point'][:self.num_points].tolist()
        self.neg_gt = neg['gt'][:self.num_points].tolist()

        # (samples, embedding channels, y, x)
        is_2d = len(embs.shape) == 4
        if is_2d:
            self.pos_data = np.array([
                embs[(point[0], slice(None), *point[1:])]
                for point in self.pos_points
            ])
            self.neg_data = np.array([
                embs[(point[0], slice(None), *point[1:])]
                for point in self.neg_points
            ])
        else:
            # (Channel, emb, z, y, x)
            assert embs.shape[0] == 1, "Multichannel 3D embs are not supported"
            self.pos_data = np.array(
                [embs[(0, slice(None), *point)] for point in self.pos_points])
            self.neg_data = np.array(
                [embs[(0, slice(None), *point)] for point in self.neg_points])

        self.data = np.concatenate((self.pos_data, self.neg_data))
        self.labels = np.concatenate((self.pos_gt, self.neg_gt))

        logger.info("data shape", self.data.shape)
        logger.info("labels shape", self.labels.shape)
        self.loss = PointLoss(torch.nn.CrossEntropyLoss())

    def create_train_pipeline(self, model):
        optimizer = self.params['optimizer'](model.parameters(),
                                             **self.params['optimizer_kwargs'])
        points = gp.ArrayKey('POINTS')
        predictions = gp.ArrayKey("PREDICTIONS")
        gt_labels = gp.ArrayKey('LABELS')

        request = gp.BatchRequest()
        request[points] = gp.ArraySpec(nonspatial=True)
        request[predictions] = gp.ArraySpec(nonspatial=True)
        request[gt_labels] = gp.ArraySpec(nonspatial=True)

        pipeline = (
            PointsLabelsSource(points, self.data, gt_labels, self.labels, 1) +
            gp.Stack(self.params['batch_size']) + gp.torch.Train(
                model,
                self.loss,
                optimizer,
                inputs={'points': points},
                loss_inputs={
                    0: predictions,
                    1: gt_labels
                },
                outputs={0: predictions},
                checkpoint_basename=self.logdir + '/checkpoints/model',
                save_every=self.params['save_every'],
                log_dir=self.logdir,
                log_every=self.log_every))

        return pipeline, request
