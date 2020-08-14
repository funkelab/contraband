import os
import pandas as pd
import gunpowder as gp
import logging
import numpy as np
import torch
import daisy
from contraband.pipelines.utils import Blur, InspectBatch, RemoveChannelDim, \
    PrepareBatch, AddSpatialDim, \
    SetDtype, AddChannelDim, RemoveSpatialDim, \
    RandomPointGenerator, RandomPointSource, \
    FillLocations, PointsLabelsSource
from contraband.pipelines.point_loss import PointLoss

logger = logging.getLogger(__name__)


class SparseBasline():
    def __init__(self, params, logdir, log_every=1):

        self.params = params
        self.logdir = logdir
        self.log_every = log_every

        split = logdir.split('/')
        exp = split[:3]

        points = pd.read_csv(
            os.path.join(*exp, "validate_points.csv"),
            converters={
                "point": lambda x: [int(i) for i in x.strip("()").split(", ")]
            })

        pos = points.loc[points['gt'] == 1]
        neg = points.loc[points['gt'] == 0]

        self.num_points = int(self.params['num_points'] / 2)

        self.pos_points = pos["point"][:self.num_points].tolist()
        self.pos_gt = pos["gt"][:self.num_points].tolist()
        self.neg_points = neg['point'][:self.num_points].tolist()
        self.neg_gt = neg['gt'][:self.num_points].tolist()

        self.data = np.concatenate((self.pos_points, self.neg_points))
        self.labels = np.concatenate((self.pos_gt, self.neg_gt))

        self.loss = PointLoss(torch.nn.CrossEntropyLoss())

    def create_train_pipeline(self, model):

        print(f"Creating training pipeline with batch size \
              {self.params['batch_size']}")

        filename = self.params['data_file']
        raw_dataset = self.params['dataset']['train']['raw']
        gt_dataset = self.params['dataset']['train']['gt']

        optimizer = self.params['optimizer'](model.parameters(),
                                             **self.params['optimizer_kwargs'])

        raw = gp.ArrayKey('RAW')
        gt_labels = gp.ArrayKey('LABELS')
        points = gp.GraphKey("POINTS")
        locations = gp.ArrayKey("LOCATIONS")
        predictions = gp.ArrayKey('PREDICTIONS')
        emb = gp.ArrayKey('EMBEDDING')

        raw_data = daisy.open_ds(filename, raw_dataset)
        source_roi = gp.Roi(raw_data.roi.get_offset(),
                            raw_data.roi.get_shape())
        source_voxel_size = gp.Coordinate(raw_data.voxel_size)
        out_voxel_size = gp.Coordinate(raw_data.voxel_size)

        # Get in and out shape
        in_shape = gp.Coordinate(model.in_shape)
        out_roi = gp.Coordinate(model.base_encoder.out_shape[2:])
        is_2d = in_shape.dims() == 2

        in_shape = in_shape * out_voxel_size
        out_roi = out_roi * out_voxel_size
        out_shape = gp.Coordinate(
            (self.params["num_points"], *model.out_shape[2:]))

        context = (in_shape - out_roi) / 2
        gt_labels_out_shape = out_roi
        # Add fake 3rd dim
        if is_2d:
            source_voxel_size = gp.Coordinate((1, *source_voxel_size))
            source_roi = gp.Roi((0, *source_roi.get_offset()),
                                (raw_data.shape[0], *source_roi.get_shape()))
            context = gp.Coordinate((0, *context))
            gt_labels_out_shape = (1, *gt_labels_out_shape)
            
            points_roi = out_voxel_size * tuple((*self.params["point_roi"],))
            points_pad = (0, *points_roi)
            context = gp.Coordinate((0, None, None))
        else:
            points_roi = source_voxel_size * tuple(self.params["point_roi"])
            points_pad = points_roi 
            context = gp.Coordinate((None, None, None))

        logger.info(f"source roi: {source_roi}")
        logger.info(f"in_shape: {in_shape}")
        logger.info(f"out_shape: {out_shape}")
        logger.info(f"voxel_size: {out_voxel_size}")
        logger.info(f"context: {context}")
        logger.info(f"out_voxel_size: {out_voxel_size}")

        request = gp.BatchRequest()
        request.add(raw, in_shape)
        request.add(points, points_roi)
        request.add(gt_labels, out_roi)
        request[locations] = gp.ArraySpec(nonspatial=True)
        request[predictions] = gp.ArraySpec(nonspatial=True)

        snapshot_request = gp.BatchRequest()
        snapshot_request[emb] = gp.ArraySpec(
            roi=gp.Roi((0, ) * in_shape.dims(),
                       gp.Coordinate((*model.base_encoder.out_shape[2:], )) *
                       out_voxel_size))

        source = (
            (gp.ZarrSource(
                filename, {
                    raw: raw_dataset,
                    gt_labels: gt_dataset
                },
                array_specs={
                    raw:
                    gp.ArraySpec(roi=source_roi,
                                 voxel_size=source_voxel_size,
                                 interpolatable=True),
                    gt_labels:
                    gp.ArraySpec(roi=source_roi, voxel_size=source_voxel_size)
                }), 
                PointsLabelsSource(points, self.data, scale=source_voxel_size)) +
            gp.MergeProvider() +
            gp.Pad(raw, context) + 
            gp.Pad(gt_labels, context) +
            gp.Pad(points, points_pad) + 
            gp.RandomLocation(ensure_nonempty=points) +
            gp.Normalize(raw, self.params['norm_factor'])
            # raw      : (source_roi)
            # gt_labels: (source_roi)
            # points   : (c=1, source_locations_shape)
            # If 2d then source_roi = (1, input_shape) in order to select a RL
        )
        source = self._augmentation_pipeline(raw, source)

        pipeline = (
            source +
            gp.Reject(ensure_nonempty=points) +
            SetDtype(gt_labels, np.int64) +
            # raw      : (source_roi)
            # gt_labels: (source_roi)
            # points   : (c=1, source_locations_shape)

            AddChannelDim(raw) +
            AddChannelDim(gt_labels)
            # raw      : (c=1, source_roi)
            # gt_labels: (c=2, source_roi)
            # points   : (c=1, source_locations_shape)
        )

        if is_2d:
            pipeline = (
                # Remove extra dim the 2d roi had
                pipeline +
                RemoveSpatialDim(raw) +
                RemoveSpatialDim(gt_labels) +
                RemoveSpatialDim(points)
                # raw      : (c=1, roi)
                # gt_labels: (c=1, roi)
                # points   : (c=1, locations_shape)
            )

        pipeline = (
            pipeline +
            FillLocations(raw, points, locations, is_2d=False, max_points=1) +
            gp.Stack(self.params['batch_size']) +
            gp.PreCache() +
            # raw      : (b, c=1, roi)
            # gt_labels: (b, c=1, roi)
            # locations: (b, c=1, locations_shape)
            # (which is what train requires)
            gp.torch.Train(
                model,
                self.loss,
                optimizer,
                inputs={
                    'raw': raw,
                    'points': locations
                },
                loss_inputs={
                    0: predictions,
                    1: gt_labels,
                    2: locations
                },
                outputs={
                    0: predictions,
                    1: emb
                },
                array_specs={
                    predictions: gp.ArraySpec(nonspatial=True),
                    emb: gp.ArraySpec(voxel_size=out_voxel_size)
                },
                checkpoint_basename=self.logdir + '/checkpoints/model',
                save_every=self.params['save_every'],
                log_dir=self.logdir,
                log_every=self.log_every) +
            # everything is 2D at this point, plus extra dimensions for
            # channels and batch
            # raw        : (b, c=1, roi)
            # gt_labels  : (b, c=1, roi)
            # predictions: (b, num_points)

            gp.Snapshot(output_dir=self.logdir + '/snapshots',
                        output_filename='it{iteration}.hdf',
                        dataset_names={
                            raw: 'raw',
                            gt_labels: 'gt_labels',
                            predictions: 'predictions',
                            emb: 'emb'
                        },
                        additional_request=snapshot_request,
                        every=self.params['save_every']) +
            InspectBatch('END') + gp.PrintProfilingStats(every=500))

        return pipeline, request

    def _augmentation_pipeline(self, raw, source):
        if 'elastic' in self.params and self.params['elastic']:
            source = source + gp.ElasticAugment(
                **self.params["elastic_params"]) + InspectBatch('After EA')

        if 'blur' in self.params and self.params['blur']:
            source = source + Blur(
                raw, **self.params["blur_params"]) + InspectBatch('After blur')

        if 'simple' in self.params and self.params['simple']:
            source = source + gp.SimpleAugment(
                **self.params["simple_params"]) + InspectBatch('After simple')

        if 'noise' in self.params and self.params['noise']:
            source = source + gp.NoiseAugment(
                raw, **
                self.params['noise_params']) + InspectBatch('After noise')
        return source
