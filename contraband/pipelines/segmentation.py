import gunpowder as gp
import logging
import numpy as np
import torch
import daisy
from contraband.pipelines.utils import Blur, InspectBatch, RemoveChannelDim, \
    PrepareBatch, AddSpatialDim, \
    SetDtype, AddChannelDim, RemoveSpatialDim

logger = logging.getLogger(__name__)


class Segmentation():

    def __init__(self, params, logdir, log_every=1):

        self.params = params
        self.logdir = logdir
        self.log_every = log_every

        self.loss = torch.nn.MSELoss()

    def create_train_pipeline(self, model):

       	print(f"Creating training pipeline with batch size {self.params['batch_size']}")
        
        filename = self.params['data_file']
        raw_dataset = self.params['dataset']['train']['raw']
        gt_dataset = self.params['dataset']['train']['gt']
    
        optimizer = self.params['optimizer'](model.parameters(), 
                                             **self.params['optimizer_kwargs'])

        raw = gp.ArrayKey('RAW')
        gt_labels = gp.ArrayKey('LABELS')
        gt_aff = gp.ArrayKey('AFFINITIES')
        predictions = gp.ArrayKey('PREDICTIONS')
        emb = gp.ArrayKey('EMBEDDING')
        
        raw_data = daisy.open_ds(filename, raw_dataset)
        source_roi = gp.Roi(raw_data.roi.get_offset(), raw_data.roi.get_shape())
        source_voxel_size = gp.Coordinate(raw_data.voxel_size)
        out_voxel_size = gp.Coordinate(raw_data.voxel_size)
        
        # Get in and out shape
        in_shape = gp.Coordinate(model.in_shape)
        out_shape = gp.Coordinate(model.out_shape[2:])
        is_2d = in_shape.dims() == 2

        in_shape = in_shape * out_voxel_size
        out_shape = out_shape * out_voxel_size

        context = (in_shape - out_shape) / 2
        gt_labels_out_shape = out_shape
        # Add fake 3rd dim 
        if is_2d: 
            source_voxel_size = gp.Coordinate((1, *source_voxel_size))
            source_roi = gp.Roi((0, *source_roi.get_offset()), 
                                (raw_data.shape[0], *source_roi.get_shape()))
            context = gp.Coordinate((0, *context))
            aff_neighborhood = [[0, -1, 0], [0, 0, -1]]
            gt_labels_out_shape = (1, *gt_labels_out_shape)
        else: 
            aff_neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

        logger.info(f"source roi: {source_roi}")
        logger.info(f"in_shape: {in_shape}")
        logger.info(f"out_shape: {out_shape}")
        logger.info(f"voxel_size: {out_voxel_size}")
        logger.info(f"context: {context}")

        request = gp.BatchRequest()
        request.add(raw, in_shape)
        request.add(gt_aff, out_shape)
        request.add(predictions, out_shape)

        snapshot_request = gp.BatchRequest()
        snapshot_request[emb] = gp.ArraySpec(
            roi=gp.Roi(
                (0,) * in_shape.dims(),
                gp.Coordinate((*model.base_encoder.out_shape[2:],)) * out_voxel_size))
        snapshot_request[gt_labels] = gp.ArraySpec(roi=gp.Roi(context, gt_labels_out_shape))

        source = (
            gp.ZarrSource(
                filename,
                {
                    raw: raw_dataset,
                    gt_labels: gt_dataset 
                },
                array_specs={
                    raw: gp.ArraySpec(
                        roi=source_roi,
                        voxel_size=source_voxel_size,
                        interpolatable=True),
                    gt_labels: gp.ArraySpec(
                        roi=source_roi,
                        voxel_size=source_voxel_size)
                }
            ) +
            gp.Normalize(raw, self.params['norm_factor']) +
            gp.Pad(raw, context) + 
            gp.Pad(gt_labels, context) +
            gp.RandomLocation()
            # raw      : (l=1, h, w)
            # gt_labels: (l=1, h, w)
        )
        source = self._augmentation_pipeline(raw, source)

        pipeline = (
            source +
            # raw      : (l=1, h, w)
            # gt_labels: (l=1, h, w)
            gp.AddAffinities(aff_neighborhood,
                             gt_labels, gt_aff) + 
            SetDtype(gt_aff, np.float32) + 
            # raw      : (l=1, h, w)
            # gt_aff   : (c=2, l=1, h, w)
            AddChannelDim(raw)
            # raw      : (c=1, l=1, h, w)
            # gt_aff   : (c=2, l=1, h, w)
        )

        if is_2d:
            pipeline = (
                pipeline + 
                RemoveSpatialDim(raw) +
                RemoveSpatialDim(gt_aff)
                # raw      : (c=1, h, w)
                # gt_aff   : (c=2, h, w)
            )

        pipeline = (
            pipeline + 
            gp.Stack(self.params['batch_size']) +
            gp.PreCache() +
            # raw      : (b, c=1, h, w)
            # gt_aff   : (b, c=2, h, w)
            # (which is what train requires)
            gp.torch.Train(
                model, self.loss, optimizer,
                inputs={
                    'raw': raw
                },
                loss_inputs={
                    0: predictions,
                    1: gt_aff
                },
                outputs={
                    0: predictions,
                    1: emb
                },
                array_specs={
                    predictions: gp.ArraySpec(voxel_size=out_voxel_size),
                },
                checkpoint_basename=self.logdir + '/checkpoints/model',
                save_every=self.params['save_every'],
                log_dir=self.logdir,
                log_every=self.log_every
            ) + 
            # everything is 2D at this point, plus extra dimensions for
            # channels and batch
            # raw        : (b, c=1, h, w)
            # gt_aff     : (b, c=2, h, w)
            # predictions: (b, c=2, h, w)

            # Crop GT to look at labels
            gp.Crop(gt_labels, gp.Roi(context, gt_labels_out_shape)) +
            gp.Snapshot(
                output_dir=self.logdir + '/snapshots',
                output_filename='it{iteration}.hdf',
                dataset_names={
                    raw: 'raw',
                    gt_labels: 'gt_labels',
                    predictions: 'predictions',
                    gt_aff: 'gt_aff',
                    emb: 'emb'
                },
                additional_request=snapshot_request,
                every=self.params['save_every']) +
            gp.PrintProfilingStats(every=500)
        )

        return pipeline, request

    def _augmentation_pipeline(self, raw, source):
        if 'elastic' in self.params and self.params['elastic']:
            source = source + gp.ElasticAugment(**self.params["elastic_params"])

        if 'blur' in self.params and self.params['blur']:
            source = source + Blur(raw, **self.params["blur_params"])

        if 'simple' in self.params and self.params['simple']:
            source = source + gp.SimpleAugment(
                **self.params["simple_params"]) 

        if 'noise' in self.params and self.params['noise']:
            source = source + gp.NoiseAugment(raw, **self.params['noise_params'])
        return source
