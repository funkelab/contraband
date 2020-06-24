import math
import gunpowder as gp
import logging
import numpy as np
import torch
import zarr
from contraband.pipelines.utils import Blur, InspectBatch, RemoveChannelDim, \
    AddRandomPoints, PrepareBatch, AddSpatialDim, \
    SetDtype, AddChannelDim, RemoveSpatialDim

logging.basicConfig(level=logging.INFO)


class Standard2DSeg():

    def __init__(self, params, logdir, log_every=1):

        self.params = params
        self.logdir = logdir
        self.log_every = log_every

        self.loss = torch.nn.MSELoss()

    def create_train_pipeline(self, model):

       	print(f"Creating training pipeline with batch size {self.params['batch_size']}")
        
        data_file = self.params['data_file']
        raw_dataset = self.params['dataset']['train']['raw']
        gt_dataset = self.params['dataset']['train']['gt']
    
        optimizer = self.params['optimizer'](model.parameters(), 
                                             **self.params['optimizer_kwargs'])

        raw = gp.ArrayKey('RAW')
        gt_labels = gp.ArrayKey('LABELS')
        gt_aff = gp.ArrayKey('AFFINITIES')
        predictions = gp.ArrayKey('PREDICTIONS')

        request = gp.BatchRequest()
        request.add(raw, (260, 260))
        request.add(gt_aff, (168, 168))
        request.add(predictions, (168, 168))
        
        source_shape = zarr.open(data_file)[raw_dataset].shape
        gt_source_shape = zarr.open(data_file)[gt_dataset].shape
        # plt.show()
        raw_roi = gp.Roi((0, 0, 0), source_shape)
        gt_roi = gp.Roi((0, 0, 0), gt_source_shape) 

        source = (
            gp.ZarrSource(
                data_file,
                {
                    raw: raw_dataset,
                    gt_labels: gt_dataset 
                },
                # fake 3D data
                array_specs={
                    raw: gp.ArraySpec(
                        roi=raw_roi,
                        voxel_size=(1, 1, 1),
                        interpolatable=True),
                    gt_labels: gp.ArraySpec(
                        roi=gt_roi,
                        voxel_size=(1, 1, 1),
                        interpolatable=True,
                        dtype=np.uint32)
                }
            ) +
            # SetDtype(gt_aff, np.uint8) +
            gp.Normalize(raw, self.params['norm_factor']) +
            gp.Pad(raw, (0, 200, 200)) + 
            gp.Pad(gt_labels, (0, 300, 300)) +
            gp.RandomLocation()
            # raw      : (l=1, h, w)
            # gt_labels: (l=1, h, w)
        )
        source = self._augmentation_pipeline(raw, source)

        pipeline = (
            source +
            # raw      : (l=1, h, w)
            # gt_labels: (l=1, h, w)
            gp.AddAffinities([[0, -1, 0], [0, 0, -1]],
                             gt_labels, gt_aff) + 
            gp.Normalize(gt_aff) + 
            # raw      : (l=1, h, w)
            # gt_aff   : (c=2, l=1, h, w)
            AddChannelDim(raw) +
            # raw      : (c=1, l=1, h, w)
            # gt_aff   : (c=2, l=1, h, w)
            RemoveSpatialDim(raw) +
            RemoveSpatialDim(gt_aff) +
            # raw      : (c=1, h, w)
            # gt_aff   : (c=2, h, w)
            # InspectBatch('before stack:') +
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
                    0: predictions
                },
                array_specs={
                    predictions: gp.ArraySpec(voxel_size=(1, 1)),
                },
                checkpoint_basename=self.logdir + '/checkpoints/model',
                save_every=1,
                log_dir=self.logdir,
                log_every=self.log_every
            ) + 
            # everything is 2D at this point, plus extra dimensions for
            # channels and batch
            # raw        : (b, c=1, h, w)
            # gt_aff     : (b, c=2, h, w)
            # predictions: (b, c=2, h, w)
            gp.Snapshot(
                output_dir=self.logdir + '/snapshots',
                output_filename='it{iteration}.hdf',
                dataset_names={
                    raw: 'raw',
                    predictions: 'predictions',
                    gt_labels: 'gt_labels'
                },
                every=500) +
            gp.PrintProfilingStats(every=10)
        )

        return pipeline, request

    def _augmentation_pipeline(self, raw, source):
        if 'elastic' in self.params and self.params['elastic']:
            source = source + gp.ElasticAugment(
                control_point_spacing=(1, 10, 10),
                jitter_sigma=(0, 0.1, 0.1),
                rotation_interval=(0, math.pi / 2))

        if 'blur' in self.params and self.params['blur']:
            source = source + Blur(raw, sigma=[0, 1, 1])

        if 'simple' in self.params and self.params['simple']:
            source = source + gp.SimpleAugment(
                mirror_only=(1, 2),
                transpose_only=(1, 2)) 

        if 'noise' in self.params and self.params['noise']:
            source = source + gp.NoiseAugment(raw, var=0.01)

        return source
