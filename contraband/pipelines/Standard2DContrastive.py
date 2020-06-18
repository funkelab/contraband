import sys
import os

import gunpowder as gp
import logging
import math
import torch
import zarr
from pipelines.utils import Blur, InspectBatch, RemoveChannelDim, AddRandomPoints, PrepareBatch, AddSpatialDim, SetDtype, AddChannelDim, RemoveSpatialDim
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from pipelines.contrastive_loss import contrastive_volume_loss

logging.basicConfig(level=logging.INFO)


class Standard2DContrastive():

    def __init__(self, params, logdir, log_every=1):

        self.params = params
        self.logdir = logdir
        self.log_every = log_every

        temperature = 1.0

        def loss(emb_0, emb_1, locations_0, locations_1):
            return contrastive_volume_loss(
                emb_0,
                emb_1,
                locations_0,
                locations_1,
                temperature)
        self.training_loss = loss
        self.val_loss = torch.nn.MSELoss()

    def _make_train_augmentation_pipeline(self, raw, source):
        if 'elastic' in self.params and self.params['elastic']:
            source = source + gp.ElasticAugment(
                control_point_spacing=(1, 10, 10),
                jitter_sigma=(0, 0.1, 0.1),
                rotation_interval=(0, math.pi/2))

        if 'blur' in self.params and self.params['blur']:
            source = source + Blur(raw, sigma=[0, 1, 1])

        if 'simple' in self.params and self.params['simple']:
            source = source + gp.SimpleAugment(
                mirror_only=(1, 2),
                transpose_only=(1, 2)) 

        if 'noise' in self.params and self.params['noise']:
            source = source + gp.NoiseAugment(raw, var=0.01)
        return source

    def create_train_pipeline(self, model):

        optimizer = self.params['optimizer'](model.parameters(), 
                                                   **self.params['optimizer_kwargs'])

        filename = 'data/ctc/Fluo-N2DH-SIM+.zarr'

        raw_0 = gp.ArrayKey('RAW_0')
        points_0 = gp.GraphKey('POINTS_0')
        locations_0 = gp.ArrayKey('LOCATIONS_0')
        emb_0 = gp.ArrayKey('EMBEDDING_0')
        raw_1 = gp.ArrayKey('RAW_1')
        points_1 = gp.GraphKey('POINTS_1')
        locations_1 = gp.ArrayKey('LOCATIONS_1')
        emb_1 = gp.ArrayKey('EMBEDDING_1')

        request = gp.BatchRequest()
        request.add(raw_0, (1, 260, 260))
        request.add(raw_1, (1, 260, 260))
        request.add(points_0, (1, 168, 168))
        request.add(points_1, (1, 168, 168))
        request[locations_0] = gp.ArraySpec(nonspatial=True)
        request[locations_1] = gp.ArraySpec(nonspatial=True)

        snapshot_request = gp.BatchRequest()
        snapshot_request[emb_0] = gp.ArraySpec(roi=request[points_0].roi)
        snapshot_request[emb_1] = gp.ArraySpec(roi=request[points_1].roi)

        source_shape = zarr.open(filename)['train/raw'].shape
        raw_roi = gp.Roi((0, 0, 0), source_shape)

        sources = tuple(
            gp.ZarrSource(
                filename,
                {
                    raw: 'train/raw'
                },
                # fake 3D data
                array_specs={
                    raw: gp.ArraySpec(
                        roi=raw_roi,
                        voxel_size=(1, 1, 1),
                        interpolatable=True)
                }) +
            gp.Normalize(raw, factor=1.0/4) +
            gp.Pad(raw, (0, 200, 200)) +
            AddRandomPoints(points, for_array=raw, density=0.0005) 

            for raw, points in zip([raw_0, raw_1], [points_0, points_1])
        )
        sources = tuple(
            self._make_train_augmentation_pipeline(raw, source) 
                for raw, source in zip([raw_0, raw_1], sources)
        )

        pipeline = (
            sources +
            gp.MergeProvider() +
            gp.Crop(raw_0, raw_roi) +
            gp.RandomLocation() +
            PrepareBatch(
                raw_0, raw_1,
                points_0, points_1,
                locations_0, locations_1) +
            gp.PreCache() +
            gp.torch.Train(
                model, self.training_loss, optimizer,
                inputs={
                    'raw_0': raw_0,
                    'raw_1': raw_1
                },
                loss_inputs={
                    'emb_0': emb_0,
                    'emb_1': emb_1,
                    'locations_0': locations_0,
                    'locations_1': locations_1
                },
                outputs={
                    2: emb_0,
                    3: emb_1
                },
                array_specs={
                    emb_0: gp.ArraySpec(voxel_size=(1, 1)),
                    emb_1: gp.ArraySpec(voxel_size=(1, 1))
                },
                checkpoint_basename=self.logdir + '/checkpoints/model',
                save_every=1,
                log_dir=self.logdir,
                log_every=self.log_every) +
            # everything is 3D, except emb_0 and emb_1
            AddSpatialDim(emb_0) +
            AddSpatialDim(emb_1) +
            # now everything is 3D
            RemoveChannelDim(raw_0) +
            RemoveChannelDim(raw_1) +
            RemoveChannelDim(emb_0) +
            RemoveChannelDim(emb_1) +
            gp.Snapshot(
                output_dir=self.logdir + '/snapshots/train',
                output_filename='it{iteration}.hdf',
                dataset_names={
                    raw_0: 'raw_0',
                    raw_1: 'raw_1',
                    points_0: 'points_0',
                    points_1: 'points_1',
                    emb_0: 'emb_0',
                    emb_1: 'emb_1'
                },
                additional_request=snapshot_request,
                every=500) +
            gp.PrintProfilingStats(every=10)
        )

        return pipeline, request

