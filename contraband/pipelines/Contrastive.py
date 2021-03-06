import gunpowder as gp
import logging
import torch
from contraband.pipelines.utils import (
    Blur, 
    InspectBatch,
    RemoveChannelDim,
    RandomPointSource,
    PrepareBatch,
    AddSpatialDim,
    SetDtype,
    AddChannelDim,
    RemoveSpatialDim,
    RejectArray,
    RandomPointGenerator,
    RandomSourceGenerator,
    RandomMultiBranchSource)
from contraband.pipelines.contrastive_loss import ContrastiveVolumeLoss
import daisy
import numpy as np

logger = logging.getLogger(__name__)


class Contrastive():

    def __init__(self, params, logdir, log_every=1):

        self.params = params
        self.logdir = logdir
        self.log_every = log_every
        self.val_loss = torch.nn.MSELoss()

    def _make_train_augmentation_pipeline(self, raw, source):
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

    def create_train_pipeline(self, model):

        optimizer = self.params['optimizer'](model.parameters(), 
                                                   **self.params['optimizer_kwargs'])

        filename = self.params['data_file']
        datasets = self.params['dataset']

        raw_0 = gp.ArrayKey('RAW_0')
        points_0 = gp.GraphKey('POINTS_0')
        locations_0 = gp.ArrayKey('LOCATIONS_0')
        emb_0 = gp.ArrayKey('EMBEDDING_0')
        raw_1 = gp.ArrayKey('RAW_1')
        points_1 = gp.GraphKey('POINTS_1')
        locations_1 = gp.ArrayKey('LOCATIONS_1')
        emb_1 = gp.ArrayKey('EMBEDDING_1')

        data = daisy.open_ds(filename, datasets[0])
        source_roi = gp.Roi(data.roi.get_offset(), data.roi.get_shape())
        voxel_size = gp.Coordinate(data.voxel_size)
        
        # Get in and out shape
        in_shape = gp.Coordinate(model.in_shape)
        out_shape = gp.Coordinate(model.out_shape[2:])
        is_2d = in_shape.dims() == 2

        emb_voxel_size = voxel_size

        cv_loss = ContrastiveVolumeLoss(self.params['temperature'],
                                        self.params['point_density'],
                                        out_shape * voxel_size)

        # Add fake 3rd dim 
        if is_2d: 
            in_shape = gp.Coordinate((1, *in_shape))
            out_shape = gp.Coordinate((1, *out_shape))
            voxel_size = gp.Coordinate((1, *voxel_size))
            source_roi = gp.Roi((0, *source_roi.get_offset()), 
                                (data.shape[0], *source_roi.get_shape()))

        in_shape = in_shape * voxel_size
        out_shape = out_shape * voxel_size
        
        logger.info(f"source roi: {source_roi}")
        logger.info(f"in_shape: {in_shape}")
        logger.info(f"out_shape: {out_shape}")
        logger.info(f"voxel_size: {voxel_size}")


        request = gp.BatchRequest()
        request.add(raw_0, in_shape)
        request.add(raw_1, in_shape)
        request.add(points_0, out_shape)
        request.add(points_1, out_shape)
        request[locations_0] = gp.ArraySpec(nonspatial=True)
        request[locations_1] = gp.ArraySpec(nonspatial=True)

        snapshot_request = gp.BatchRequest()
        snapshot_request[emb_0] = gp.ArraySpec(roi=request[points_0].roi)
        snapshot_request[emb_1] = gp.ArraySpec(roi=request[points_1].roi)

        random_point_generator = RandomPointGenerator(
            density=self.params['point_density'], repetitions=2)
        
        # Use volume to calculate probabilities, RandomSourceGenerator will
        # normalize volumes to probablilties
        probabilities = np.array([np.product(daisy.open_ds(filename, dataset).shape) 
                                  for dataset in datasets])
        random_source_generator = RandomSourceGenerator(num_sources=len(datasets), 
                                                        probabilities=probabilities,
                                                        repetitions=2)

        array_sources = tuple(
            tuple(
                gp.ZarrSource(
                    filename,
                    {
                        raw: dataset 
                    },
                    # fake 3D data
                    array_specs={
                        raw: gp.ArraySpec(
                            roi=source_roi,
                            voxel_size=voxel_size,
                            interpolatable=True)
                    })

                for dataset in datasets
            )
            for raw in [raw_0, raw_1]
        )

        # Choose a random dataset to pull from
        array_sources = \
            tuple(arrays +
                  RandomMultiBranchSource(random_source_generator) + 
                  gp.Normalize(raw, self.params['norm_factor']) +
                  gp.Pad(raw, None)
                  for raw, arrays 
                  in zip([raw_0, raw_1], array_sources))

        point_sources = tuple((
            RandomPointSource(
                points_0,
                random_point_generator=random_point_generator),
            RandomPointSource(
                points_1,
                random_point_generator=random_point_generator)
        ))
        
        # Merge the point and array sources together. 
        # There is one array and point source per branch.
        sources = tuple(
            (array_source, point_source) +
            gp.MergeProvider()
            for array_source, point_source in zip(array_sources, point_sources))
    
        sources = tuple(
            self._make_train_augmentation_pipeline(raw, source) 
            for raw, source in zip([raw_0, raw_1], sources)
        )
        
        pipeline = (
            sources +
            gp.MergeProvider() +
            gp.Crop(raw_0, source_roi) +
            gp.Crop(raw_1, source_roi) +
            gp.RandomLocation() +
            PrepareBatch(
                raw_0, raw_1,
                points_0, points_1,
                locations_0, locations_1,
                is_2d) +
            RejectArray(ensure_nonempty=locations_0) + 
            RejectArray(ensure_nonempty=locations_1))

        if not is_2d:
            pipeline = (
                pipeline +
                AddChannelDim(raw_0) + 
                AddChannelDim(raw_1)
            )

        pipeline = (
            pipeline + 
            gp.PreCache() +
            gp.torch.Train(
                model, cv_loss, optimizer,
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
                    emb_0: gp.ArraySpec(voxel_size=emb_voxel_size),
                    emb_1: gp.ArraySpec(voxel_size=emb_voxel_size)
                },
                checkpoint_basename=self.logdir + '/contrastive/checkpoints/model',
                save_every=self.params['save_every'],
                log_dir=self.logdir + "/contrastive",
                log_every=self.log_every))

        if is_2d:
            pipeline = (
                pipeline + 
                # everything is 3D, except emb_0 and emb_1
                AddSpatialDim(emb_0) +
                AddSpatialDim(emb_1)
            )

        pipeline = (
            pipeline + 
            # now everything is 3D
            RemoveChannelDim(raw_0) +
            RemoveChannelDim(raw_1) +
            RemoveChannelDim(emb_0) +
            RemoveChannelDim(emb_1) +
            gp.Snapshot(
                output_dir=self.logdir + '/contrastive/snapshots',
                output_filename='it{iteration}.hdf',
                dataset_names={
                    raw_0: 'raw_0',
                    raw_1: 'raw_1',
                    locations_0: 'locations_0',
                    locations_1: 'locations_1',
                    emb_0: 'emb_0',
                    emb_1: 'emb_1'
                },
                additional_request=snapshot_request,
                every=self.params['save_every']) +
            gp.PrintProfilingStats(every=500)
        )

        return pipeline, request

