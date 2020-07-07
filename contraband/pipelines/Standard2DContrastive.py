import gunpowder as gp
import logging
import math
import torch
import zarr
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
    RandomPointGenerator)
from contraband.pipelines.contrastive_loss import contrastive_volume_loss

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
            source = source + gp.ElasticAugment(**self.params["elastic_params"])
            #source = source + gp.ElasticAugment(
            #    control_point_spacing=(1, 10, 10),
            #    jitter_sigma=(0, 0.1, 0.1),
            #    rotation_interval=(0, math.pi / 2))

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
        dataset = self.params['dataset']

        raw_0 = gp.ArrayKey('RAW_0')
        points_0 = gp.GraphKey('POINTS_0')
        locations_0 = gp.ArrayKey('LOCATIONS_0')
        emb_0 = gp.ArrayKey('EMBEDDING_0')
        raw_1 = gp.ArrayKey('RAW_1')
        points_1 = gp.GraphKey('POINTS_1')
        locations_1 = gp.ArrayKey('LOCATIONS_1')
        emb_1 = gp.ArrayKey('EMBEDDING_1')

        in_shape = gp.Coordinate((1, *model.in_shape))
        out_shape = gp.Coordinate((1, *(model.out_shape)[2:]))
        print("in_shape: ", in_shape)
        print("out_shape: ", out_shape)

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

        source_shape = zarr.open(filename)[dataset].shape
        raw_roi = gp.Roi((0, 0, 0), source_shape)

        random_point_generator = RandomPointGenerator(
            density=self.params['point_density'], repetitions=2)

        array_sources = tuple(
            gp.ZarrSource(
                filename,
                {
                    raw: dataset 
                },
                # fake 3D data
                array_specs={
                    raw: gp.ArraySpec(
                        roi=raw_roi,
                        voxel_size=(1, 1, 1),
                        interpolatable=True)
                }) +
            gp.Normalize(raw, self.params['norm_factor']) +
            gp.Pad(raw, None) 
            # AddRandomPoints(points, for_array=raw, density=0.0005) 

            for raw in [raw_0, raw_1]
        )

        point_sources = tuple((
            RandomPointSource(
                points_0,
                random_point_generator=random_point_generator),
            RandomPointSource(
                points_1,
                random_point_generator=random_point_generator)
            ))

        sources = tuple(tuple((raw, points)) for raw, points in zip(array_sources, point_sources))
        sources = tuple(
            source + 
            gp.MergeProvider()
            for source in sources)

        sources = tuple(
            self._make_train_augmentation_pipeline(raw, source) 
            for raw, source in zip([raw_0, raw_1], sources)
        )
        
        pipeline = (
            sources +
            gp.MergeProvider() +
            gp.Crop(raw_0, raw_roi) +
            gp.Crop(raw_1, raw_roi) +
            gp.RandomLocation() +
            PrepareBatch(
                raw_0, raw_1,
                points_0, points_1,
                locations_0, locations_1) +
            RejectArray(ensure_nonempty=locations_0) + 
            RejectArray(ensure_nonempty=locations_1) + 
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
                checkpoint_basename=self.logdir + '/contrastive/checkpoints/model',
                save_every=self.params['save_every'],
                log_dir=self.logdir + "/contrastive",
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
                output_dir=self.logdir + '/contrastive/snapshots',
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
                every=self.params['save_every']) +
            gp.PrintProfilingStats(every=10)
        )

        return pipeline, request

