import gunpowder as gp
import logging
import math
import torch
import daisy
from contraband.pipelines.utils import (
    Blur, 
    InspectBatch,
    RemoveChannelDim,
    AddRandomPoints,
    PrepareBatch,
    AddSpatialDim,
    SetDtype,
    AddChannelDim,
    RemoveSpatialDim)
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
            print('elastic: ', self.params['elastic'])
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

        data = daisy.open_ds(filename, dataset)
        source_shape = gp.Roi((0,0,0), tuple(data.shape))
        source_roi = gp.Roi(data.roi.get_offset(), data.roi.get_shape())
        voxel_size = gp.Coordinate(data.voxel_size)

        in_shape = gp.Coordinate(model.in_shape)
        out_shape = gp.Coordinate((model.out_shape)[2:])

        in_shape = in_shape * voxel_size
        out_shape = out_shape * voxel_size

        is_2d = source_roi.dims() == 2
        if is_2d:
            source_roi = gp.Roi((0, *source_roi.get_offset()),
                                (1, *source_roi.get_shape()))
            voxel_size = gp.Coordinate((1, *voxel_size))
            in_shape = gp.Coordinate((1, *in_shape))
            out_shape = gp.Coordinate((1, *out_shape))


        context = (in_shape - out_shape) / 2

        print("sources_shape: ", source_shape)
        print("voxel_size: ", voxel_size)
        print("source_roi: ", source_roi)
        print("in_shape: ", in_shape)
        print("out_shape: ", out_shape)
        print("context: ", context)


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

        sources = tuple(
            gp.ZarrSource(
                filename,
                {
                    raw: dataset 
                },
                # fake 3D data
                array_specs={
                    raw: gp.ArraySpec(
                        roi=source_shape,
                        voxel_size=voxel_size,
                        interpolatable=True)
                }) +
            InspectBatch("Source: ") +
            gp.Normalize(raw, factor=self.params['norm_factor']) +
            gp.Pad(raw, context) +
            AddRandomPoints(points, for_array=raw, source_shape=source_shape.get_shape(), 
                            context=context, voxel_size=voxel_size, 
                            density=0.0005) 

            for raw, points in zip([raw_0, raw_1], [points_0, points_1])
        )
        sources = tuple(
            self._make_train_augmentation_pipeline(raw, source) 
            for raw, source in zip([raw_0, raw_1], sources)
        )
        if is_2d:
            #source_roi = source_roi[1:]
            voxel_size = voxel_size[1:]

        pipeline = (
            sources +
            gp.MergeProvider() +
            #gp.Crop(raw_0, source_shape) + 
            #gp.Crop(raw_0, source_shape) + 
            gp.RandomLocation() +
            #InspectBatch("Before preaprebatch: ") + 
            PrepareBatch(
                raw_0, raw_1,
                points_0, points_1,
                locations_0, locations_1) +
            #InspectBatch("Before reject: ") + 
            gp.Reject(ensure_nonempty=points_0) + 
            gp.Reject(ensure_nonempty=points_1) + 
            #gp.PreCache() +
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
                    emb_0: gp.ArraySpec(voxel_size=voxel_size),
                    emb_1: gp.ArraySpec(voxel_size=voxel_size)
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

