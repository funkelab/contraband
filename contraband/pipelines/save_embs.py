import gunpowder as gp
import shutil
import os
import zarr
from contraband.pipelines.utils import AddSpatialDim, AddChannelDim, RemoveSpatialDim, RemoveChannelDim, InspectBatch
import numpy as np
import daisy
import logging
logger = logging.getLogger(__name__)


class SaveEmbs():

    def __init__(self, model, params, dataset, data_file, curr_log_dir):
        """
            Saves the embeddings for each contrastive model.
        """
        self.data_file = data_file 
        self.dataset = dataset

        self.params = params
        self.model = model
        self.curr_log_dir = curr_log_dir

    def save_embs(self):

        pipeline, request, predictions = self.make_pipeline()

        with gp.build(pipeline):
            try:
                shutil.rmtree(os.path.join(self.curr_log_dir, self.dataset + '_embs.zarr'))
            except OSError as e:
                pass

            pipeline.request_batch(gp.BatchRequest())

    def make_pipeline(self):
        raw = gp.ArrayKey('RAW')
        embs = gp.ArrayKey('EMBS')

        source_shape = zarr.open(self.data_file)[self.dataset].shape
        raw_roi = gp.Roi(np.zeros(len(source_shape[1:])), source_shape[1:])

        data = daisy.open_ds(self.data_file, self.dataset)
        source_roi = gp.Roi(data.roi.get_offset(), data.roi.get_shape())
        voxel_size = gp.Coordinate(data.voxel_size)
        
        # Get in and out shape
        in_shape = gp.Coordinate(self.model.in_shape)
        out_shape = gp.Coordinate(self.model.out_shape[2:])

        is_2d = in_shape.dims() == 2

        logger.info(f"source roi: {source_roi}")
        logger.info(f"in_shape: {in_shape}")
        logger.info(f"out_shape: {out_shape}")
        logger.info(f"voxel_size: {voxel_size}")
        in_shape = in_shape * voxel_size
        out_shape = out_shape * voxel_size

        logger.info(f"source roi: {source_roi}")
        logger.info(f"in_shape: {in_shape}")
        logger.info(f"out_shape: {out_shape}")
        logger.info(f"voxel_size: {voxel_size}")

        request = gp.BatchRequest()
        request.add(raw, in_shape)
        request.add(embs, out_shape)
        
        context = (in_shape - out_shape) / 2

        source = (
            gp.ZarrSource(
                self.data_file,
                {
                    raw: self.dataset,
                },
                array_specs={
                    raw: gp.ArraySpec(
                        roi=source_roi,
                        interpolatable=False)
                }
            )
        ) 

        if is_2d:
            source = (
                source + 
                AddChannelDim(raw, axis=1)
            )
        else:
            source = (
                source + 
                AddChannelDim(raw, axis=0) +
                AddChannelDim(raw)
            )

        source = (
            source
            # raw      : (c=1, roi)
        )

        with gp.build(source):
            raw_roi = source.spec[raw].roi 
            logger.info(f"raw_roi: {raw_roi}")

        pipeline = (
            source +

            gp.Normalize(raw, factor=self.params['norm_factor']) +
            gp.Pad(raw, context) + 

            gp.PreCache() +

            gp.torch.Predict(
                self.model,
                        
                inputs={
                    'raw': raw
                },
                outputs={
                    0: embs
                }, 
                array_specs={
                    embs: gp.ArraySpec(roi=raw_roi)
                }
            ))  
        
        pipeline = (
            pipeline + 
            gp.ZarrWrite(
                {
                    embs: 'embs',
                },
                output_dir=self.curr_log_dir,
                output_filename=self.dataset + '_embs.zarr',
                compression_type='gzip'
            ) +
            gp.Scan(request)
        )

        return pipeline, request, embs
