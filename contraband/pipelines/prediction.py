import gunpowder as gp
import os
import zarr
from contraband.pipelines.utils import AddSpatialDim, AddChannelDim, RemoveSpatialDim, RemoveChannelDim, InspectBatch
import numpy as np
import daisy

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Predict():

    def __init__(self, model, params, curr_log_dir):
        self.data_file = params['data_file']
        self.dataset = params['dataset']['validate']['raw']

        self.params = params
        self.model = model
        self.curr_log_dir = curr_log_dir

    def get_predictions(self):

        pipeline, request, predictions = self.make_pipeline()

        with gp.build(pipeline):
            try:
                os.remove(os.path.join(self.curr_log_dir, 'predictions.zarr'))
            except OSError as e:
              pass

            pipeline.request_batch(gp.BatchRequest())
            f = daisy.open_ds(os.path.join(self.curr_log_dir, 'predictions.zarr'), 
                              "predictions")
            return f 

    def make_pipeline(self):
        raw = gp.ArrayKey('RAW')
        pred_affs = gp.ArrayKey('PREDICTIONS')

        source_shape = zarr.open(self.data_file)[self.dataset].shape
        raw_roi = gp.Roi(np.zeros(len(source_shape[1:])), source_shape[1:])

        data = daisy.open_ds(self.data_file, self.dataset)
        source_roi = gp.Roi(data.roi.get_offset(), data.roi.get_shape())
        voxel_size = gp.Coordinate(data.voxel_size)
        
        # Get in and out shape
        in_shape = gp.Coordinate(self.model.in_shape)
        out_shape = gp.Coordinate(self.model.out_shape[2:])

        is_2d = in_shape.dims() == 2

        in_shape = in_shape * voxel_size
        out_shape = out_shape * voxel_size

        logger.info(f"source roi: {source_roi}")
        logger.info(f"in_shape: {in_shape}")
        logger.info(f"out_shape: {out_shape}")
        logger.info(f"voxel_size: {voxel_size}")

        request = gp.BatchRequest()
        request.add(raw, in_shape)
        request.add(pred_affs, out_shape)
        
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
                        interpolatable=True)
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
                    0: pred_affs
                }, 
                array_specs={
                    pred_affs: gp.ArraySpec(roi=raw_roi)
                }
            ))  
        
        pipeline = (
            pipeline + 
            gp.ZarrWrite(
                {
                    pred_affs: 'predictions',
                },
                output_dir=self.curr_log_dir,
                output_filename='predictions.zarr',
                compression_type='gzip'
            ) +
            gp.Scan(request)
        )

        return pipeline, request, pred_affs
