import h5py
import gunpowder as gp
import os
import zarr
from contraband.pipelines.utils import AddChannelDim, RemoveSpatialDim, RemoveChannelDim
import numpy as np
import daisy


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
            #pipeline.request_batch(gp.BatchRequest())
            f = daisy.open_ds(os.path.join(self.curr_log_dir, 'predictions.hdf'),
                              os.path.join(self.curr_log_dir, "predictions"))
            return f 

    def make_pipeline(self):
        raw = gp.ArrayKey('RAW')
        pred_affs = gp.ArrayKey('PREDICTIONS')

        source_shape = zarr.open(self.data_file)[self.dataset].shape
        raw_roi = gp.Roi(np.zeros(len(source_shape[1:])), source_shape[1:])
        print(raw_roi)

        print(self.model.in_shape)
        print("Out channels: ", list(self.model.out_shape)[-raw_roi.dims():])
        input_size = gp.Coordinate(self.model.in_shape)
        output_size = gp.Coordinate(list(self.model.out_shape)[-raw_roi.dims():])

        request = gp.BatchRequest()
        request.add(raw, input_size)
        request.add(pred_affs, output_size)

        context = (input_size - output_size) / 2

        source = (
            gp.ZarrSource(
                self.data_file,
                {
                    raw: self.dataset,
                },
                array_specs={
                    raw: gp.ArraySpec(
                        roi=raw_roi,
                        interpolatable=True)
                }
            ) +
            AddChannelDim(raw, axis=1)

            # raw      : (c=1, roi)
        )

        with gp.build(source):
            raw_roi = source.spec[raw].roi 
            print("raw_roi: ", raw_roi)
            print("source_shape: ", source_shape)
            print(context)

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
                    pred_affs: gp.ArraySpec(roi=raw_roi),
                }
            ) + 

            gp.Hdf5Write(
                {
                    pred_affs: os.path.join(self.curr_log_dir, 
                                            'predictions'),
                },
                output_dir=self.curr_log_dir,
                output_filename='predictions.hdf',
                compression_type='gzip'
            ) +
            gp.Scan(request)
        )

        return pipeline, request, pred_affs
