import h5py
import gunpowder as gp
import os
import zarr
from contraband.pipelines.utils import AddChannelDim, RemoveSpatialDim, RemoveChannelDim
import numpy as np


class Predict():

    def __init__(self, model, dataset, curr_log_dir):
        self.dataset = dataset
        self.model = model
        self.curr_log_dir = curr_log_dir

    def get_predictions(self):

        pipeline, request, predictions = self.make_pipeline()

        with gp.build(pipeline):
            pipeline.request_batch(gp.BatchRequest())
            with h5py.File(os.path.join(self.curr_log_dir,
                                        'seg/predictions.hdf'), 'r') as f:

                print("datasets: ", list(f.keys()))
                pred_affs = f[os.path.join(self.curr_log_dir, "seg/pred_affs")]
                return np.array(pred_affs)

    def make_pipeline(self):
        raw = gp.ArrayKey('RAW')
        pred_affs = gp.ArrayKey('PREDICTIONS')
        
        input_size = gp.Coordinate((260, 260))
        output_size = gp.Coordinate((168, 168))

        request = gp.BatchRequest()
        request.add(raw, input_size)
        request.add(pred_affs, output_size)

        source_shape = zarr.open(self.dataset)['validate/raw'].shape
        raw_roi = gp.Roi((0, 0), source_shape[1:])
        print(raw_roi)

        context = (input_size - output_size) / 2

        source = (
            gp.ZarrSource(
                self.dataset,
                {
                    raw: 'validate/raw',
                },
                array_specs={
                    raw: gp.ArraySpec(
                        roi=raw_roi,
                        voxel_size=(1, 1),
                        interpolatable=True)
                }
            ) +
            AddChannelDim(raw, axis=1)

            # RemoveSpatialDim(raw)
            # raw      : (c=1, l=1, h, w)
        )

        with gp.build(source):
            raw_roi = source.spec[raw].roi 
            print("raw_roi: ", raw_roi)
            print("source_shape: ", source_shape)
            print(context)

        pipeline = (
            source +

            gp.Normalize(raw, factor=1.0 / 4) +
            gp.Pad(raw, context) +
            # raw      : (l=1, h, w)
            # raw      : (c=1, h, w)
            # gt_aff   : (c=2, h, w)
            # InspectBatch('before stack:') +
            gp.PreCache() +
            # raw      : (b, c=1, h, w)
            # gt_aff   : (b, c=2, h, w)
            # (which is what train requires)
            gp.torch.Predict(
                self.model,
                        
                inputs={
                    'raw': raw
                },
                outputs={
                    0: pred_affs
                },
                array_specs={
                    pred_affs: gp.ArraySpec(roi=raw_roi,
                                            voxel_size=(1, 1)),
                }
            ) + 

            gp.Hdf5Write(
                {
                    pred_affs: os.path.join(self.curr_log_dir, 
                                            'seg/pred_affs'),
                },
                output_dir=os.path.join(self.curr_log_dir, 'seg'),
                output_filename='predictions.hdf',
                compression_type='gzip'
            ) +
            gp.Scan(request)
        )

        return pipeline, request, pred_affs
