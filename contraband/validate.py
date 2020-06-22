import os
from contraband.post_processing.agglomerate import agglomerate
import numpy as np
import waterz
import zarr
from contraband.pipelines.prediction import Predict


def validate(model, pipeline, dataset, curr_log_dir, thresholds):

    predict = Predict(model, dataset, curr_log_dir)
    
    pred_aff = predict.get_predictions()
    labels = np.array(zarr.open(dataset, 'r')['validate/gt']).astype(np.uint64)

    for i in range(pred_aff.shape[0]):
        curr_segmentation = agglomerate(pred_aff[i],
                                        thresholds=thresholds, is_2d=True)
        curr_segmentation = list(curr_segmentation)[0]

        print("Agglomeration: ", curr_segmentation.shape)
        # print(segmentation)
        print(np.isnan(curr_segmentation).any())
        print("Labels: ", labels[i].shape)
        # print(labels)
        print(np.isnan(labels).any())

        metrics = waterz.evaluate(curr_segmentation, labels[i, np.newaxis])
        print("metrics", metrics)

    # Save a sample
    curr_segmentation = agglomerate(pred_aff[0],
                                    thresholds=thresholds, is_2d=True)
    curr_segmentation = list(curr_segmentation)[0][0]

    sample = zarr.open_group(os.path.join(curr_log_dir, 'seg/sample.zarr'))
    sample.create_group('sample')
    seg = sample.create_dataset('segmentation', shape=curr_segmentation.shape,
                                chunks=(100, 100), dtype='i4')
    seg.data = curr_segmentation
    gt = sample.create_dataset('gt', shape=labels[0].shape, chunks=(100, 100), dtype='i4')
    gt.data = labels[0]

