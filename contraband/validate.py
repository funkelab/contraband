import os
from contraband.post_processing.agglomerate import agglomerate
from contraband.pipelines.prediction import Predict
import numpy as np
import waterz
import zarr


def validate(model, data_file, dataset, curr_log_dir, thresholds):
    """ Preforms validation over the whole test dataset.
    Args:
        
        model (SegmentationVolumeNet):
            The segmentation model to use for validation.
        
        data_file (``str``) 
            The path to the zarr file

        dataset (``dict``):
            Dictionary containing the validation dataset paths from the 
            param_dict.
        
        curr_log_dir (``str``):
            The path to the current expirement and param combination. 

        thresholds (``list``): 
            The thresholds to use for the agglomeration. 
    """

    predict = Predict(model, data_file, dataset['raw'], curr_log_dir)
    
    pred_aff = predict.get_predictions()
    labels = np.array(zarr.open(data_file, 'r')[dataset['gt']]).astype(np.uint64)

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
    # sample.create_group('sample')
    seg = sample.create_dataset('segmentation', shape=curr_segmentation.shape,
                                chunks=(100, 100), dtype='i4')
    seg.data = curr_segmentation
    gt = sample.create_dataset('gt', shape=labels[0].shape, chunks=(100, 100), dtype='i4')
    gt.data = labels[0]

