import os
from contraband.post_processing.agglomerate import agglomerate
from contraband.pipelines.prediction import Predict
import numpy as np
import waterz
import daisy
import pandas as pd


def insert_dim(a, s, dim=0):
    return a[:dim] + (s, ) + a[dim:]


def save_samples(samples, labels, labels_dataset, thresholds, curr_log_dir, checkpoint):
    # [samples, thresholds, ....)
    for sample in range(len(samples)):
        seg = daisy.prepare_ds(os.path.join(curr_log_dir, 
                               'samples/sample_' + str(checkpoint) + '.zarr'),
                               ds_name='segmentation_' + str(sample),
                               total_roi=labels_dataset.roi,
                               voxel_size=labels_dataset.voxel_size,
                               dtype=labels_dataset.dtype,
                               num_channels=len(thresholds))

        gt = daisy.prepare_ds(os.path.join(curr_log_dir,
                              'samples/sample_' + str(checkpoint) + '.zarr'),
                              ds_name='gt_' + str(sample),
                              total_roi=labels_dataset.roi,
                              voxel_size=labels_dataset.voxel_size,
                              dtype=labels_dataset.dtype,
                              num_channels=len(thresholds))
        
        seg.data[:] = np.squeeze(samples[sample])
        gt.data[:] = np.squeeze(np.broadcast_to(labels[sample], samples[sample].shape))


def validate(model, pipeline, data_file, dataset, curr_log_dir, thresholds, checkpoint):
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

    pred_aff = pipeline.get_predictions()
    labels_dataset = daisy.open_ds(data_file, dataset['gt'], 'r')

    labels = np.array(labels_dataset.data).astype(np.uint64)
    samples = np.zeros((2, len(thresholds), *labels.shape[1:]))
    metrics = {threshold: {'voi_split': [], 'voi_merge': [], 'rand_split': [], 'rand_merge': []}
               for threshold in thresholds}
    for i in range(pred_aff.shape[0]):
        curr_segmentation = agglomerate(pred_aff[i],
                                        thresholds=thresholds, is_2d=True)

        threshold = 0
        for segmentation, label in zip(curr_segmentation, labels[i]):
            curr_segmentation = segmentation
            # Save required number of samples
            if i < samples.shape[0]:
                samples[i] = curr_segmentation

            print("Agglomeration: ", curr_segmentation.shape)
            # print(segmentation)
            print(np.isnan(curr_segmentation).any())
            print("Labels: ", labels[i].shape)
            # print(labels)
            print(np.isnan(labels).any())

            curr_metrics = waterz.evaluate(curr_segmentation, labels[i, np.newaxis])
            print("metrics", curr_metrics)
            for key in metrics[thresholds[threshold]].keys():
                metrics[thresholds[threshold]][key].append(curr_metrics[key])

            threshold += 1

    averaged_metrics = {threshold: {metric: sum(total) / labels.shape[0]
                        for metric, total in metrics_per_threshold.items()}
                        for threshold, metrics_per_threshold in metrics.items()}
    print("averaged_metrics: ", averaged_metrics)

    for threshold in thresholds:
        metrics_file = os.path.join(curr_log_dir, 'metrics_' + str(threshold) + '.csv')
        if os.path.isfile(metrics_file):
            prev_metrics = pd.read_csv(metrics_file, index_col=0)
            print(prev_metrics.columns)
            print("test: ", list(averaged_metrics[threshold].values()))
            prev_metrics.loc[checkpoint] = list(averaged_metrics[threshold].values())
            prev_metrics.to_csv(metrics_file) 
            print('prev_metrics', prev_metrics)
        else: 
            temp = pd.DataFrame([averaged_metrics[threshold]], index=[checkpoint]).to_csv(
                metrics_file) 
            print("temp: ", temp)
    
    save_samples(samples, labels, labels_dataset, thresholds, curr_log_dir, checkpoint)
