import os
from contraband.post_processing.agglomerate import agglomerate
from contraband.pipelines.prediction import Predict
import numpy as np
import waterz
import daisy
import pandas as pd
from contraband import utils
import logging

logger = logging.getLogger(__name__)


def insert_dim(a, s, dim=0):
    return a[:dim] + (s, ) + a[dim:]


def save_samples(pred_affs, 
                 pred_affs_ds,
                 segmentation,
                 labels, 
                 labels_dataset,
                 fragments,
                 boundarys,
                 distances,
                 seeds,
                 history,
                 threshold,
                 curr_log_dir,
                 checkpoint,
                 is_2d):
    voxel_size = labels_dataset.voxel_size 
    roi = labels_dataset.roi
    if is_2d:
        voxel_size = (1, *voxel_size)
        roi = daisy.Roi((0, *roi.get_offset()), (1, *roi.get_shape()))
        labels = labels[np.newaxis]
        segmentation = segmentation[np.newaxis]

    zarr_file = os.path.join(curr_log_dir, 
                             f'samples/sample_{checkpoint}_thresh' +
                             f'_{threshold}.zarr')

    seg = daisy.prepare_ds(zarr_file,
                           ds_name='segmentation',
                           total_roi=roi,
                           voxel_size=voxel_size,
                           dtype=np.uint64,
                           num_channels=1)

    gt = daisy.prepare_ds(zarr_file,
                          ds_name='gt',
                          total_roi=roi,
                          voxel_size=voxel_size,
                          dtype=np.uint64,
                          num_channels=1)

    pred = daisy.prepare_ds(zarr_file,
                            ds_name='affs',
                            total_roi=pred_affs_ds.roi,
                            voxel_size=pred_affs_ds.voxel_size,
                            dtype=pred_affs_ds.dtype,
                            num_channels=pred_affs.shape[0])

    utils.save_zarr(fragments,
                    zarr_file,
                    ds='fragment',
                    roi=fragments.shape,
                    voxel_size=voxel_size,
                    fit_voxel=True) 

    utils.save_zarr(boundarys,
                    zarr_file,
                    ds='boundary',
                    roi=boundarys.shape,
                    voxel_size=voxel_size,
                    fit_voxel=True) 

    utils.save_zarr(distances,
                    zarr_file,
                    ds='dist_trfm',
                    roi=distances.shape,
                    voxel_size=voxel_size,
                    fit_voxel=True) 

    utils.save_zarr(seeds,
                    zarr_file,
                    ds='seeds',
                    roi=seeds.shape,
                    voxel_size=voxel_size,
                    fit_voxel=True) 
    
    seg.data[:] = segmentation
    gt.data[:] = labels
    pred.data[:] = pred_affs

    history = [merge for thresh in history for merge in thresh]
    history = pd.DataFrame(history)
    history.to_csv(os.path.join(curr_log_dir, 
                                f'samples/sample_{checkpoint}_hist' +
                                f'_{threshold}.csv'))
        


def validate(model, pipeline, data_file, dataset, curr_log_dir, thresholds, checkpoint, has_background):
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

        has_background (``bool``):
            If the data to be validated on has background to be predicted.
    """

    pred_aff_ds = pipeline.get_predictions()
    is_2d = pred_aff_ds.roi.dims() == 2
    pred_aff = np.array(pred_aff_ds.data)

    num_channels = pred_aff.shape[0]

    labels_dataset = daisy.open_ds(data_file, dataset['gt'], 'r')
    labels = np.array(labels_dataset.data).astype(np.uint64)
   
    logging.info("Thresholds: {thresholds}")
    metrics = {threshold: {'voi_split': [], 
                           'voi_merge': [], 
                           'rand_split': [],
                           'rand_merge': [],
                           'voi_sum': []}
               for threshold in thresholds}
    
    if num_channels == 1:
        labels = labels[np.newaxis]
    # pick large value to start
    for channel in range(num_channels):
        curr_segmentation, fragments, boundary, distance, seeds = \
            agglomerate(pred_aff[channel],
                        thresholds=thresholds,
                        is_2d=is_2d,
                        has_background=has_background)

        threshold = 0
        for segmentation in curr_segmentation:
            
            logger.info(f"Agglomeration: {segmentation[0].shape}")

            logger.info(f"Labels: {labels[channel].shape}")
            label = labels[channel]
            if len(label.shape) == 2:
                label = label[np.newaxis]

            curr_metrics = waterz.evaluate(segmentation[0], label)
            curr_metrics['voi_sum'] = curr_metrics['voi_split'] + curr_metrics['voi_merge']
            logger.info(f"metrics{curr_metrics}")
            for key in metrics[thresholds[threshold]].keys():
                metrics[thresholds[threshold]][key].append(curr_metrics[key])
        
            threshold += 1

    averaged_metrics = {threshold: {metric: sum(total) / labels.shape[0]
                        for metric, total in metrics_per_threshold.items()}
                        for threshold, metrics_per_threshold in metrics.items()}
    logger.info(f"averaged_metrics: {averaged_metrics}")
    
    # Get sample from best threshold
    best_index = np.argmin([metric['voi_sum'] for metric in averaged_metrics.values()])
    best_threshold = thresholds[best_index]
    print(f'best_index: {best_index}') 
    print(f"best_thresh: {best_threshold}")

    sample_channel = 0
    segmentation, fragments, best_boundary, best_dist, best_seeds = \
        agglomerate(pred_aff[sample_channel],
                    thresholds=[best_threshold],
                    is_2d=is_2d,
                    has_background=has_background)
    best_fragments = fragments.copy()
    # Should only happen once
    for seg in segmentation:
        best_seg = seg[0][sample_channel]
        best_history = seg[1]

    os.makedirs(os.path.join(curr_log_dir, "metrics"), exist_ok=True)
    metrics_file = os.path.join(curr_log_dir, "metrics", 'metrics_' + checkpoint + '.csv')
    pd.DataFrame.from_dict(averaged_metrics, orient='index').to_csv(metrics_file)
    
    save_samples(pred_aff[sample_channel],
                 pred_aff_ds,
                 best_seg,
                 labels[sample_channel],
                 labels_dataset,
                 best_fragments,
                 best_boundary,
                 best_dist,
                 best_seeds,
                 best_history,
                 best_threshold,
                 curr_log_dir,
                 checkpoint,
                 is_2d)
