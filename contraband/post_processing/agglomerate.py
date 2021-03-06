import numpy as np
import time
import waterz
from contraband.post_processing.watershed import watershed
from contraband import utils
import os
import pandas

scoring_functions = {
    'mean_aff':
    'OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>',
    'max_aff':
    'OneMinus<MaxAffinity<RegionGraphType, ScoreValue>>',
    'max_10':
    'OneMinus<MeanMaxKAffinity<RegionGraphType, 10, ScoreValue>>',

    # quantile merge functions, initialized with max affinity
    '15_aff_maxinit':
    'OneMinus<QuantileAffinity<RegionGraphType, 15, ScoreValue>>',
    '15_aff_maxinit_histograms':
    'OneMinus<HistogramQuantileAffinity<RegionGraphType, 15, ScoreValue, 256>>',
    '25_aff_maxinit':
    'OneMinus<QuantileAffinity<RegionGraphType, 25, ScoreValue>>',
    '25_aff_maxinit_histograms':
    'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256>>',
    'median_aff_maxinit':
    'OneMinus<QuantileAffinity<RegionGraphType, 50, ScoreValue>>',
    'median_aff_maxinit_histograms':
    'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>',
    '75_aff_maxinit':
    'OneMinus<QuantileAffinity<RegionGraphType, 75, ScoreValue>>',
    '75_aff_maxinit_histograms':
    'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256>>',
    '85_aff_maxinit':
    'OneMinus<QuantileAffinity<RegionGraphType, 85, ScoreValue>>',
    '85_aff_maxinit_histograms':
    'OneMinus<HistogramQuantileAffinity<RegionGraphType, 85, ScoreValue, 256>>',

    # quantile merge functions, initialized with quantile
    '15_aff':
    'OneMinus<QuantileAffinity<RegionGraphType, 15, ScoreValue, false>>',
    '15_aff_histograms':
    'OneMinus<HistogramQuantileAffinity<RegionGraphType, 15, ScoreValue, 256, false>>',
    '25_aff':
    'OneMinus<QuantileAffinity<RegionGraphType, 25, ScoreValue, false>>',
    '25_aff_histograms':
    'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>',
    'median_aff':
    'OneMinus<QuantileAffinity<RegionGraphType, 50, ScoreValue, false>>',
    'median_aff_histograms':
    'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, false>>',
    '75_aff':
    'OneMinus<QuantileAffinity<RegionGraphType, 75, ScoreValue, false>>',
    '75_aff_histograms':
    'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>',
    '85_aff':
    'OneMinus<QuantileAffinity<RegionGraphType, 85, ScoreValue, false>>',
    '85_aff_histograms':
    'OneMinus<HistogramQuantileAffinity<RegionGraphType, 85, ScoreValue, 256, false>>',
}


def agglomerate_with_waterz(affs,
                            thresholds,
                            histogram_quantiles=False,
                            discrete_queue=False,
                            merge_function='median_aff',
                            init_with_max=True,
                            return_merge_history=False,
                            return_region_graph=False,
                            has_background=None):

    print("Extracting initial fragments...")
    fragments, affs_xy, distances, seeds = watershed(affs, 'maxima_distance', has_background)

    print("Agglomerating with %s", merge_function)

    if init_with_max:
        merge_function += '_maxinit'
    if histogram_quantiles:
        merge_function += '_histograms'

    discretize_queue = 0
    if discrete_queue:
        discretize_queue = 256

    return ( 
        waterz.agglomerate(
            affs,
            thresholds,
            fragments=fragments,
            scoring_function=scoring_functions[merge_function],
            discretize_queue=discretize_queue,
            return_merge_history=return_merge_history,
            return_region_graph=return_region_graph), 
        fragments, 
        affs_xy, 
        distances,
        seeds
    )


def agglomerate(affs, thresholds, is_2d, has_background):

    thresholds = list(thresholds)

    if is_2d:
        affs = affs[:, np.newaxis, :, :]
        affs = np.concatenate((np.zeros_like(affs[0])[np.newaxis], affs))
    print(affs.shape)

    print("Agglomerating " + " at thresholds " + str(thresholds))

    start = time.time()
    segmentation, fragments, affs_xy, distances, seeds = \
        agglomerate_with_waterz(affs, thresholds,
                                return_merge_history=True,
                                has_background=has_background)
    print("Finished agglomeration in " + str(time.time() - start) + "s")
    return segmentation, fragments, affs_xy, distances, seeds

