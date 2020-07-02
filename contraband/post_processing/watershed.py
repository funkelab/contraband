import numpy as np
import mahotas
from scipy import ndimage
from contraband import utils
import os

use_mahotas_watershed = True
seed_distance = 10


def get_seeds(boundary, method='grid', next_id=1, 
              curr_log_dir='',
              curr_ckpt=0,
              curr_sample=0,
              max_samples=0):

    if method == 'grid':

        height = boundary.shape[0]
        width = boundary.shape[1]

        seed_positions = np.ogrid[0:height:seed_distance,
                                  0:width:seed_distance]
        num_seeds_y = seed_positions[0].size
        num_seeds_x = seed_positions[1].size
        num_seeds = num_seeds_x * num_seeds_y

        seeds = np.zeros_like(boundary).astype(np.int32)
        seeds[seed_positions] = np.arange(next_id,
                                          next_id + num_seeds).reshape(
                                              (num_seeds_y, num_seeds_x))

    if method == 'minima':

        minima = mahotas.regmin(boundary)
        seeds, num_seeds = mahotas.label(minima)
        seeds += next_id
        seeds[seeds == next_id] = 0

    if method == 'maxima_distance':

        distance = mahotas.distance(boundary < 0.5)
        maxima = mahotas.regmax(distance)
        seeds, num_seeds = mahotas.label(maxima)
        seeds += next_id
        seeds[seeds == next_id] = 0
        
        return seeds, num_seeds, distance


    return seeds, num_seeds


def watershed(affs, seed_method, curr_log_dir='',
              curr_ckpt=0,
              curr_sample=0,
              max_samples=0):

    affs_xy = 1.0 - 0.5 * (affs[1] + affs[2])
    depth = affs_xy.shape[0]

    fragments = np.zeros_like(affs[0]).astype(np.uint64)

    if curr_sample < max_samples:
        utils.save_zarr(affs_xy,
                        os.path.join(curr_log_dir, 'samples/sample_' +
                                     str(curr_ckpt) + '.zarr'),
                        ds='boundary_' + str(curr_sample),
                        roi=affs_xy.shape) 
    next_id = 1
    distances = np.zeros_like(affs_xy) 
    for z in range(depth):

        seed_data = get_seeds(affs_xy[z],
                              next_id=next_id,
                              method=seed_method,
                              curr_log_dir=curr_log_dir,
                              curr_ckpt=curr_ckpt,
                              curr_sample=curr_sample,
                              max_samples=max_samples)

        if seed_method == 'maxima_distance':
            seeds, num_seeds, distance = seed_data
            distances[z] = distance
        else:
            seeds, num_seeds = seed_data


        if use_mahotas_watershed:
            fragments[z] = mahotas.cwatershed(affs_xy[z], seeds)
        else:
            fragments[z] = ndimage.watershed_ift(
                (255.0 * affs_xy[z]).astype(np.uint8), seeds)

        next_id += num_seeds

    if curr_sample < max_samples:
        utils.save_zarr(distances,
                        os.path.join(curr_log_dir, 'samples/sample_' +
                                     str(curr_ckpt) + '.zarr'),
                        ds='dist_trfm' + str(curr_sample),
                        roi=distances.shape) 

    return fragments
