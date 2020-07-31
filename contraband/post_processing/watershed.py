import numpy as np
import mahotas
from scipy import ndimage
from contraband import utils

use_mahotas_watershed = True
seed_distance = 10


def get_seeds(boundary, method='grid', next_id=1):

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


def watershed(affs, seed_method, has_background=True, curr_log_dir=''):

    affs_xy = 1.0 - 0.5 * (affs[1] + affs[2])
    depth = affs_xy.shape[0]
    
    # (z, y, x)
    fragments = np.zeros_like(affs[0]).astype(np.uint64)

    next_id = 1
    distances = np.zeros_like(affs_xy) 
    seeds_list = np.zeros_like(affs_xy)
    for z in range(depth):

        seed_data = get_seeds(affs_xy[z],
                              next_id=next_id,
                              method=seed_method)

        if seed_method == 'maxima_distance':
            seeds, num_seeds, distance = seed_data
            if has_background:
                min_dist = np.min(distance)
                lowest = np.array(np.where(distance == min_dist))
                chosen_points = np.random.choice(np.arange(0, lowest.shape[1]), 20)
                background_points = np.take(lowest, chosen_points, axis=1)
                background_points = tuple(background_points[i] for i in range(len(seeds.shape)))

                num_seeds += 1 + next_id 
                print(f'background_point {background_points}')
                print(f"num_seeds {num_seeds}")
                print(f"max_seed {np.max(seeds)}")
                seeds[background_points] = num_seeds
            distances[z] = distance
            seeds_list[z] = seeds
        else:
            seeds, num_seeds = seed_data


        if use_mahotas_watershed:
            fragments[z] = mahotas.cwatershed(affs_xy[z], seeds)
        else:
            fragments[z] = ndimage.watershed_ift(
                (255.0 * affs_xy[z]).astype(np.uint8), seeds)

        next_id += num_seeds

    return fragments, affs_xy, distances, seeds_list
