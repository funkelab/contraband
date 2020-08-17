import os
import numpy as np
import pandas as pd
import daisy
import argparse

pos_points = [] 
pos_gt = []
neg_points = [] 
neg_gt = []

parser = argparse.ArgumentParser()
parser.add_argument('-data_file')
parser.add_argument('-gt_dataset')
parser.add_argument('-exp')
parser.add_argument('-num_points', default=500)
args = vars(parser.parse_args())

data_file = args['data_file']
if data_file == "fluo":
    filename = "data/ctc/Fluo-N2DH-SIM+.zarr"
elif data_file == "17_A1":
    filename = "data/ctc/17_1A.zarr"
else:
    raise ValueError("invalid model name")

gt_dataset = args["gt_dataset"] 

gt_data = daisy.open_ds(filename, gt_dataset)

source_shape = gt_data.data.shape 
roi = gt_data.roi
dims = roi.dims()

num_points = args["num_points"]
while len(pos_points) < num_points:

    possible_point = np.floor((np.random.random(len(source_shape)) * source_shape)).astype(np.long)
    selection_index = tuple(possible_point[i] for i in range(possible_point.shape[0]))
    gt_point = gt_data.data[selection_index]
    if gt_point > 0:
        pos_gt.append(1)
        pos_points.append(selection_index)

while len(neg_points) < num_points:

    possible_point = np.floor((np.random.random(len(source_shape)) * source_shape)).astype(np.long)
    selection_index = tuple(possible_point[i] for i in range(possible_point.shape[0]))
    gt_point = gt_data.data[selection_index]
    if gt_point == 0:
        neg_gt.append(0)
        neg_points.append(selection_index)

df = pd.DataFrame()

df['point'] = (pos_points + neg_points)
df['gt'] = (pos_gt + neg_gt)

expirement_dir = [
    filename
    for filename in os.listdir(os.path.join("expirements", data_file))
    if filename.startswith('EXP' + str(args["exp"]))
][0]

expirement_dir = os.path.join("expirements", data_file, expirement_dir)
df.to_csv(os.path.join(expirement_dir, "validate_points.csv"))
