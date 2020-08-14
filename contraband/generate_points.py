import numpy as np
import pandas as pd
import daisy

pos_points = [] 
pos_data = [] 
pos_gt = []
neg_points = [] 
neg_data = [] 
neg_gt = []
n = 500

#filename = "data/ctc/Fluo-N2DH-SIM+.zarr"
filename = "data/ctc/17_1A.zarr"
embs_file = "expirements/17_A1/EXP0-testing-0-0-0/"
raw_dataset = "validate/raw"
gt_dataset = "validate/gt"

raw_data = daisy.open_ds(filename, raw_dataset)
gt_data = daisy.open_ds(filename, gt_dataset)

source_shape = raw_data.data.shape 
roi = raw_data.roi
dims = roi.dims()

while len(pos_points) < n:

    possible_point = np.floor((np.random.random(len(source_shape)) * source_shape)).astype(np.long)
    selection_index = tuple(possible_point[i] for i in range(possible_point.shape[0]))
    gt_point = gt_data.data[selection_index]
    if gt_point > 0:
        pos_gt.append(1)
        pos_data.append(raw_data.data[selection_index])
        pos_points.append(selection_index)

while len(neg_points) < n:

    possible_point = np.floor((np.random.random(len(source_shape)) * source_shape)).astype(np.long)
    selection_index = tuple(possible_point[i] for i in range(possible_point.shape[0]))
    gt_point = gt_data.data[selection_index]
    if gt_point == 0:
        neg_gt.append(0)
        neg_data.append(raw_data.data[selection_index])
        neg_points.append(selection_index)

df = pd.DataFrame()

df['point'] = (pos_points + neg_points)
df['data'] = (pos_data + neg_data)
df['gt'] = (pos_gt + neg_gt)

#df.to_csv("expirements/fluo/EXP7-point_loss-8-3-20/validate_points.csv")
df.to_csv("expirements/17_A1/EXP0-testing-0-0-0/validate_points.csv")
