import os
import sys
import argparse
import pandas as pd
from pprint import pprint


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-dataset')
    parser.add_argument('-exp')
    parser.add_argument('-checkpoints',
                        nargs='+')
    parser.add_argument('--fullpath')
    args = vars(parser.parse_args())

    if args['fullpath'] is not None:
        split = args['fullpath'].split('-')
        exp = split[0]
    else:
        exp = args['exp']

    dataset = args['dataset']
    datasets = ['fluo', '17_A1']
    if dataset not in datasets:
        raise ValueError("invalid dataset name")

    checkpoints = []
    if 'checkpoints' in args:
        checkpoints = args['checkpoints']
    
    expirement_dir = [
        filename
        for filename in os.listdir(os.path.join("expirements", dataset))
        if filename.startswith('EXP' + str(exp))
    ][0]

    logdir = os.path.join("expirements", dataset, expirement_dir)
    
    multi_index = []
    data = {} 
    for root, subdir, files in os.walk(logdir, topdown=False):
        dirs = root.split('/')
        if dirs[-1] == 'metrics':
            seg_ckpts = {}
            contrastive_comb = dirs[3]
            seg_comb = dirs[5]
            contrastive_ckpt = dirs[6]

            if checkpoints and contrastive_ckpt not in checkpoints:
                continue

            for f in files:
                seg_ckpt = f.split('_')[1][:-4]
                metrics = pd.read_csv(os.path.join(root, f), index_col=0)
                seg_metrics = metrics[metrics.voi_sum == metrics.voi_sum.min()]
                print(seg_metrics.index.values)
                for index in seg_metrics.index.values:
                    data[(contrastive_comb, seg_comb, contrastive_ckpt, seg_ckpt, index)] = seg_metrics.loc[index].to_dict()
    df = pd.DataFrame(data).transpose()
    df.index.names = ["contrastive_comb", "seg_comb", "contrastive_ckpt", "seg_ckpt", "threshold"]
    df.to_csv(os.path.join(logdir, "best_metrics.csv"))
    print(df.transpose())
        




