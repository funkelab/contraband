import os
import argparse
import pandas as pd
from pprint import pprint


if __name__ == '__main__':
    """
        Gets all the validation metrics for validation run. 
        Uses pandas multi_index with
        "contrastive_comb", "seg_comb", "contrastive_ckpt", "seg_ckpt", "threshold"
        as indicies
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset')
    parser.add_argument('-exp')
    parser.add_argument('-checkpoints',
                        nargs='+')
    parser.add_argument('--ignore',
                        nargs='+')
    args = vars(parser.parse_args())

    exp = args['exp']

    dataset = args['dataset']
    datasets = ['fluo', '17_A1']
    if dataset not in datasets:
        raise ValueError("invalid dataset name")

    checkpoints = []
    if 'checkpoints' in args:
        checkpoints = args['checkpoints']

    folders_to_ignore = []
    if 'ignore' in args:
        folders_to_ignore = args['ignore']
    
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
        if dirs[-1] == 'metrics' and (folders_to_ignore is None or dirs[-2] not in folders_to_ignore):
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
                for index in seg_metrics.index.values:
                    data[(contrastive_comb, seg_comb, contrastive_ckpt, seg_ckpt, index)] = seg_metrics.loc[index].to_dict()
    df = pd.DataFrame(data).transpose()
    df.index.names = ["contrastive_comb", "seg_comb", "contrastive_ckpt", "seg_ckpt", "threshold"]
    df.to_csv(os.path.join(logdir, "best_metrics.csv"))
        




