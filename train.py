import argparse
from contraband.Trainer import Trainer


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-dataset')
    parser.add_argument('-exp')
    parser.add_argument('-mode')
    parser.add_argument('-index')
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
        raise ValueError("invalid model name")

    index = None
    try:
        index = int(args['index'])
    except Exception as e:
        print(e)
    if index is None:
        raise ValueError("index is not specified or is not an int")
    print(index)
    mode = args['mode']

    Trainer(dataset, exp, mode).train(index)
