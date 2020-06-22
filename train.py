import argparse
from contraband.Trainer import Trainer
from contraband.models.Unet2D import Unet2D


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-model')
    parser.add_argument('-exp')
    parser.add_argument('-mode')
    parser.add_argument('--fullpath')
    args = vars(parser.parse_args())

    if args['fullpath'] is not None:
        split = args['fullpath'].split('-')
        exp = split[0]
    else:
        exp = args['exp']

    model = args['model']
    if model == 'Unet2D':
        model = Unet2D()
    else:
        raise ValueError("invalid model name")

    mode = args['mode']

    Trainer(model, exp, mode).train()
