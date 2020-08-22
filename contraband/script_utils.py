import argparse
import contraband.param_mapping as mapping
import os
from pprint import pformat
from contraband import utils


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset')
    parser.add_argument('-exp')
    parser.add_argument('-index')
    parser.add_argument('-checkpoints',
                        nargs='+')
    args = vars(parser.parse_args())

    exp = args['exp']

    dataset = args['dataset']
    datasets = ['fluo', '17_A1']
    if dataset not in datasets:
        raise ValueError("invalid dataset name")

    index = None
    try:
        index = int(args['index'])
    except Exception as e:
        print(e)
    if index is None:
        raise ValueError("index is not specified or is not an int")

    checkpoints = []
    if 'checkpoints' in args:
        checkpoints = args['checkpoints']

    return dataset, exp, index, checkpoints


def make_dirs(logdir, index):

    curr_log_dir = os.path.join(logdir, "combination-" + str(index))
    os.makedirs(curr_log_dir, exist_ok=True)
    assert os.path.isdir(curr_log_dir), \
        os.path.join("Dir ", curr_log_dir, "doesn't exist")

    os.makedirs(curr_log_dir + '/contrastive/checkpoints', exist_ok=True)
    os.makedirs(os.path.join(curr_log_dir, 'seg'), exist_ok=True)

    assert os.path.isdir(curr_log_dir + '/contrastive/checkpoints'), \
        "Dir " + curr_log_dir + "doesn't exist"
    assert os.path.isdir(curr_log_dir + '/seg'), \
        "Dir " + curr_log_dir + "doesn't exist"

    return curr_log_dir


def get_logdir(dataset, expirement_num):

    expirement_dir = [
        filename
        for filename in os.listdir(os.path.join("expirements", dataset))
        if filename.startswith('EXP' + str(expirement_num))
    ][0]

    logdir = os.path.join("expirements", dataset, expirement_dir)

    assert os.path.isdir(
        logdir), "Dir " + logdir + " doesn't exist"

    return logdir, expirement_dir


def get_params(params):

    contrastive_params = mapping.generate_param_grid(params['contrastive'])
    seg_params = mapping.generate_param_grid(params['seg'])
    model_params = params['model']
    if 'save_embs' in params:
        embedding_params = params['save_embs']

    # Get correct combinations of parameters
    index_combs = {
        "contrastive": contrastive_params,
        "model": model_params
    }
    index_combs = mapping.generate_param_grid(index_combs)
    contrastive_params = [comb['contrastive'] for comb in index_combs]
    model_params = [comb['model'] for comb in index_combs]

    return contrastive_params, seg_params, model_params, embedding_params


def get_model(index, model_params, logger):
    logger.info(f"Model params: {pformat(model_params[index])}")

    mapping.map_model_params(model_params[index])

    model = model_params[index]['model']

    return model


def log_params(curr_log_dir, index, root_handler, params):
    """
        Logs the parameters given.
    """
    logger = utils.create_logger(curr_log_dir, index=index)
    logger.addHandler(root_handler)

    logger.info("Current log dir: " + curr_log_dir)
    logger.info('Training with parameter combination ' + str(index))
    logger.info("With parameters: " + pformat(params[index]))
    logger.info("")
