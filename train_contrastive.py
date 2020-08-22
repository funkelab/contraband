from contraband import script_utils
from contraband import utils
from pprint import pformat
from contraband.pipelines.contrastive import Contrastive
from contraband.models.contrastive_volume_net import ContrastiveVolumeNet
import contraband.param_mapping as mapping
import json
import os
import numpy as np
import gunpowder as gp
import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def train_loop(model, pipeline_params, model_params,
               pipeline, curr_log_dir):

    pipeline = pipeline(pipeline_params, curr_log_dir)

    volume_net = ContrastiveVolumeNet(
        model, model_params['h_channels'],
        model_params['contrastive_out_channels'])
    print("Model's state_dict:")
    for param_tensor in volume_net.state_dict():
        print(param_tensor, "\t",
              volume_net.state_dict()[param_tensor].size())

    training_pipeline, train_request = pipeline.create_train_pipeline(
        volume_net)

    # Saves the history here. It outputs history to file 'save_every' times.
    # This way it is consitant with the checkpoints.
    history_path = os.path.join(curr_log_dir, "contrastive/history.npy")
    loss, start_idx = utils.get_history(history_path)
    with gp.build(training_pipeline):
        curr_loss = []
        for i in range(start_idx, pipeline_params['num_iterations']):
            batch = training_pipeline.request_batch(train_request)
            curr_loss.append(batch.loss)
            if len(curr_loss) % pipeline_params['save_every'] == 0:
                loss = loss + curr_loss
                np.save(history_path, loss, allow_pickle=True)
                curr_loss = []


if __name__ == '__main__':

    dataset, exp, index, _ = script_utils.get_args()
    logdir, expirement_dir = script_utils.get_logdir(dataset, exp)

    curr_log_dir = script_utils.make_dirs(logdir, index)

    root_logger = utils.create_logger(logdir, name='root')
    root_handler = root_logger.handlers[0]

    root_logger.info("Starting experiment %s with dataset %s",
                     expirement_dir.split('-')[0], dataset)
    params = json.load(open(logdir + "/param_dict.json"))

    contrastive_params, _, model_params, _ = script_utils.get_params(params)

    root_logger.info("Parameter dict: %s", pformat(contrastive_params[index]))
    root_logger.info("")

    pipeline = Contrastive

    model = script_utils.get_model(index, model_params, root_logger)

    # log parameters
    script_utils.log_params(curr_log_dir, index, root_handler, contrastive_params)

    # map the contrastive_params
    mapping.map_params(contrastive_params[index])

    # Make the actual model parameters
    model.make_model(model_params[index])

    # Start the training loop
    train_loop(model, contrastive_params[index],
               model_params[index], pipeline,
               curr_log_dir)
