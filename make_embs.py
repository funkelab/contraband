from contraband import script_utils
from pprint import pformat
from contraband import utils
from contraband.pipelines.save_embs import SaveEmbs
from contraband.models.contrastive_volume_net import SegmentationVolumeNet
from contraband.models.placeholder import Placeholder
import contraband.param_mapping as mapping
import json
import os
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def emb_loop(model,
             pipeline_params,
             embedding_params,
             model_params,
             curr_log_dir,
             checkpoints):

    for checkpoint in utils.get_checkpoints(os.path.join(
            curr_log_dir, "contrastive/checkpoints"),
            match='checkpoint',
            white_list=checkpoints):

        checkpoint_log_dir = os.path.join(
            curr_log_dir,
            'embs/contrastive_ckpt' + checkpoint.split('_')[2])
        os.makedirs(checkpoint_log_dir, exist_ok=True)

        utils.load_model(model,
                         "base_encoder.",
                         os.path.join(curr_log_dir,
                                      'contrastive/checkpoints',
                                      checkpoint),
                         freeze_model=True)

        for ds in embedding_params["datasets"]:
            SaveEmbs(model, pipeline_params, ds,
                     embedding_params["data_file"][0],
                     checkpoint_log_dir).save_embs()


if __name__ == '__main__':

    dataset, exp, index, checkpoints = script_utils.get_args()
    logdir, expirement_dir = script_utils.get_logdir(dataset, exp)

    curr_log_dir = script_utils.make_dirs(logdir, index)

    root_logger = utils.create_logger(logdir, name='root')
    root_handler = root_logger.handlers[0]

    params = json.load(open(logdir + "/param_dict.json"))

    root_logger.info("Starting experiment %s with dataset %s",
                     expirement_dir.split('-')[0], dataset)

    contrastive_params, _, model_params, embedding_params = script_utils.get_params(params)
    root_logger.info(f"Parameter dict: {pformat(contrastive_params[index])}")
    root_logger.info("")

    model = script_utils.get_model(index, model_params, root_logger)
    model.eval()

    script_utils.log_params(curr_log_dir, index, root_handler,
                     contrastive_params)

    mapping.map_params(contrastive_params[index])
    model.make_model(model_params[index])

    emb_loop(model,
             contrastive_params[index],
             embedding_params,
             model_params[index],
             curr_log_dir,
             checkpoints)
