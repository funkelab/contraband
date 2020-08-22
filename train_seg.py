from contraband import script_utils
from contraband import utils
from pprint import pformat
from contraband.pipelines.segmentation import Segmentation
from contraband.pipelines.sparse_sh import SparseSH
from contraband.pipelines.sparse_baseline import SparseBasline
from contraband.models.contrastive_volume_net import SegmentationVolumeNet
from contraband.models.placeholder import Placeholder
import contraband.param_mapping as mapping
import json
import os
import numpy as np
import gunpowder as gp
import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def train_loop(model, 
               pipeline_params, 
               model_params,
               pipeline,
               root_logger,
               checkpoints,
               curr_log_dir,
               seg_comb_dir):

    # Create placeholder checkpoint if we are using a basline
    if 'baseline' in pipeline_params and pipeline_params["baseline"]:
        open(
            os.path.join(curr_log_dir, "contrastive/checkpoints",
                         "baseline_checkpoint_0"), 'a')

    # Loop over contrastive checkpoints
    for checkpoint in utils.get_checkpoints(os.path.join(
            curr_log_dir, "contrastive/checkpoints"),
            match='checkpoint',
            white_list=checkpoints):

        # This is the dir for training on the current contrastive
        # checkpoint. Here will contain the seg head checkpoints
        checkpoint_log_dir = os.path.join(
            seg_comb_dir, 'contrastive_ckpt' + checkpoint.split('_')[2])
        os.makedirs(os.path.join(checkpoint_log_dir, 'checkpoints'),
                    exist_ok=True)

        curr_pipeline = pipeline(pipeline_params, checkpoint_log_dir)

        seg_head = pipeline_params['seg_head'](
            model, model_params['h_channels'],
            pipeline_params['seg_out_channels'])

        volume_net = SegmentationVolumeNet(model, seg_head)

        # We want to load the contrastive checkpoint if it isn't
        # a baseline or a placeholder
        if ('baseline' not in pipeline_params or not
            pipeline_params['baseline']) and \
                model.name != "Placeholder":

            root_logger.info("Loading contrastive model...")
            volume_net.load_base_encoder(
                os.path.join(curr_log_dir, 'contrastive/checkpoints',
                             checkpoint))

        elif model.name != "Placeholder" and pipeline_params["freeze_base"]:
            root_logger.info("Freezing base")
            for param in volume_net.base_encoder.parameters():
                param.requires_grad = False
        else:
            root_logger.info("Not freezing baseline...")

        print("Model's state_dict:")
        for param_tensor in volume_net.state_dict():
            print(param_tensor, "\t",
                  volume_net.state_dict()[param_tensor].size())

        training_pipeline, train_request = curr_pipeline.create_train_pipeline(
            volume_net)

        # Saves the history here. It outputs history to file 'save_every' times.
        # This way it is consitant with the checkpoints.
        print(os.path.join(checkpoint_log_dir, "history.npy"))
        history_path = os.path.join(checkpoint_log_dir, "history.npy")
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

    dataset, exp, index, checkpoints = script_utils.get_args()
    logdir, expirement_dir = script_utils.get_logdir(dataset, exp)

    curr_log_dir = script_utils.make_dirs(logdir, index)

    root_logger = utils.create_logger(logdir, name='root')
    root_handler = root_logger.handlers[0]

    root_logger.info("Starting experiment %s with dataset %s",
                     expirement_dir.split('-')[0], dataset)

    params = json.load(open(logdir + "/param_dict.json"))
    contrastive_params, seg_params, model_params, _ = script_utils.get_params(params)

    root_logger.info(f"Parameter dict: {pformat(contrastive_params[index])}")
    root_logger.info("")

    model = script_utils.get_model(index, model_params, root_logger)

    for i, seg_comb in enumerate(seg_params):
        # If we are using the sparse_seg_head model then we don't want the Unet attached.
        if seg_comb['seg_head'] == 'Sparse' and (
                'baseline' not in seg_comb
                or not seg_comb['baseline']):
            model = Placeholder()
            pipeline = SparseSH
        elif 'baseline' in seg_comb and seg_comb['baseline']:
            pipeline = SparseBasline
        else:
            pipeline = Segmentation
        
        model.make_model(model_params[index])

        seg_comb_dir = os.path.join(curr_log_dir,
                                    "seg/combination-" + str(i))
        script_utils.log_params(seg_comb_dir, i, root_handler,
                                seg_params)

        mapping.map_params(seg_params[i])
        # Start the training loop
        train_loop(model,
                   seg_params[index],
                   model_params[index],
                   pipeline,
                   root_logger,
                   checkpoints,
                   curr_log_dir,
                   seg_comb_dir)
