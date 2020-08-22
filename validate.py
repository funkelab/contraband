from contraband import script_utils
from pprint import pformat
from contraband import utils
from contraband.validate import validate
from contraband.pipelines.prediction import Predict
from contraband.models.contrastive_volume_net import SegmentationVolumeNet
from contraband.models.placeholder import Placeholder
import contraband.param_mapping as mapping
import json
import os
import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def val_loop(model, pipeline_params, model_params, curr_log_dir, checkpoints):

    for contrastive_ckpt in utils.get_checkpoints(
            curr_log_dir, match='ckpt', white_list=checkpoints):
        for checkpoint in utils.get_checkpoints(os.path.join(
                curr_log_dir, contrastive_ckpt, 'checkpoints'),
                match='checkpoint'):
            checkpoint_log_dir = os.path.join(
                curr_log_dir,
                'contrastive_ckpt' + contrastive_ckpt.split('ckpt')[1])
            os.makedirs(checkpoint_log_dir + '/checkpoints', exist_ok=True)
            os.makedirs(checkpoint_log_dir + '/samples', exist_ok=True)

            seg_head = pipeline_params['seg_head'](
                model, model_params['h_channels'],
                pipeline_params['seg_out_channels'])

            volume_net = SegmentationVolumeNet(model, seg_head)

            print("Model's state_dict:")
            for param_tensor in volume_net.state_dict():
                print(param_tensor, "\t",
                      volume_net.state_dict()[param_tensor].size())

            volume_net.load_base_encoder(
                os.path.join(checkpoint_log_dir, 'checkpoints',
                             checkpoint))
            volume_net.load_seg_head(
                os.path.join(checkpoint_log_dir, 'checkpoints',
                             checkpoint))

            # Put into eval mode
            if model.name == "Placeholder":
                # We have to specify an input shape because the base
                # is just a placeholder
                input_shape = [1,  # batch 
                               model_params["h_channels"],
                               *model_params["in_shape"]]
                seg_head.eval(input_shape)
            else:
                seg_head.eval()
            volume_net.eval()

            # If we are using embedings as input, modify 'data_file' param.
            if model.name == "Placeholder":
                # Going back 2 dirs from curr_log_dir
                # gets us to the dir containing 'embs'
                pipeline_params["embs_file"] = os.path.join(
                    *curr_log_dir.split('/')[:-2], "embs",
                    contrastive_ckpt, "validate", "raw_embs.zarr")

            pipeline = Predict(volume_net, pipeline_params,
                               checkpoint_log_dir)

            validate(volume_net,
                     pipeline,
                     pipeline_params['data_file'],
                     pipeline_params['dataset']['validate'],
                     checkpoint_log_dir,
                     pipeline_params['thresholds'],
                     checkpoint.split('_')[2],
                     has_background=pipeline_params['has_background'])


if __name__ == '__main__':

    dataset, exp, index, checkpoints = script_utils.get_args()
    logdir, expirement_dir = script_utils.get_logdir(dataset, exp)

    curr_log_dir = script_utils.make_dirs(logdir, index)

    root_logger = utils.create_logger(logdir, name='root')
    root_handler = root_logger.handlers[0]

    params = json.load(open(logdir + "/param_dict.json"))
    root_logger.info("Starting experiment %s with dataset %s",
                     expirement_dir.split('-')[0], dataset)

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
    model.make_model(model_params[index])

    seg_comb_dir = os.path.join(curr_log_dir,
                                "seg/combination-" + str(i))
    script_utils.log_params(seg_comb_dir, i, root_handler, seg_params)

    mapping.map_params(seg_params[i])
    # Start the training loop
    val_loop(model, seg_comb, model_params[index], seg_comb_dir, checkpoints)
