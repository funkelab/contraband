from contraband.post_processing.agglomerate import agglomerate
from contraband.models.contrastive_volume_net import SegmentationVolumeNet, ContrastiveVolumeNet
from shutil import copy
import gunpowder as gp
import json
import os
from contraband.validate import validate
import contraband.param_mapping as mapping
import contraband.utils as utils
from contraband.pipelines.prediction import Predict
from contraband.pipelines.save_embs import SaveEmbs
from contraband.models.placeholder import Placeholder
from contraband.segmentation_heads.sparse_seg_head import SparseSegHead
import numpy as np
from pprint import pformat

import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("gunpowder.nodes.elastic_augment").setLevel(logging.INFO)


class Trainer:
    """
        This contains the training logic for the models and pipelines. 

        Args:
            
            The dataset being trained on. Should be the name of the folder
            in the expirement dir
    
        expirement_num (`int`): 
            
            The number of the expirement being run

        mode (`string`):

            The mode to run the piplines in (contrastive, seg, val, emb)
            contrastive - trains the contrastive pipeline
            seg - traings the segmentation pipeline
            val - makes prediction on whole validate dataeset and runs
            validation script.
            emb - makes embedding for enitre dataset give.

        checkpoints (`list`, optional):

            These are the contrastive checkpoints that are wanted to use
            for segmentation, validation, or embedding. If not specified
            these modes will use all the checkpoints.

    """
    def __init__(self, dataset, expirement_num, mode, checkpoints=None):

        expirement_dir = [
            filename
            for filename in os.listdir(os.path.join("expirements", dataset))
            if filename.startswith('EXP' + str(expirement_num))
        ][0]

        self.logdir = os.path.join("expirements", dataset, expirement_dir)

        assert os.path.isdir(
            self.logdir), "Dir " + self.logdir + " doesn't exist"

        self.params = json.load(open(self.logdir + "/param_dict.json"))

        self.root_logger = utils.create_logger(self.logdir, name='root')
        self.root_handler = self.root_logger.handlers[0]

        self.root_logger.info("Starting experiment %s with dataset %s",
                              expirement_dir.split('-')[:2], dataset)
        self.root_logger.info("Parameter dict: %s", self.params)
        self.root_logger.info("")

        self.checkpoints = checkpoints

        self.contrastive_params = mapping.generate_param_grid(
            self.params['contrastive'])
        self.contrastive_combs = len(self.contrastive_params)
        self.seg_params = mapping.generate_param_grid(self.params['seg'])
        self.model_params = self.params['model']
        if 'save_embs' in self.params:
            self.embedding_params = self.params['save_embs']

        # Get correct combinations of parameters
        index_combs = {
            "contrastive": self.contrastive_params,
            "model": self.model_params
        }
        index_combs = mapping.generate_param_grid(index_combs)
        self.contrastive_params = [comb['contrastive'] for comb in index_combs]
        self.model_params = [comb['model'] for comb in index_combs]

        self.root_logger.info("All model params: {self.model_params}")
        self.pipeline = self.params['pipeline']

        self.mode = mode

    def train(self, index):
        """
            Starts the evaluation for any of the modes and given combination 
            index.

            Args:
                
                index (`int`):

                    This corresponds to the specific combination of contrastive
                    and model parameters that should be run. If there are 4 
                    combinations of model and contrastive parameters, then 
                    index=1 will do the combination at index 1
        """

        curr_log_dir = os.path.join(self.logdir, "combination-" + str(index))
        os.makedirs(curr_log_dir, exist_ok=True)
        assert os.path.isdir(curr_log_dir), \
            os.path.join("Dir ", curr_log_dir, "doesn't exist")

        os.makedirs(curr_log_dir + '/contrastive/checkpoints', exist_ok=True)
        os.makedirs(os.path.join(curr_log_dir, 'seg'), exist_ok=True)

        assert os.path.isdir(curr_log_dir + '/contrastive/checkpoints'), \
            "Dir " + curr_log_dir + "doesn't exist"
        assert os.path.isdir(curr_log_dir + '/seg'), \
            "Dir " + curr_log_dir + "doesn't exist"

        self.train_one(index)

    def train_one(self, index):
        curr_log_dir = os.path.join(self.logdir, "combination-" + str(index))

        utils.log_params(curr_log_dir, index, self.root_handler,
                         pformat(self.model_params[index]))
        utils.log_params(curr_log_dir, index, self.root_handler,
                         self.model_params)
        mapping.map_model_params(self.model_params[index])

        model = self.model_params[index]['model']

        pipeline = mapping.map_pipeline(self.mode, self.pipeline)
        if self.mode == 'contrastive':
            # log parameters
            utils.log_params(curr_log_dir, index, self.root_handler,
                             self.contrastive_params)

            # map the contrastive_params
            mapping.map_params(self.contrastive_params[index])

            # Make the actual model parameters
            model.make_model(self.model_params[index])

            # Start the training loop
            self._contrastive_train_loop(model, self.contrastive_params[index],
                                         self.model_params[index], pipeline,
                                         curr_log_dir)
        elif self.mode == 'seg':
            for i, seg_comb in enumerate(self.seg_params):

                # If we are using the sparse_seg_head model then we don't want the Unet attached.
                if seg_comb['seg_head'] == 'Sparse' and (
                        'baseline' not in seg_comb
                        or not seg_comb['baseline']):
                    model = Placeholder()
                model.make_model(self.model_params[index])

                mapping.map_params(self.seg_params[i])
                seg_comb_dir = os.path.join(curr_log_dir,
                                            "seg/combination-" + str(i))
                os.makedirs(seg_comb_dir, exist_ok=True)

                utils.log_params(seg_comb_dir, i, self.root_handler,
                                 self.seg_params)

                self._seg_train_loop(model, seg_comb, self.model_params[index],
                                     pipeline, curr_log_dir, seg_comb_dir)
        elif self.mode == 'val':
            for i, seg_comb in enumerate(self.seg_params):
                # If we are using the sparse_seg_head model then we don't want the Unet attached.
                if seg_comb['seg_head'] == 'Sparse' and (
                        'baseline' not in seg_comb
                        or not seg_comb['baseline']):
                    model = Placeholder()
                model.make_model(self.model_params[index])

                seg_comb_dir = os.path.join(curr_log_dir,
                                            "seg/combination-" + str(i))
                utils.log_params(seg_comb_dir, i, self.root_handler,
                                 self.seg_params)

                mapping.map_params(self.seg_params[i])

                self._validate(model, seg_comb, self.model_params[index],
                               seg_comb_dir)
        elif self.mode == 'emb':
            utils.log_params(curr_log_dir, index, self.root_handler,
                             self.contrastive_params)
            utils.log_params(curr_log_dir, index, self.root_handler,
                             self.seg_params)
            mapping.map_params(self.contrastive_params[index])
            model.make_model(self.model_params[index])

            self._embed(model, self.contrastive_params[index],
                        self.model_params, curr_log_dir)

    def _contrastive_train_loop(self, model, pipeline_params, model_params,
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

    def _seg_train_loop(self, model, pipeline_params, model_params, pipeline,
                        curr_log_dir, seg_comb_dir):

        # Create placeholder checkpoint if we are using a basline
        if 'baseline' in pipeline_params and pipeline_params["baseline"]:
            open(
                os.path.join(curr_log_dir, "contrastive/checkpoints",
                             "baseline_checkpoint_0"), 'a')

        # Loop over contrastive checkpoints
        for checkpoint in utils.get_checkpoints(os.path.join(
                curr_log_dir, "contrastive/checkpoints"),
                                                match='checkpoint',
                                                white_list=self.checkpoints):

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

                self.root_logger.info("Loading contrastive model...")
                volume_net.load_base_encoder(
                    os.path.join(curr_log_dir, 'contrastive/checkpoints',
                                 checkpoint))

            elif model.name != "Placeholder" and pipeline_params["freeze_base"]:
                self.root_logger.info("Freezing base")
                for param in volume_net.base_encoder.parameters():
                    param.requires_grad = False
            else:
                self.root_logger.info("Not freezing baseline...")
            print([param.requires_grad for param in volume_net.parameters()])

            print("Model's state_dict:")
            for param_tensor in volume_net.state_dict():
                print(param_tensor, "\t",
                      volume_net.state_dict()[param_tensor].size())

            training_pipeline, train_request = curr_pipeline.create_train_pipeline(
                volume_net)

            # Saves the history here. It outputs history to file 'save_every' times.
            # This way it is consitant with the checkpoints.
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

    def _validate(self, model, pipeline_params, model_params, curr_log_dir):

        for contrastive_ckpt in utils.get_checkpoints(
                curr_log_dir, match='ckpt', white_list=self.checkpoints):
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

    def _embed(self, model, pipeline_params, model_params, curr_log_dir):

        for checkpoint in utils.get_checkpoints(os.path.join(
                curr_log_dir, "contrastive/checkpoints"),
                                                match='checkpoint',
                                                white_list=self.checkpoints):
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

            for ds in self.embedding_params["datasets"]:
                pipeline = SaveEmbs(model, pipeline_params, ds,
                                    self.embedding_params["data_file"][0],
                                    checkpoint_log_dir).save_embs()
