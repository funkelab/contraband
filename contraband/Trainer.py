from contraband.post_processing.agglomerate import agglomerate
from contraband.models.ContrastiveVolumeNet import SegmentationVolumeNet, ContrastiveVolumeNet
from shutil import copy
import gunpowder as gp
import json
import os
from contraband.validate import validate
import contraband.param_mapping as mapping
import contraband.utils as utils
from contraband.pipelines.prediction import Predict
import numpy as np
from pprint import pformat

class Trainer:
    def __init__(self, dataset, expirement_num, mode, checkpoints):

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

        self.contrastive_params = mapping.generate_param_grid(self.params['contrastive'])
        self.contrastive_combs = len(self.contrastive_params)
        self.seg_params = mapping.generate_param_grid(self.params['seg'])
        self.model_params = self.params['model'] 

        # Get correct combinations of parameters
        index_combs = {"contrastive": self.contrastive_params, "model": self.model_params}
        index_combs = mapping.generate_param_grid(index_combs)
        self.contrastive_params = [comb['contrastive'] for comb in index_combs]
        self.model_params = [comb['model'] for comb in index_combs]

        self.root_logger.info("All model params: {self.model_params}")
        self.pipeline = self.params['pipeline']

        self.mode = mode

    def train(self, index):

        curr_log_dir = os.path.join(self.logdir,
                                    "combination-" + str(index))
        os.makedirs(curr_log_dir, exist_ok=True)
        assert os.path.isdir(curr_log_dir), \
            os.path.join("Dir ", curr_log_dir, "doesn't exist")

        os.makedirs(curr_log_dir + '/contrastive/checkpoints',
                    exist_ok=True)
        os.makedirs(os.path.join(curr_log_dir, 'seg'), exist_ok=True) 

        assert os.path.isdir(curr_log_dir + '/contrastive/checkpoints'), \
            "Dir " + curr_log_dir + "doesn't exist"
        assert os.path.isdir(curr_log_dir + '/seg'), \
            "Dir " + curr_log_dir + "doesn't exist"

        self.train_one(index)

    def train_one(self, index):

        curr_log_dir = os.path.join(self.logdir, "combination-" + str(index))

        utils.log_params(curr_log_dir, 
                         index,
                         self.root_handler,
                         pformat(self.model_params[index]))
        utils.log_params(curr_log_dir, index, self.root_handler, self.model_params)
        mapping.map_model_params(self.model_params[index])

        model = self.model_params[index]['model']
        model.make_model(self.model_params[index])

        pipeline = mapping.map_pipeline(self.mode, self.pipeline)
        if self.mode == 'contrastive':
            utils.log_params(curr_log_dir, index, self.root_handler, self.contrastive_params)
            mapping.map_params(self.contrastive_params[index])
            self._contrastive_train_loop(model,
                                         self.contrastive_params[index], 
                                         self.model_params[index],
                                         pipeline,
                                         curr_log_dir)
        elif self.mode == 'seg':
            for i, seg_comb in enumerate(self.seg_params):
                mapping.map_params(self.seg_params[i])
                seg_comb_dir = os.path.join(curr_log_dir, "seg/combination-" + str(i))
                os.makedirs(seg_comb_dir, exist_ok=True)

                utils.log_params(seg_comb_dir, i, self.root_handler, self.seg_params)

                self._seg_train_loop(model,
                                     seg_comb,
                                     self.model_params[index], 
                                     pipeline,
                                     curr_log_dir,
                                     seg_comb_dir)
        else:
            for i, seg_comb in enumerate(self.seg_params):
                seg_comb_dir = os.path.join(curr_log_dir, "seg/combination-" + str(i))
                utils.log_params(seg_comb_dir, i, self.root_handler, self.seg_params)

                mapping.map_params(self.seg_params[i])

                self._validate(model,
                               seg_comb,
                               self.model_params[index],
                               seg_comb_dir)

    def _contrastive_train_loop(self, model, pipeline_params, model_params, pipeline, curr_log_dir):
        pipeline = pipeline(pipeline_params, curr_log_dir)
        
        volume_net = ContrastiveVolumeNet(model,
                                          model_params['h_channels'],
                                          model_params['contrastive_out_channels'])
        print("Model's state_dict:")
        for param_tensor in volume_net.state_dict():
            print(param_tensor, "\t", volume_net.state_dict()[param_tensor].size())

        training_pipeline, train_request = pipeline.create_train_pipeline(
            volume_net)

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


    def _seg_train_loop(self, model, pipeline_params, model_params, pipeline, curr_log_dir, seg_comb_dir):
        if 'baseline' in pipeline_params and pipeline_params["baseline"]:
            open(os.path.join(curr_log_dir, "contrastive/checkpoints", "baseline_checkpoint_0"), 'a') 

        for checkpoint in utils.get_checkpoints(os.path.join(curr_log_dir, 
                                                "contrastive/checkpoints"),
                                                match='checkpoint',
                                                white_list=self.checkpoints):
            checkpoint_log_dir = os.path.join(seg_comb_dir, 
                                              'contrastive_ckpt' +
                                              checkpoint.split('_')[2]) 
            os.makedirs(os.path.join(checkpoint_log_dir, 'checkpoints'), exist_ok=True)

            curr_pipeline = pipeline(pipeline_params, checkpoint_log_dir)

            seg_head = pipeline_params['seg_head'](model, 
                                          model_params['h_channels'],
                                          pipeline_params['seg_out_channels'])

            volume_net = SegmentationVolumeNet(model, seg_head)
            if 'baseline' not in pipeline_params or not pipeline_params['baseline']:
                self.root_logger.info("Loading contrastive model...")
                volume_net.load_base_encoder(os.path.join(curr_log_dir, 'contrastive/checkpoints', checkpoint))
            elif pipeline_params["freeze_base"]:
                self.root_logger.info("Freezing base") 
                for param in volume_net.base_encoder.parameters():
                    param.requires_grad = False
            else:
                self.root_logger.info("Not freezing baseline...")
            print([param.requires_grad for param in volume_net.parameters()])

            print("Model's state_dict:")
            for param_tensor in volume_net.state_dict():
                print(param_tensor, "\t", volume_net.state_dict()[param_tensor].size())

            training_pipeline, train_request = curr_pipeline.create_train_pipeline(
                volume_net)

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

        for contrastive_ckpt in utils.get_checkpoints(curr_log_dir, match='ckpt',
                                                      white_list=self.checkpoints):
            for checkpoint in utils.get_checkpoints(os.path.join(curr_log_dir, 
                                                    contrastive_ckpt, 'checkpoints'), match='checkpoint'):
                checkpoint_log_dir = os.path.join(curr_log_dir, 
                                                  'contrastive_ckpt' +
                                                  contrastive_ckpt.split('ckpt')[1]) 
                os.makedirs(checkpoint_log_dir + '/checkpoints', exist_ok=True)
                os.makedirs(checkpoint_log_dir + '/samples', exist_ok=True)

                seg_head = pipeline_params['seg_head'](model, 
                                              model_params['h_channels'],
                                              pipeline_params['seg_out_channels'])

                volume_net = SegmentationVolumeNet(model, seg_head)

                print("Model's state_dict:")
                for param_tensor in volume_net.state_dict():
                    print(param_tensor, "\t", volume_net.state_dict()[param_tensor].size())

                volume_net.load_base_encoder(os.path.join(checkpoint_log_dir, 'checkpoints', checkpoint))
                volume_net.load_seg_head(os.path.join(checkpoint_log_dir, 'checkpoints', checkpoint))
                volume_net.eval()

                pipeline = Predict(volume_net, pipeline_params, checkpoint_log_dir)

                validate(volume_net, pipeline, pipeline_params['data_file'], 
                         pipeline_params['dataset']['validate'], checkpoint_log_dir,
                         pipeline_params['thresholds'], checkpoint.split('_')[2],
                         has_background=pipeline_params['has_background'])
