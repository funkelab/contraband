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

class Trainer:
    def __init__(self, model, expirement_num, mode):

        expirement_dir = [
            filename
            for filename in os.listdir(os.path.join("models", model.name))
            if filename.startswith('EXP' + str(expirement_num))
        ][0]

        self.logdir = os.path.join("models", model.name, expirement_dir)

        assert os.path.isdir(
            self.logdir), "Dir " + self.logdir + " doesn't exist"

        self.params = json.load(open(self.logdir + "/param_dict.json"))

        self.root_logger = utils.create_logger(self.logdir, name='root')
        self.root_handler = self.root_logger.handlers[0]

        self.root_logger.info("Starting experiment %s with model %s",
                              expirement_dir.split('-')[:2], model.name)
        self.root_logger.info("Parameter dict: %s", self.params)
        self.root_logger.info("")

        copy("contraband/models/" + model.name + ".py", self.logdir)

        self.model = model
        self.contrastive_combs = len(mapping.generate_param_grid(self.params['contrastive']))

        print(self.params)
        if mode == 'contrastive':
            self.params = mapping.generate_param_grid(self.params['contrastive'])
        elif mode == 'seg':
            self.params = mapping.generate_param_grid(self.params['seg'])
        elif mode == 'val':
            self.params = mapping.generate_param_grid(self.params['seg'])
        else:
            raise ValueError('Incorrect mode specified' + str(mode))
        self.mode = mode


    def train(self, index):

        # parameters = pd.DataFrame(self.params)
        # parameters["val_accuracy"] = np.nan
        # parameters["val_loss"] = np.nan

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

            # parameters.loc[index, 'val_loss'] = history['val_loss'][0]
            # parameters.loc[index, 'val_accuracy'] = history['val_accuracy'][0]
            # parameters.iloc[index].to_json(curr_log_dir + '/params.json')

            # logger.info("Val loss: " + str(history['val_loss'][0]))
            # logger.info("Val accuracy: " + str(history['val_accuracy'][0]))

        # parameters.to_csv(self.logdir + "/hyperparameter_matrix.csv")

    def train_one(self, index):

        curr_log_dir = os.path.join(self.logdir, "combination-" + str(index))

        pipeline = mapping.map_pipeline(self.mode, self.model.pipeline)

        if self.mode == 'contrastive':
            utils.log_params(curr_log_dir, index, self.root_handler, self.params)
            mapping.map_params(self.params[index])
            self._contrastive_train_loop(self.params[index], pipeline, curr_log_dir)
        elif self.mode == 'seg':
            for i, seg_comb in enumerate(self.params):
                mapping.map_params(self.params[i])
                seg_comb_dir = os.path.join(curr_log_dir, "seg/combination-" + str(i))
                os.makedirs(seg_comb_dir, exist_ok=True)

                utils.log_params(seg_comb_dir, i, self.root_handler, self.params)

                self._seg_train_loop(seg_comb, pipeline, curr_log_dir, seg_comb_dir)
        else:
            for i, seg_comb in enumerate(self.params):
                seg_comb_dir = os.path.join(curr_log_dir, "seg/combination-" + str(i))
                utils.log_params(seg_comb_dir, i, self.root_handler, self.params)
                mapping.map_params(self.params[i])

                self._validate(seg_comb, seg_comb_dir)
        # return history.history

    def _contrastive_train_loop(self, params, pipeline, curr_log_dir):
        pipeline = pipeline(params, curr_log_dir)

        volume_net = ContrastiveVolumeNet(self.model,
                                          params['h_channels'],
                                          params['out_channels'])

        print("Model's state_dict:")
        for param_tensor in volume_net.state_dict():
            print(param_tensor, "\t", volume_net.state_dict()[param_tensor].size())

        training_pipeline, train_request = pipeline.create_train_pipeline(
            volume_net)

        history_path = os.path.join(curr_log_dir, "contrastive/history.npy")
        loss, start_idx = utils.get_history(history_path)
        with gp.build(training_pipeline):
            curr_loss = [] 
            for i in range(start_idx, params['num_iterations']):
                batch = training_pipeline.request_batch(train_request)
                curr_loss.append(batch.loss)
                if len(curr_loss) % params['save_every'] == 0:   
                    loss = loss + curr_loss
                    np.save(history_path, loss, allow_pickle=True)
                    curr_loss = []



    def _seg_train_loop(self, params, pipeline, curr_log_dir, seg_comb_dir):
        for checkpoint in utils.get_checkpoints(os.path.join(curr_log_dir, 
                                                "contrastive/checkpoints"), match='checkpoint'):
            checkpoint_log_dir = os.path.join(seg_comb_dir, 
                                              'contrastive_ckpt' +
                                              checkpoint.split('_')[2]) 
            os.makedirs(os.path.join(checkpoint_log_dir, 'checkpoints'), exist_ok=True)

            curr_pipeline = pipeline(params, checkpoint_log_dir)
            seg_head = params['seg_head'](self.model, 
                                          params['h_channels'],
                                          params['out_channels'])
            volume_net = SegmentationVolumeNet(self.model, seg_head)
            volume_net.load(os.path.join(curr_log_dir, 'contrastive/checkpoints', checkpoint))

            print("Model's state_dict:")
            for param_tensor in volume_net.state_dict():
                print(param_tensor, "\t", volume_net.state_dict()[param_tensor].size())

            training_pipeline, train_request = curr_pipeline.create_train_pipeline(
                volume_net)

            history_path = os.path.join(checkpoint_log_dir, "history.npy")
            loss, start_idx = utils.get_history(history_path)
            with gp.build(training_pipeline):
                curr_loss = [] 
                for i in range(start_idx, params['num_iterations']):
                    batch = training_pipeline.request_batch(train_request)
                    curr_loss.append(batch.loss)
                    if len(curr_loss) % params['save_every'] == 0:   
                        loss = loss + curr_loss
                        np.save(history_path, loss, allow_pickle=True)
                        curr_loss = []


    def _validate(self, params, curr_log_dir):

        for contrastive_ckpt in utils.get_checkpoints(curr_log_dir, match='ckpt'):
            for checkpoint in utils.get_checkpoints(os.path.join(curr_log_dir, 
                                                    contrastive_ckpt, 'checkpoints'), match='checkpoint'):
                checkpoint_log_dir = os.path.join(curr_log_dir, 
                                                  'contrastive_ckpt' +
                                                  contrastive_ckpt.split('ckpt')[1]) 
                os.makedirs(checkpoint_log_dir + '/checkpoints', exist_ok=True)
                os.makedirs(checkpoint_log_dir + '/samples', exist_ok=True)

                seg_head = params['seg_head'](self.model, 
                                              params['h_channels'],
                                              params['out_channels'])
                volume_net = SegmentationVolumeNet(self.model, seg_head)
                volume_net.load(os.path.join(checkpoint_log_dir, 'checkpoints', checkpoint))

                pipeline = Predict(volume_net, params, checkpoint_log_dir)

                validate(volume_net, pipeline, params['data_file'], 
                         params['dataset']['validate'], checkpoint_log_dir,
                         params['thresholds'], checkpoint.split('_')[2])
