import sys
import os
import logging
from itertools import product
import json
from shutil import copy
import gunpowder as gp
import torch
import argparse
from pipelines.Standard2DContrastive import Standard2DContrastive
from pipelines.Standard2DSeg import Standard2DSeg
from models.Unet2D import Unet2D
from segmentation_heads.SimpleSegHead import SimpleSegHead
from torch.nn import MSELoss
import waterz
import numpy as np
from models.ContrastiveVolumeNet import SegmentationVolumeNet, ContrastiveVolumeNet

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src import SL


class trainer:

    def __init__(self, model, expirement_num, mode):

        expirement_dir = [filename for filename in os.listdir("models" + SL + model.name) \
                          if filename.startswith('EXP' + str(expirement_num))][0]

        self.logdir = "models" + SL + model.name + SL + expirement_dir 

        assert os.path.isdir(self.logdir), "Dir " + self.logdir + " doesn't exist"

        self.params = json.load(open(self.logdir + "/param_dict.json"))

        self.root_logger = self.create_logger(self.logdir, name='root')
        self.root_handler = self.root_logger.handlers[0]

        self.root_logger.info("Starting expirement " + str(expirement_dir.split('-')[:2]) \
                              + 'with model ' + model.name)
        self.root_logger.info("Parameter dict: " + str(self.params))
        self.root_logger.info("")

        copy("src/models/" + model.name + ".py", self.logdir)

        self.model = model
        print(self.params)
        if mode == 'contrastive':
            self.params = self.generate_param_grid(self.params['contrastive'])
        elif mode == 'seg':
            self.params = self.generate_param_grid(self.params['seg'])
        else:
            raise ValueError('Incorrect mode specified' + str(mode))
        self.mode = mode


    def train(self):

        # parameters = pd.DataFrame(self.params)
        # parameters["val_accuracy"] = np.nan
        # parameters["val_loss"] = np.nan

        for index in range(len(self.params)):
            curr_log_dir = self.logdir + SL + "combination-" + str(index)
            os.makedirs(curr_log_dir, exist_ok=True)
            assert os.path.isdir(curr_log_dir), "Dir " + curr_log_dir + "doesn't exist"
            os.makedirs(curr_log_dir + '/checkpoints', exist_ok=True)
            assert os.path.isdir(curr_log_dir + '/checkpoints'), "Dir " + curr_log_dir + "doesn't exist"

            logger = self.create_logger(curr_log_dir, index=index)
            logger.addHandler(self.root_handler)

            logger.info('Training with parameter combination ' + str(index))
            logger.info("With parameters: " + str(self.params[index]))
            logger.info("")

            self.train_one(index, logger)

            # parameters.loc[index, 'val_loss'] = history['val_loss'][0]
            # parameters.loc[index, 'val_accuracy'] = history['val_accuracy'][0]
            # parameters.iloc[index].to_json(curr_log_dir + '/params.json')

            # logger.info("Val loss: " + str(history['val_loss'][0]))
            #logger.info("Val accuracy: " + str(history['val_accuracy'][0]))

        # parameters.to_csv(self.logdir + "/hyperparameter_matrix.csv")

    def train_one(self, index, logger):

        curr_log_dir = self.logdir + SL + "combination-" + str(index)
        logger.info("Current log dir: " + curr_log_dir)

        self.map_params(index)

        pipeline = self.map_pipeline(self.model.pipeline)
        pipeline = pipeline(self.params[index], curr_log_dir)

        if mode == 'contrastive':
            self._contrastive_train_loop(self.params[index], pipeline)
        else:
            self._seg_train_loop(self.params[index], pipeline, curr_log_dir)
        # return history.history

    def _contrastive_train_loop(self, params, pipeline):
        volume_net = ContrastiveVolumeNet(self.model, 20, 3)

        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size()) 

        training_pipeline, train_request = pipeline.create_train_pipeline(volume_net)
        with gp.build(training_pipeline):
            for i in range(params['num_iterations']):
                batch = training_pipeline.request_batch(train_request)
                print(batch.loss)

    def _seg_train_loop(self, params, pipeline, curr_log_dir):
        checkpoint = curr_log_dir + '/checkpoints/model_checkpoint_1'
        seg_head = params['seg_head'](self.model, 20, 2)
        volume_net = SegmentationVolumeNet(self.model, seg_head)
        volume_net.load(checkpoint)

        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size()) 

        training_pipeline, train_request = pipeline.create_train_pipeline(volume_net)
        val_pipeline, val_request, gt_aff, predictions = pipeline.create_val_pipeline(volume_net)
        with gp.build(training_pipeline), gp.build(val_pipeline):
            for i in range(params['num_iterations']):
                batch = training_pipeline.request_batch(train_request)
                print(batch)
                if i % 2 == 0:
                    for i in range(5):
                        batch = val_pipeline.request_batch(val_request)

    def generate_param_grid(self, params):
        return [dict(zip(params.keys(), values)) for values in product(*params.values())]

    def create_logger(self, log_dir, name=None, index=None):
        if name is None:
            assert index is not None, "Must specify index in create logger"
            name = 'combination-' + str(index)

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_dir + '/' + name + ".log")
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def map_params(self, index):
        if self.params[index]['optimizer'] == 'adam':
            kwargs = {}
            if 'lr' in self.params[index]:
                kwargs['lr'] = self.params[index]['lr']
            if 'clipvalue' in self.params[index]:
                kwargs['clipvalue'] = self.params[index]['clipvalue']
            elif 'clipnorm' in self.params[index]:
                kwargs['clipnorm'] = self.params[index]['clipnorm']
            if 'decay' in self.params[index]:
                kwargs['decay'] = self.params[index]['decay']
            self.params[index]['optimizer'] = torch.optim.Adam
            self.params[index]['optimizer_kwargs'] = kwargs

        if 'seg_head' in self.params[index]:
            if self.params[index]['seg_head'] == 'SimpleSegHead':
                self.params[index]['seg_head'] = SimpleSegHead

    def map_pipeline(self, pipeline):
        if pipeline == "Standard2D":
            if mode == 'contrastive':
                return Standard2DContrastive
            else:
                return Standard2DSeg
        else:
            raise ValueError('Incorrect pipeline: ' + pipeline)


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

    trainer(model, exp, mode).train()



