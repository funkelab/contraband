import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import matplotlib.pyplot as plt
import logging
from itertools import product
import json
from datetime import datetime
import pandas as pd
import numpy as np
from shutil import copy
import gunpowder as gp
from pipelines.standard_2d import standard_2d
from models.unet_2d import unet_2d
from src import SL
import torch
import argparse

class trainer:

    def __init__(self, model, expirement_name, expirement_num,
                 date=datetime.now().strftime("%m-%d-%y")):

        self.logdir = "models" + SL + model.name + SL + "EXP" + str(expirement_num) + '-' \
                      + expirement_name + '-' + date

        assert os.path.isdir(self.logdir), "Dir " + self.logdir + " doesn't exist"

        self.params = json.load(open(self.logdir + "/param_dict.json"))

        self.root_logger = self.create_logger(self.logdir, name='root')
        self.root_handler = self.root_logger.handlers[0]

        self.root_logger.info("Starting expirement " + expirement_name + '-' + str(expirement_num) \
                              + 'with model ' + model.name)
        self.root_logger.info("Parameter dict: " + str(self.params))
        self.root_logger.info("")

        copy("src/models/" + model.name + ".py", self.logdir)

        self.model = model
        self.params = self.generate_param_grid(self.params)

    def train(self):

        parameters = pd.DataFrame(self.params)
        parameters["val_accuracy"] = np.nan
        parameters["val_loss"] = np.nan

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

        training_model = self.model.create_model(self.params[index], training=True)
        val_model = self.model.create_model(self.params[index], training=False)

        print("Model's state_dict:")
        for param_tensor in training_model.state_dict():
            print(param_tensor, "\t", training_model.state_dict()[param_tensor].size()) 

        pipeline = self.map_pipeline(self.model.pipeline)

        pipeline = pipeline(self.params[index], curr_log_dir)
        training_pipeline, train_request = pipeline.create_train_pipeline(training_model)
        val_pipeline, val_request = pipeline.create_val_pipeline(val_model)
        # model = self.model.create_model(self.params, index, logger)

        # with gp.build(training_pipeline):
        #     for i in range(self.params[index]['num_iterations']):
        #         batch = training_pipeline.request_batch(train_request) 
                # print(batch)
        with gp.build(val_pipeline):
            for i in range(self.params[index]['num_iterations']):
                batch = val_pipeline.request_batch(val_request)
        # return history.history

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


    def map_pipeline(self, pipeline):
        if pipeline == "standard_2d":
            return standard_2d


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-model')
    parser.add_argument('-desc')
    parser.add_argument('-exp')
    parser.add_argument('-date')
    parser.add_argument('--fullpath')
    args = vars(parser.parse_args())

    if args['fullpath'] is not None:
        split = args['fullpath'].split('-')
        exp = split[0]
        desc = split[1]
        date = '-'.join(split[2:])
    else:
        exp = args['exp']
        desc = args['desc']
        date = args['date']
    
    model = args['model']
    if model == 'unet_2d':
        model = unet_2d()
    else:
        raise ValueError("invalid model name")

    trainer(model, desc, exp, date).train()



