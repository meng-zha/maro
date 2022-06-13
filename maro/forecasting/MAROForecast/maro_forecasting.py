import os
import argparse

import sys
sys.path.append(os.path.dirname(__file__))

import torch
from torch.utils.data import DataLoader

from fore_config.config import Config
from dataset.build_dataset import build_dataset
from network.build_model import build_model
from fore_test.evaluator import *


class MAROForecasting(object):
    def __init__(self, config_file):
        self.conf = Config(config_file)
        self.model_path = None
        if getattr(self.conf.test, "model_path", False):
            self.model_path = self.conf.test.model_path
            model_dic, self.encoding_dic = self.model_load(self.model_path)
            self.model = build_model(self.conf).to(self.conf.experiment.device)
            self.model.load_state_dict(model_dic)

    def train(self, data):
        raise NotImplementedError

    def predict(self, data, model_path=None):
        if self.model_path is None and model_path is None:
            raise Exception("No model path is set.")
        
        if model_path is not None:
            model_dic, self.encoding_dic = self.model_load(model_path)
            self.model.load_state_dict(model_dic)

        test_dataset = build_dataset(self.conf, self.encoding_dic, data)['test']
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        evaluator = eval(self.conf.test.name)(self.conf, test_loader, self.model)
        evaluator.build_predict()

        return evaluator.predict_pf
    
    def model_load(self, model_path):
        saved_dic = torch.load(model_path)
        model_dic = saved_dic['model_dic']
        encoding_dic = saved_dic['encoding_dic']
        return model_dic, encoding_dic
