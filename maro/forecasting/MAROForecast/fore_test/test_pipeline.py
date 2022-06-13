import os
import torch
import datetime
import numpy
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
from network.build_model import build_model
from test.evaluator import *

def test(conf, test_dataset):
    model_path = getattr(conf.test, "model_path", os.path.join(conf.model_dir, "model_best.pth"))
    if not os.path.exists(model_path):
        raise BaseException("No model found. Please set the model path correctly or train first.")

    model = build_model(conf).to(conf.experiment.device)
    model.load_state_dict(torch.load(model_path)['model_dic'])

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    conf.logger.info("Starting testing")
    conf.logger.info("Total testing set: " + str(len(test_dataset)))
    evaluator = eval(conf.test.name)(conf, test_loader, model)

    conf.logger.info("Predicting the result.")
    evaluator.build_predict()

    conf.logger.info("Generating the ground truth.")
    evaluator.build_ground_truth()

    conf.logger.info("Testing by " + conf.test.name + ".")
    acc = evaluator.eval()
    conf.logger.info("Testing results: " + str(acc))

    conf.logger.info("Test done.")

