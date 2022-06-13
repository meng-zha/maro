import pandas as pd
import torch
from fore_test.test_utility.build_predict import *

class BaseEvaluator(object):
    def __init__(self, conf, test_loader, model):
        super().__init__()
        self.conf = conf
        self.test_loader = test_loader
        self.model = model


    def build_predict(self):
        predict_pf = pd.DataFrame()
        self.model.eval()
        with torch.no_grad():
            for data in self.test_loader:
                output = self.model(data).cpu()
                predict_pf_row = build_predict_df(self.conf, output, data, self.test_loader.dataset.meta_info)
                predict_pf = pd.concat([predict_pf, predict_pf_row])
        return predict_pf

    def build_ground_truth(self):
        raise NotImplementedError
    
    def eval(self):
        if not hasattr(self, "predict_pf"):
            raise BaseException("No predict result found. Please run build_predict first.")
        if not hasattr(self, "ground_truth_pf"):
            raise BaseException("No ground truth result found. Please run build_ground_truth first.")