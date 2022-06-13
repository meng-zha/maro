import os
import pandas as pd
import datetime

from dataset.data_utility.dataset_utility import load_from_file
from fore_test.evaluator.base_evaluator import BaseEvaluator
from fore_test.test_utility.test_data_utility import *

class DefaultEvaluator(BaseEvaluator):
    def __init__(self, conf, test_loader, model):
        super().__init__(conf, test_loader, model)

    def build_predict(self):
        date_column = self.conf.data.shared_columns.date
        self.predict_pf = super().build_predict()
        self.predict_pf[date_column] = self.predict_pf[date_column] + datetime.timedelta(1) 
        
    def build_ground_truth(self):
        date_column = self.conf.data.shared_columns.date
        target_column = self.conf.data.shared_columns.target
        node_columns = self.conf.data.shared_columns.nodes
        start_date = self.test_loader.dataset.start_date
        end_date = self.test_loader.dataset.end_date
        
        # load ground truth df from files
        ground_files_path = [os.path.join(self.conf.data.root_path, file) for file in self.conf.test.ground_truth.files]
        df_total = load_from_file(ground_files_path, parse_date=date_column).sort_values(by=[date_column])

        index = node_columns + [date_column]
        df_target = df_total[index + [target_column]]

        his_dates = self.conf.data.sequence_feature.history.dates
        fut_dates = self.conf.data.sequence_feature.future.dates
        ground_truth_pf = df_target[ \
            (df_target[date_column] >= pd.to_datetime(start_date) + datetime.timedelta(his_dates)) &\
            (df_target[date_column] <= pd.to_datetime(end_date) - datetime.timedelta(fut_dates-1))]
        self.ground_truth_pf = ground_truth_pf
    
    def eval(self):
        if not hasattr(self, "predict_pf"):
            raise BaseException("No predict result found. Please run build_predict first.")
        if not hasattr(self, "ground_truth_pf"):
            raise BaseException("No ground truth result found. Please run build_ground_truth first.")

        target_column = self.conf.test.eval.label
        predict_column = self.conf.test.eval.predict
        date_column = self.conf.data.shared_columns.date
        node_columns = self.conf.data.shared_columns.nodes
        index = node_columns + [date_column]
        df_merge = pd.merge(self.ground_truth_pf.set_index(index), self.predict_pf.set_index(index),\
            how='outer', left_index=True, right_index=True).reset_index()
        df_merge = df_merge[df_merge[predict_column].notna().all(1)]
        df_merge = df_merge[df_merge[target_column].notna().all(1)]

        # TODO: build evaluation metrics for common test methods?
        metric = 'cal_' + self.conf.test.eval.metric
        acc = eval(metric)(df_merge[predict_column], df_merge[target_column])

        return acc