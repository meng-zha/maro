from cmath import nan
import pandas as pd
import datetime
from torch.utils.data import Dataset
from dataset.data_utility.feature_process import *

class WalmartDataset(Dataset):
    def __init__(self, conf, dataset_list):
        super().__init__()
        self.conf = conf

        # Sequence data
        # TODO: Current assume only one sequence dataset
        self.seq_data = [dataset["raw_data"] for dataset in dataset_list if dataset["type"]=="sequence"][0]
        self.seq_conf = [dataset_conf for dataset_conf in conf.data.dataset if dataset_conf.type=="sequence"][0]
        self.node_list = self.conf.data.shared_columns.nodes
        self.node_num = {column: len(dataset_list[0]["encoding_mapping"][column].keys()) for column in self.conf.data.shared_columns.nodes}
        self.meta_info = {"encoding_mapping": dataset_list[0]["encoding_mapping"]} # meta info for each sequence, such as encoding mapping

        # Update graph 
        graph_type_list = ["partially_dynamic_graph", "static_graph"]
        self.graph_dic = {}
        for dataset in dataset_list:
            if dataset["type"] in graph_type_list:
                # TODO: Current assume only one sequence dataset
                self.graph_dic[dataset["name"]] = {"type": dataset["type"], "raw_data": dataset["raw_data"]}

        self.his_dates = conf.data.sequence_feature.history.dates
        self.fut_dates = conf.data.sequence_feature.future.dates
        self.date_column = self.conf.data.shared_columns.date
        self.start_date = min(self.seq_data[self.date_column])
        self.end_date = max(self.seq_data[self.date_column])

    def __len__(self):
        return len(pd.date_range(self.start_date, self.end_date)) - self.fut_dates - self.his_dates + 1

    def __getitem__(self, index):
        data = {}
        # Sequence feature 
        start_date = self.start_date + datetime.timedelta(index)
        current_date = self.start_date + datetime.timedelta(index + self.his_dates - 1)
        end_date = self.start_date + datetime.timedelta(index + self.his_dates + self.fut_dates - 1)
        batch_seq_data = self.seq_data[
            (self.seq_data[self.date_column] >= start_date) & \
            (self.seq_data[self.date_column] <= end_date) \
        ].copy()


        meta_info = {
            'node_num': self.node_num,
            'date': current_date.strftime('%Y-%m-%d')
        }
        for feature_conf in self.conf.data.sequence_feature.features:
            batch_seq_data, meta_info = processor_single_feature(
                self.conf,
                feature_conf,
                batch_seq_data, 
                meta_info
            )

        history_tensor = extract_features(
            batch_seq_data, 
            self.conf.data.sequence_feature.history.features,
            self.date_column, 
            start_date, 
            current_date,
            self.conf.experiment.device
        )
        data["history_features"] = history_tensor

        future_tensor = extract_features(
            batch_seq_data, 
            self.conf.data.sequence_feature.future.features,
            self.date_column, 
            current_date + datetime.timedelta(1), 
            end_date,
            self.conf.experiment.device
        )
        data["future_features"] = future_tensor

        label_tensor = extract_features(
            batch_seq_data, 
            self.conf.data.sequence_feature.label.features,
            self.date_column, 
            current_date + datetime.timedelta(1), 
            end_date,
            self.conf.experiment.device
        )
        data["labels"] = label_tensor

        node_tensor = extract_features(
            batch_seq_data, 
            self.conf.data.sequence_feature.node.features,
            self.date_column, 
            current_date + datetime.timedelta(1), 
            end_date,
            self.conf.experiment.device
        )
        data["node"] = node_tensor
        
        # Graph feature
        graph_adj = self.conf.data.graph_feature.adj
        for graph_name in self.graph_dic:
            grapy_type = self.graph_dic[graph_name]["type"]
            graph_data = self.graph_dic[graph_name]["raw_data"]
            if grapy_type == "partially_dynamic_graph":
                graph_data = graph_data[graph_data[self.date_column]==current_date][graph_adj].values
            elif grapy_type == "static_graph":
                graph_data = graph_data[graph_adj].values
            else:
                raise NotImplementedError
            graph_tensor = torch.from_numpy(graph_data).to(dtype=torch.float, device=self.conf.experiment.device) 
            data[graph_name] = graph_tensor

        data["meta_info"] = meta_info
        return data