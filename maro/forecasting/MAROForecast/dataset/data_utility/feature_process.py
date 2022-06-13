import pandas as pd
import numpy as np
import torch

def processor_single_feature(conf, feature_conf, df_data, meta_info):
    feature_column = feature_conf.column
    meta_info[feature_column] = {}
    # Fill na
    fillna_value = getattr(feature_conf, "fillna", 0)
    df_data[feature_column].fillna(fillna_value, inplace=True)
    # Normalize by histure data
    if getattr(feature_conf, "group_normalize_by_history", False):
        normed_feature_values, mean, std = group_normalize_by_history(conf, feature_column, df_data, meta_info)
        df_data[feature_column] = normed_feature_values
        meta_info[feature_column]["mean"] = mean
        meta_info[feature_column]["std"] = std
    return df_data, meta_info

def group_normalize_by_history(conf, feature_column, df_data, meta_info):
    date_column = conf.data.shared_columns.date
    group_nodes = conf.data.shared_columns.nodes
    date = meta_info["date"]
    node_nums = np.prod(list(meta_info["node_num"].values()))

    node_index = df_data[df_data[date_column]==date][group_nodes]
    his_feature_df = df_data[df_data[date_column]<=date].groupby(group_nodes)[feature_column]
    mean = his_feature_df.mean()
    mean_params = pd.merge(mean, node_index, on=group_nodes, how='right')[feature_column].values
    std = his_feature_df.var().fillna(0) + 1
    std_params = (pd.merge(std, node_index, on=group_nodes, how='right')[feature_column].values)**0.5
    feature_values = df_data[feature_column].values.reshape(node_nums, -1)
    normed_feature_values = ((feature_values - mean_params[:,None]) / std_params[:,None]).reshape(-1)
    return normed_feature_values, mean_params, std_params

def extract_features(df_data, feature_columns, date_column, start_date, end_date, device):
    feature_df = df_data[(df_data[date_column]>=start_date) & (df_data[date_column]<=end_date)][feature_columns]
    dates = (end_date - start_date).days + 1
    shape = [-1, dates, len(feature_columns)]
    feature_nd = np.array(feature_df.values, dtype=np.float).reshape(*shape)
    feature_tensor = torch.from_numpy(feature_nd).to(device)
    return feature_tensor
