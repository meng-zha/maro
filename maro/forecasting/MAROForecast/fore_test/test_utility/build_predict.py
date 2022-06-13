import pandas as pd
import numpy as np
def build_predict_df(conf, predict_tensor, data_tensor, dataset_metas):
    result = pd.DataFrame()
    for column_conf in conf.test.predict.columns:
        column = column_conf.column
        source = column_conf.source
        if source[0] == "tensor":
            tensor = source[1]
            tensor_index = source[2]
            column_value = data_tensor[tensor][:, :, 0, tensor_index].cpu().numpy().reshape(-1)
        elif source[0] == "meta_info":
            meta_name = source[1]
            value = data_tensor["meta_info"][meta_name]
            column_value = value * int(predict_tensor.shape[1] / len(value))
        elif source[0] == "output":
            tensor_index = source[1]
            column_value = predict_tensor[:,:,0, tensor_index].cpu().numpy().reshape(-1)
        elif source[0] == "total":
            tensor_index = source[1]
            column_value = predict_tensor[:,:,:, tensor_index].cpu().numpy().reshape(-1,predict_tensor.shape[2])
        
        if getattr(column_conf, "denormalization", False):
            mean = data_tensor['meta_info'][conf.data.shared_columns.target]['mean'].cpu().numpy().reshape(-1)
            std = data_tensor['meta_info'][conf.data.shared_columns.target]['std'].cpu().numpy().reshape(-1)
            if len(mean.shape) != len(column_value.shape):
                mean = mean[:,None]
                std = std[:,None]
            column_value = column_value * std + mean
        if getattr(column_conf, "decoding", None):
            mapping = dataset_metas["encoding_mapping"][column_conf.decoding]
            reverse_mapping = {v: k for k, v in mapping.items()}
            column_value = list(map(reverse_mapping.get, column_value))
        if getattr(column_conf, "relu", False):
            column_value = np.maximum(column_value, 0)
        if getattr(column_conf, "to_datetime", False):
            column_value = pd.to_datetime(column_value)
        if source[0] == "total":
            column = [str(i) for i in range(predict_tensor.shape[2])]
        result[column] = column_value

    if hasattr(conf.test.predict, "filter_noise"):
        source = conf.test.predict.filter_noise
        tensor = source[0]
        tensor_index = source[1]
        noise_value = data_tensor[tensor][:, :, 0, tensor_index].cpu().numpy().reshape(-1)
        result = result[noise_value == 0.0]

    return result