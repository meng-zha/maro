import os
from dataset.data_utility.dataset_utility import *
from dataset.walmart_dataset import *

# Load train, val and test set
def load_and_split(conf, root_path, dataset_conf):
    data_column = conf.data.shared_columns.date
    parse_date = data_column if getattr(dataset_conf, "parse_date", False) else None
    subset_conf = dataset_conf.subset
    splited_dataset_dic = {}

    if subset_conf.split_type == "split_by_field":
        file_paths = [os.path.join(root_path, file_name) for file_name in subset_conf.shared_files]
        df_raw = load_from_file(file_paths, parse_date)
        mode_dic = vars(subset_conf.mode)
        for mode in mode_dic:
            splited_dataset_dic[mode] = df_raw[(df_raw[subset_conf.split_field] >= mode_dic[mode].start) & (df_raw[subset_conf.split_field] <= mode_dic[mode].end)]
    elif subset_conf.split_type == "split_by_file":
        mode_dic = vars(subset_conf.mode)
        for mode in mode_dic:
            file_paths = [os.path.join(root_path, file_name) for file_name in mode_dic[mode]]
            df_raw = load_from_file(file_paths, parse_date)
            splited_dataset_dic[mode] = df_raw
    elif subset_conf.split_type == "no_split":
        file_paths = [os.path.join(root_path, file_name) for file_name in subset_conf.shared_files]
        df_raw = load_from_file(file_paths, parse_date)
        mode_list = ["train", "val", "test"]
        for mode in mode_list:
            splited_dataset_dic[mode] = df_raw
    else:
        raise NotImplementedError
    return splited_dataset_dic


def build_dataset(conf, encoding_dic = None, data_update = None):
    dataset_dic = {}
    if encoding_dic is None:
        encoding_mapping_dic = {}
    else:
        encoding_mapping_dic = encoding_dic
    data_raw_dic = {}
    for dataset_conf in conf.data.dataset:
        conf.logger.info("Load dataset " + dataset_conf.name)
        if data_update is not None and data_update.get(dataset_conf.name, False) is not False:
            splited_dataset_dic = {}
            mode_list = ["train", "val", "test"]
            for mode in mode_list:
                splited_dataset_dic[mode] = data_update[dataset_conf.name].copy()
        else:
            splited_dataset_dic = load_and_split(conf, conf.data.root_path, dataset_conf)
        data_raw_dic[dataset_conf.name] = splited_dataset_dic
        # Preprocess subsets in each mode
        for mode_name in splited_dataset_dic:
            df_mode_set = splited_dataset_dic[mode_name]

            # Backup columns
            backup_columns =  getattr(dataset_conf, "backup_columns", [])
            for column in backup_columns:
                df_mode_set[column + "_ori"] = df_mode_set[column].copy(deep=True)

            # Encoding
            if hasattr(dataset_conf, "encoding"):
                for encoding_conf in dataset_conf.encoding:       
                    if encoding_conf.column not in encoding_mapping_dic:
                        encoding_mapping_dic[encoding_conf.column] = {}             
                    new_mapping, df_mode_set = encode(df_mode_set, encoding_conf, encoding_mapping_dic)
                    encoding_mapping_dic[encoding_conf.column] = new_mapping
            
    for dataset_conf in conf.data.dataset:
        conf.logger.info("Load dataset " + dataset_conf.name)
        splited_dataset_dic = data_raw_dic[dataset_conf.name]
        for mode_name in splited_dataset_dic:
            df_mode_set = splited_dataset_dic[mode_name]
            if hasattr(dataset_conf, "encode_from_encoded"):
                for encoding_conf in dataset_conf.encode_from_encoded:
                    df_mode_set = encode_from_encoded(
                        df_mode_set, 
                        encoding_conf, 
                        encoding_mapping_dic[encoding_conf.name]
                    )

            # Fill missing combinations
            if hasattr(dataset_conf, "fill_missing_row"):
                df_mode_set = fill_missing_row(df_mode_set, encoding_mapping_dic, dataset_conf.fill_missing_row)

            # Sort by columns
            if hasattr(dataset_conf, "sort_by"):
                df_mode_set = df_mode_set.sort_values(by=dataset_conf.sort_by, ascending=True)

            # Filter noise
            if hasattr(dataset_conf, "filter_noise"):
                df_mode_set = filter_noise(conf, df_mode_set, dataset_conf)

            if mode_name not in dataset_dic:
                dataset_dic[mode_name] = []
            dataset_dic[mode_name].append({"raw_data": df_mode_set, "name": dataset_conf.name,\
                 "type": dataset_conf.type, "encoding_mapping": encoding_mapping_dic})

    # Send to target dataset and genertate to total dataset 
    conf.logger.info("Process dataset into " + conf.data.name)
    total_dataset_dic = {mode_set: eval(conf.data.name)(conf, dataset_dic[mode_set]) for mode_set in dataset_dic}
    conf.logger.info(conf.data.name + " is ready")
    return total_dataset_dic