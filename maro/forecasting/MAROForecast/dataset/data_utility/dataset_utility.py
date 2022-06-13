import pandas as pd
from itertools import product

def load_from_file(file_paths, parse_date=None):
    parse_dates = [] if parse_date == None else [parse_date]
    df_raws = []
    for file_path in file_paths:
        if file_path.endswith(".csv"):
            df_raws.append(pd.read_csv(file_path, parse_dates=parse_dates))
        elif file_path.endswith(".tsv"):
            df_raws.append(pd.read_csv(file_path, parse_dates=parse_dates, sep="\t"))
        elif file_path.endswith(".xlsx"):
            df_raws.append(pd.read_excel(file_path, parse_dates=parse_dates))
            raise NotImplementedError 
    return pd.concat(df_raws, axis=0)

# Fill columns' product
def fill_missing_row(df_raw, encoding_mapping, fill_missing_row_conf):
    if fill_missing_row_conf.function == "combination":
        columns = fill_missing_row_conf.columns
        columns_value = []
        for column in columns:
            if column in encoding_mapping.keys():
                columns_value.append(encoding_mapping[column].values())
            else:
                columns_value.append(df_raw[column].unique())
        df_all_combination = pd.DataFrame(
            data=product(*columns_value),
            columns=columns
        )
        df_raw = df_raw.merge(
            right=df_all_combination,
            how='right',
            on=columns
        )
    else:
        raise NotImplementedError
    return df_raw

# Encoding by data
def encode(df_raw, encoding_conf, origin_mapping_dic):
    column = encoding_conf.column
    mapping = {}
    origin_mapping = origin_mapping_dic[encoding_conf.column]
    mapping.update(origin_mapping)
    df_raw[column].fillna('nan', inplace = True)
    target_items = list(df_raw[column].unique())
    new_items = sorted(set(target_items) - set(mapping.keys()), key=target_items.index)
    if encoding_conf.function == "indexing_from_1":
        begin_index = len(mapping) + 1
        mapping.update({value: index + begin_index for index, value in enumerate(new_items)})
    else:
        raise NotImplementedError
    df_raw[column] = [mapping[item] for item in df_raw[column]]
    return mapping, df_raw

# Encoding from encoded
def encode_from_encoded(df_raw, encoding_conf, encoded_mapping):
    columns = encoding_conf.columns
    if getattr(encoding_conf, "filter_no_encoded", False):
        mask = pd.Series(True, index=df_raw.index)
        for column in columns:
            mask &= df_raw[column].isin(encoded_mapping.keys())
        df_raw = df_raw[mask]
    else:
        raise NotImplementedError
    for column in columns:
        df_raw[column] = [encoded_mapping[item] for item in df_raw[column]]
    return df_raw

# Filter noise
def filter_noise(conf, df_raw, dataset_conf):
    filter_noise = dataset_conf.filter_noise
    if filter_noise.function == "filter_when_no_target":
        target_column = conf.data.shared_columns.target
        mask = pd.isna(df_raw[target_column])
    else:
        raise NotImplementedError
    if filter_noise.action == "set_node_to_na":
        for column in conf.data.shared_columns.nodes:
            df_raw[column][mask] = pd.NA
    else:
        raise NotImplementedError
    noise_column = conf.data.shared_columns.noise
    df_raw[noise_column] = [False] * len(mask)
    df_raw[noise_column][mask] = True
    return df_raw
