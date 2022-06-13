import datetime
import numpy as np
import pandas as pd
def fill_zero_target_with_history_mean(df, start_date, end_date, abandon_dates, his_dates):
    length = df.shape[1]
    # days before first day
    first_sales_day = length - ((df>0)*np.arange(length, 0, -1)).max(axis=1).values
    df[first_sales_day>abandon_dates] = np.nan

    for i in range(length):
        df.iloc[first_sales_day>i, i] = np.nan

    # fill with mean
    for i in pd.date_range(start_date+datetime.timedelta(his_dates+7),end_date):
        na_index = df[i].isna()
        df.loc[na_index, i] = get_time_span(df[na_index], i, his_dates-7, his_dates/7, '7D').mean(axis=1).values
    
    return df

def get_history_mean(df, start_date, end_date, his_dates):
    for i in pd.date_range(start_date+datetime.timedelta(his_dates), end_date)[::-1]:
        df[i] = get_time_span(df, i, his_dates, his_dates, 'D').mean(axis=1).values
    df.loc[:,:his_dates] = np.nan
    return df

def get_time_span(df, anchor, back, periods, freq='D'):
    return df[pd.date_range(anchor-datetime.timedelta(back), periods=periods, freq=freq)]

def cal_mape(pred, actual):
    esp = 1e-9
    pred = np.array(pred)
    actual = np.array(actual)
    mape = (np.abs(pred - actual).sum()) / (np.abs(actual).sum() + esp)
    return mape

def cal_smape(pred, actual):
    esp = 1e-9
    pred = np.array(pred)
    actual = np.array(actual)
    smape = 2 * (np.abs(pred - actual).sum()) / ((np.abs(pred)+np.abs(actual)).sum() + esp)
    return smape