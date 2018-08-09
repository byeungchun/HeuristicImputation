import random

import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger('HMLI')


def resize_timeseries_data(df, start_date='2010-1-1', end_date='2017-12-1'):
    """
    Remove any time series which has null observations from the dataframe

    Change data type to float
    :param df: BIS macro time series dataframe
    :param start_date: Start date
    :param end_date:  End date
    :return:
    """
    # Preprocessing 1 - save obs from Jan 2010 to Dec 2017
    df = df.loc[start_date:end_date]
    # Preprocessing 2 - remove TS have any missing values
    df = df.applymap(lambda x: np.nan if x is None or len(x) == 0 else x)
    df = df.dropna(axis=1, how='any')
    df = df.astype(float)
    # Preprocessing 3 - remove TS all same values
    df = df.loc[:, (df.diff().iloc[1:, :] != 0).all()]
    return df


def thinout_highly_correlated_timeseries(df, correlation_cutoff=0.97):
    """
    Remove similar series and remain low correlation series in the dataframe

    :param df: BIS macro time series dataframe
    :param correlation_cutoff: cutoff to remove similar time series
    :return: ditinguished time series dataframe
    """
    logger.debug('Number of TS before high correlation series removal: %d', df.shape[1])
    df_corr_matrix = df.corr()
    map_similar_ts = {}
    for i, v in enumerate(df_corr_matrix.columns):
        _res = list((df_corr_matrix[v][df_corr_matrix[v] > correlation_cutoff]).index)
        del _res[_res.index(v)]
        map_similar_ts[v] = _res
    lst_ts = []
    lst_ts_key = []
    for i, v in enumerate(map_similar_ts.keys()):
        if i == 0 or v not in lst_ts:
            lst_ts_key.append(v)
            lst_ts.extend(map_similar_ts[v])
    logger.debug('Number of TS after high correlation series removal: %d', len(lst_ts_key))
    return df[lst_ts_key]


def exec_value_normalization(df):
    """
    Times series value normalization for machine learning method

    :param df: BIS macro time series dataframe
    :return: Normalized BIS macro time series dataframe
    """
    arr_values = df.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    arr_scaled = scaler.fit_transform(arr_values)
    df_scaled = DataFrame(arr_scaled, index=df.index, columns=df.columns)
    return df_scaled


def create_regress_variables(df_scaled, num_prediction_period=3):
    """
    Create dependent and independent variables from BIS macro time series dataframe

    :param df_scaled: Scaled BIS time series
    :param num_prediction_period: number of prediction observations
    :return: time series code, x variable set, y variables, y test values
    """
    y_idx = random.randint(0, df_scaled.shape[1])
    ts_code = df_scaled.columns[y_idx]
    var_yall = df_scaled.iloc[:, y_idx]
    train_y = var_yall[:-num_prediction_period].values
    test_y = var_yall[-num_prediction_period:].values
    var_x = df_scaled[df_scaled.columns[df_scaled.columns != ts_code]]
    return ts_code, var_x, train_y, test_y


def create_regress_variables_with_missing_rate(df_scaled, missing_rate=0.1, idx_y=-1):
    """
    Create dependent and independent variables through missing rate

    :param df_scaled: Scaled BIS time series
    :param missing_rate: Missing data rate
    :param idx_y: Column index number for y variable
    :return: time series code, x variable set, y variables, y test values, index numbers for missing observations
    """
    logger.debug('Missing rate is %f', missing_rate)
    if idx_y == -1:
        y_idx = random.randint(0, df_scaled.shape[1])
    else:
        y_idx = idx_y
    ts_code = df_scaled.columns[y_idx]
    var_yall = df_scaled.iloc[:, y_idx]
    missing_idx = 0
    tot_obs = len(var_yall)
    lst_missing_idx = list()
    while True:
        missing_idx = missing_idx + np.ceil(np.random.exponential(1 / missing_rate))
        if missing_idx < tot_obs:
            lst_missing_idx.append(int(missing_idx))
        elif len(lst_missing_idx) == 0:
            missing_idx = 0
            continue
        else:
            break
    train_y = var_yall.drop(index=var_yall.index[lst_missing_idx])
    test_y = var_yall.iloc[lst_missing_idx]
    var_x = df_scaled[df_scaled.columns[df_scaled.columns != ts_code]]
    return ts_code, var_x, train_y, test_y, lst_missing_idx