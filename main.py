# -*- coding: utf-8 -*-

import logging
import pickle
import json
import concurrent.futures
import matplotlib.pyplot as plt
from configparser import ConfigParser
from itertools import repeat

from hmli.datahandler import save_dataframe_to_hdf5, load_monthly_time_series_from_hdf5
from hmli.datapreprosessor import resize_timeseries_data, thinout_highly_correlated_timeseries, exec_value_normalization
from hmli.hmliexecutor import execute_hmli

logger = logging.getLogger('HMLI')
logger.setLevel(logging.INFO)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s"))
logger.addHandler(consoleHandler)
logger.propagate = False


def generate_normalized_df(hdf5_file, cutoff_correlation):
    """
    Generate normalized time series dataframe

    :param cutoff_correlation: remove similar series(highly correlated) more than 0.75 correlation coefficient
    :param hdf5_file:
    :return: Normalized dataframe
    """
    df = load_monthly_time_series_from_hdf5(hdf5_file)
    df_preprocessed = resize_timeseries_data(df)
    df_uncorrelated_ts = thinout_highly_correlated_timeseries(df_preprocessed, cutoff_correlation)
    df_scaled = exec_value_normalization(df_uncorrelated_ts)
    save_dataframe_to_hdf5(df_scaled, hdf5_file, 'scaledDf')
    return df_scaled


def exec_hmli(hdf5_file, cutoff_correlation, need_preprocessing,p1,p2,p3,p4,p5,p6,p7):
    """
    HMLI executor

    :param hdf5_file:
    :param cutoff_correlation:
    :param need_preprocessing:
    :return:
    """
    lst_series_idx = list()
    lst_final_result = list()
    if need_preprocessing:
        generate_normalized_df(hdf5_file, cutoff_correlation)
    df_scaled = load_monthly_time_series_from_hdf5(hdf5_file, 'scaledDf')
    # Test
    #for i in range(10):
    for i in range(df_scaled.shape[1]):
        lst_series_idx.append(i)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for res in executor.map(execute_hmli, lst_series_idx, repeat(df_scaled),repeat(p1),repeat(p2),
                                repeat(p3), repeat(p4),repeat(p5),repeat(p6),repeat(p7)):
            lst_final_result.append(res)

    return lst_final_result


def save_final_result(lst_final_result, pickle_final_result):
    """
    Save HMLI executor result to pickle format file

    :param lst_final_result:
    :param pickle_final_result:
    """
    with open(pickle_final_result, 'wb') as handle:
        pickle.dump(lst_final_result, handle, protocol=pickle.HIGHEST_PROTOCOL)


def draw_rmse_plot(lst_final_result):
    """
    Draw rmse plot by iteration for all testing series

    :param lst_final_result:
    """
    for res in lst_final_result:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        r2 = res['corrMSE']
        r1 = res['msePerGen']['mseVal']['mean']
        ax1.plot(r1.index, r1.values)
        ax1.plot(r1.index, [r2 for x in range(len(r1.index))])
        plt.show()


def main():
    config = ConfigParser()
    config.read('hmli.ini')
    p1 = config.getint('ModelParameter', 'number of prediction observations')
    p2 = config.getint('ModelParameter', 'number of genes per chromosome')
    p3 = config.getint('ModelParameter', 'number of chromosomes per population')
    p4 = config.getint('ModelParameter', 'number of populations')
    p5 = config.getfloat('ModelParameter', 'cutoff rate of bad chromosome')
    p6 = json.loads(config.get('ModelParameter', 'random seed list'))
    p7 = json.loads(config.get('ModelParameter', 'missing rate list'))

    hdf5_file = config.get('ModelParameter', 'hdf5 data file') #r'data/mei.h5'
    pickle_final_result = config.get('ModelParameter', 'result file') #r'output/finalRes1.pickle'
    cutoff_correlation = config.getfloat('ModelParameter','cutoff correlation coefficient') # 0.97
    need_preprocessing = False

    lst_final_result = exec_hmli(hdf5_file, cutoff_correlation, need_preprocessing,p1,p2,p3,p4,p5,p6,p7)
    save_final_result(lst_final_result, pickle_final_result)
    # draw_rmse_plot(lst_final_result)


if __name__ == '__main__':
    main()

