# -*- coding: utf-8 -*-
"""
HMLS time series data handler
=============================

"""

import logging
import random
import pickle
import concurrent.futures
from itertools import repeat

from hmli.datahandler import save_dataframe_to_hdf5, load_monthly_time_series_from_hdf5
from hmli.datapreprosessor import resize_timeseries_data, thinout_highly_correlated_timeseries, \
    exec_value_normalization, create_regress_variables_with_missing_rate
from hmli.evolutionaryprocess import find_highly_correlate_series, exec_genetic_algorithm

logger = logging.getLogger('HMLI')
logger.setLevel(logging.INFO)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s"))
logger.addHandler(consoleHandler)
logger.propagate = False


def exec_test_for_ifc2018(a_series_idx, df_scaled):
    """Execusion to IFC2018 presentation test. It is 3,070 BIS macro time series
    :param a_series_idx:
    :param df_scaled:
    :return: Summary report
    """
    cutoff_correlation = 0.97
    num_prediction_period = 12
    num_gene_per_chrom = 6
    num_chrom_per_pop = 10
    num_population = 100
    cutoff_chrom_drop_rate = 0.5
    lst_random_seed = [1, 2, 3]
    lst_res = list()
    for numSeed in lst_random_seed:
        random.seed(numSeed)
        for missing_rate in [0.1, 0.4, 0.7]:
            ts_code, var_x, train_y, test_y, lst_missing_idx = \
                create_regress_variables_with_missing_rate(df_scaled, missing_rate, a_series_idx)
            logger.info('Variable Y: %s, missingRate: %f', df_scaled.columns[a_series_idx], missing_rate)
            df_rpt = exec_genetic_algorithm(var_x, train_y, test_y, num_gene_per_chrom, num_chrom_per_pop,
                                            num_population, num_prediction_period, cutoff_chrom_drop_rate,
                                            lst_missing_idx)
            aggregation_function = {'mseVal': ['min', 'median', 'mean', 'max']}
            res = {'Yseries': df_scaled.columns[a_series_idx], 'numSeed': numSeed, 'corrCutoff': cutoff_correlation,
                   'cutoffRate': cutoff_chrom_drop_rate, 'missingRate': missing_rate,
                   'numMissingValue': len(lst_missing_idx), 'numChromPerPop': num_chrom_per_pop,
                   'numGenePerChrom': num_gene_per_chrom,
                   'corrMSE': find_highly_correlate_series(df_scaled, ts_code, lst_missing_idx),
                   'msePerGen': df_rpt.groupby(by='nGen').agg(aggregation_function)}
            lst_res.append(res)
    return lst_res


def testDh(needPreproc=True, isThread=False):
    dataDir = r'C:\Users\by003457\workspace\data\ifc'
    rawFile = 'bismacromonthly.txt'
    h5File = r'C:\Users\by003457\workspace\data\ifc\bismacro.h5'
    corrVal = 0.97
    numPredPeriod = 12
    # missingRate = 3
    numGenePerChrom = 6
    numChromPerPop = 10
    numPopulation = 100
    cutoffRate = 0.5
    lstSeed = [1, 2, 3]
    if needPreproc:
        df = load_monthly_time_series_from_hdf5(h5File)
        preprocDf = resize_timeseries_data(df)
        disctinctDf = thinout_highly_correlated_timeseries(preprocDf, corrVal)
        scaledDf = exec_value_normalization(disctinctDf)
        save_dataframe_to_hdf5(scaledDf, h5File, 'scaledDf')
        return scaledDf
    else:
        if isThread:
            lstSeriesIdx = list()
            scaledDf = load_monthly_time_series_from_hdf5(h5File, 'scaledDf')
            for i in range(scaledDf.shape[1]):
                lstSeriesIdx.append(i)
            finalRes = list()
            # Thread

            # with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            # res = {executor.submit(exeConcurrentFunc, aSeriesIdx, scaledDf): aSeriesIdx for aSeriesIdx in lstSeriesIdx}
            # for future in concurrent.futures.as_completed(res):
            #   finalRes.append(future.result())
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for res in executor.map(exec_test_for_ifc2018, lstSeriesIdx, repeat(scaledDf)):
                    finalRes.append(res)
        else:
            scaledDf = load_monthly_time_series_from_hdf5(h5File, 'scaledDf')
            finalRes = list()
            # numY = random.sample([ x for x in range(scaledDf.shape[1])], 100)
            for numSeed in lstSeed:
                random.seed(numSeed)
                for missingRate in [0.1, 0.4, 0.7]:
                    for i in range(10):  # range(scaledDf.shape[1]):
                        # tsCode, varX, trainY, testY = genVariables(scaledDf, numPredPeriod)
                        tsCode, varX, trainY, testY, lstMissingIdx = create_regress_variables_with_missing_rate(scaledDf, missingRate,
                                                                                                                i)
                        logger.info('Variable Y: %s, missingRate: %f', scaledDf.columns[i], missingRate)
                        dfRpt = exec_genetic_algorithm(varX, trainY, testY, numGenePerChrom, numChromPerPop, numPopulation,
                                                       numPredPeriod, cutoffRate, lstMissingIdx)
                        aggFunc = {'mseVal': ['min', 'median', 'mean', 'max']}
                        res = {'Yseries': scaledDf.columns[i], 'numSeed': numSeed, 'corrCutoff': corrVal,
                               'cutoffRate': cutoffRate, 'missingRate': missingRate,
                               'numMissingValue': len(lstMissingIdx), 'numChromPerPop': numChromPerPop,
                               'numGenePerChrom': numGenePerChrom,
                               'corrMSE': find_highly_correlate_series(scaledDf, tsCode, lstMissingIdx),
                               'msePerGen': dfRpt.groupby(by='nGen').agg(aggFunc)}
                        finalRes.append(res)

    return finalRes


def main():
    finalRes = testDh(False, True)
    with open(r'C:\Users\by003457\workspace\data\ifc\finalRes1.pickle', 'wb') as handle:
        pickle.dump(finalRes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # scaledDf = testDh()
    # finalRes = testDh(False)
    # for res in finalRes:
    #    fig = plt.figure()
    #    ax1 = fig.add_subplot(111)
    #    r2 = res['corrMSE']
    #    r1 = res['msePerGen']['mseVal']['mean']
    #    ax1.plot(r1.index,r1.values)
    #    ax1.plot(r1.index, [r2 for x in range(len(r1.index))])
    #    plt.show()


if __name__ == '__main__':
    main()

