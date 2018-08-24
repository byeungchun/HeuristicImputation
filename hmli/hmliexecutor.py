import random

from hmli.datapreprosessor import create_regress_variables_with_missing_rate
from hmli.evolutionaryprocess import exec_genetic_algorithm, find_highly_correlate_series
import logging

logger = logging.getLogger('HMLI')


def execute_hmli(a_series_idx, df_scaled, p1, p2, p3, p4, p5, p6, p7):
    """
    Execusion to IFC2018 presentation test. It is 3,070 BIS macro time series

    :param a_series_idx:
    :param df_scaled:
    :return: Summary report
    """
    num_prediction_period = p1  # 12
    num_gene_per_chrom = p2  # 6
    num_chrom_per_pop = p3  # 10
    num_population = p4  # 100
    cutoff_chrom_drop_rate = p5  # 0.5
    lst_random_seed = p6  # [1, 2, 3]
    lst_res = list()
    for numSeed in lst_random_seed:
        random.seed(numSeed)
        for missing_rate in p7:  # [0.1, 0.4, 0.7]:
            ts_code, var_x, train_y, test_y, lst_missing_idx = \
                create_regress_variables_with_missing_rate(df_scaled, missing_rate, a_series_idx)
            logger.info('Variable Y: %s, missingRate: %f', df_scaled.columns[a_series_idx], missing_rate)
            df_rpt = exec_genetic_algorithm(var_x, train_y, test_y, num_gene_per_chrom, num_chrom_per_pop,
                                            num_population, num_prediction_period, cutoff_chrom_drop_rate,
                                            lst_missing_idx)
            aggregation_function = {'mseVal': ['min', 'median', 'mean', 'max']}
            res = {'Yseries': df_scaled.columns[a_series_idx], 'numSeed': numSeed,
                   'cutoffRate': cutoff_chrom_drop_rate, 'missingRate': missing_rate,
                   'numMissingValue': len(lst_missing_idx), 'numChromPerPop': num_chrom_per_pop,
                   'numGenePerChrom': num_gene_per_chrom,
                   'corrMSE': find_highly_correlate_series(df_scaled, ts_code, lst_missing_idx),
                   'msePerGen': df_rpt.groupby(by='nGen').agg(aggregation_function)}
            lst_res.append(res)
    return lst_res
