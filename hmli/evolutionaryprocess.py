import logging
import random
import sys

from pandas import DataFrame
from sklearn import linear_model, svm
from sklearn.metrics import mean_squared_error
import logging

logger = logging.getLogger('HMLI')


def generate_chrom(var_x, map_chrom, num_chrom_per_pop, num_gene_per_chrom, lst_chrom_log, uniq_num_chrom):
    """Create chromosome per each generations
    :param var_x: Dependent variable candidates dataframe
    :param map_chrom: Chromosome container
    :param num_chrom_per_pop: Number of chromosomes in one population
    :param num_gene_per_chrom: Number of time series in one chromosome
    :param lst_chrom_log: historical chromosome list to prevent dulplicated chromosome generation
    :param uniq_num_chrom: Number of unique chromosome
    :return: Chromosome container, Number of unique chromosome
    """
    while_cnt = 0  # Max while loop
    if len(map_chrom) > 0:  # In this case, it needs mutation process
        lst_gene = list()
        for i in map_chrom.keys():
            lst_gene.extend(map_chrom[i])
        set_gene = set(lst_gene)
        for i in range(100):
            if int(num_chrom_per_pop * 0.9) <= len(map_chrom):
                break  # because mapChrom has enough chroms.
            else:
                chrom = random.sample(set_gene, num_gene_per_chrom)
                if chrom not in lst_chrom_log:
                    map_chrom[uniq_num_chrom] = chrom
                    lst_chrom_log.append(chrom)
                    uniq_num_chrom = uniq_num_chrom + 1
    while len(map_chrom) != num_chrom_per_pop and while_cnt < num_chrom_per_pop * 100:
        while_cnt = while_cnt + 1
        chrom = random.sample(range(var_x.shape[1]), num_gene_per_chrom)
        chrom.sort()
        if chrom not in lst_chrom_log:
            map_chrom[uniq_num_chrom] = chrom
            lst_chrom_log.append(chrom)
            uniq_num_chrom = uniq_num_chrom + 1
    if len(map_chrom) != num_chrom_per_pop:
        logger.error('number of chromosome is less than ' + str(num_chrom_per_pop))
        sys.exit(-1)
    return map_chrom, uniq_num_chrom


def fit_curve_through_linear_regression(train_x, train_y, test_x, test_y):
    """Fitting curve through OLS regression and calculate MSE between test and actual values
    :param train_x:
    :param train_y:
    :param test_x:
    :param test_y:
    :return: Mean square error value
    """
    model = linear_model.LinearRegression()
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    val_mse = mean_squared_error(test_y, pred_y)
    return val_mse


def fit_curve_through_support_vector_machine(train_x, train_y, test_x, test_y):
    """itting curve through support vector machine and calculate MSE between test and actual values
    :param train_x:
    :param train_y:
    :param test_x:
    :param test_y:
    :return: Mean square error value
    """
    model = svm.SVR()
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    val_mse = mean_squared_error(test_y, pred_y)
    return val_mse


def remove_recessive_chromosome(map_mse, map_chrom, cutoff_rate):
    """Remove recessive(high error value) chromosome
    :param map_mse: Mean square error array
    :param map_chrom: Chromosome array
    :param cutoff_rate:
    """
    map_mse = sorted(map_mse.items(), key=lambda kv: kv[1], reverse=True)
    num_cutoff = int(len(map_mse) * cutoff_rate)
    for x in range(num_cutoff):
        del map_chrom[map_mse[x][0]]


def find_highly_correlate_series(df_scaled, ts_code, lst_missing_idx):
    """Find most highly corredated series and calcluate MSE between y variable and the correlated series
    :param df_scaled:
    :param ts_code:
    :param lst_missing_idx:
    :return: Correlated value
    """
    df2 = df_scaled.drop(index=df_scaled.index[lst_missing_idx])
    corr = df2.corr()
    highly_corr_ts = corr.loc[ts_code, :].sort_values(ascending=False)[:2].index
    corr_vals = df_scaled.loc[:, highly_corr_ts].iloc[lst_missing_idx, :].values
    logger.debug('corrVal: %s', str(corr_vals.round(2).tolist()))
    corr_mse = mean_squared_error(corr_vals[:, 0], corr_vals[:, 1])
    return corr_mse


def exec_genetic_algorithm(var_x, train_y, test_y, num_gene_per_chrom, num_chrom_per_pop, num_population,
                           num_pred_period, cutoff_rate, lst_missing_idx):
    """
    :param var_x:
    :param train_y:
    :param test_y:
    :param num_gene_per_chrom:
    :param num_chrom_per_pop:
    :param num_population:
    :param num_pred_period:
    :param cutoff_rate:
    :param lst_missing_idx:
    :return:
    """
    lst_report = list()  # nGen, nChrom,
    lst_chrom_log = list()  # all chromosomes for all generations
    uniq_num_chrom = 0
    for iGen in range(num_population):
        if iGen == 0: map_chrom = dict()  # all chromosomes for one population
        # else: #Create mutation
        map_chrom, uniq_num_chrom = generate_chrom(var_x, map_chrom, num_chrom_per_pop, num_gene_per_chrom,
                                                   lst_chrom_log, uniq_num_chrom)
        map_mse = dict()
        for iChrom, mapChromKey in enumerate(map_chrom):
            train_x = var_x.drop(index=var_x.index[lst_missing_idx])
            train_x = train_x.iloc[:, map_chrom[mapChromKey]].values
            test_x = var_x.iloc[lst_missing_idx, map_chrom[mapChromKey]].values
            map_mse[mapChromKey] = fit_curve_through_support_vector_machine(train_x, train_y, test_x, test_y)
            lst_report.append(
                [iGen, iChrom, mapChromKey, map_chrom[mapChromKey], len(lst_missing_idx), map_mse[mapChromKey]])
        logging.debug('%d generation', iGen)
        remove_recessive_chromosome(map_mse, map_chrom, cutoff_rate)
    df_rpt = DataFrame(lst_report, columns=['nGen', 'nChrom', 'uChromKey', 'chromosome', 'nMissingValues', 'mseVal'])
    return df_rpt
