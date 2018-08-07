# -*- coding: utf-8 -*-
"""
HMLS time series data handler
=============================

"""

import sys, logging
import random
import pickle
import numpy as np
import concurrent.futures
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn import svm
from itertools import repeat

from hmli.datahandler import save_dataframe_to_hdf5, load_monthly_time_series_from_hdf5

logger = logging.getLogger('HMLI')
logger.setLevel(logging.INFO)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s"))
logger.addHandler(consoleHandler)
logger.propagate = False


def preprocessDf(df, startDt='2010-1-1', toDt='2017-12-1'):
    # Preprocessing 1 - save obs from Jan 2010 to Dec 2017
    df2 = df.loc[startDt:toDt]
    # Preprocessing 2 - remove TS have any missing values
    df2 = df2.applymap(lambda x: np.nan if x is None or len(x) == 0 else x)
    df2 = df2.dropna(axis=1, how='any')
    df2 = df2.astype(float)
    # Preprocessing 3 - remove TS all same values
    df2 = df2.loc[:, (df2.diff().iloc[1:, :] != 0).all()]
    return df2


# Remove similar time series which has high correlation
def removeSimilarSeries(df, corrVal=0.97):
    logger.debug('Number of TS before high correlation series removal: %d', df.shape[1])
    df2 = df.corr()
    mapSimilarTS = {}
    for i, v in enumerate(df2.columns):
        _res = list((df2[v][df2[v] > corrVal]).index)
        del _res[_res.index(v)]
        mapSimilarTS[v] = _res
    lstTs = []
    lstTsFinal = []
    for i, v in enumerate(mapSimilarTS.keys()):
        if i == 0:
            lstTsFinal.append(v)
            lstTs.extend(mapSimilarTS[v])
        else:
            if v not in lstTs:
                lstTsFinal.append(v)
                lstTs.extend(mapSimilarTS[v])
    logger.debug('Number of TS after high correlation series removal: %d', len(lstTsFinal))
    return df[lstTsFinal]


def genNormalizedVariables(df):
    values = df.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    scaledDf = DataFrame(scaled, index=df.index, columns=df.columns)
    return scaledDf


def genVariables(scaledDf, numPredPeriod=3):
    yIdx = random.randint(0, scaledDf.shape[1])
    tsCode = scaledDf.columns[yIdx]
    varYall = scaledDf.iloc[:, yIdx]
    trainY = varYall[:-numPredPeriod].values
    testY = varYall[-numPredPeriod:].values
    varX = scaledDf[scaledDf.columns[scaledDf.columns != tsCode]]
    return tsCode, varX, trainY, testY


def genVariablesWithMissingVals(scaledDf, missingRate=0.1, pickY=-1):
    logger.debug('Missing rate is %f', missingRate)
    if pickY == -1:
        yIdx = random.randint(0, scaledDf.shape[1])
    else:
        yIdx = pickY
    tsCode = scaledDf.columns[yIdx]
    varYall = scaledDf.iloc[:, yIdx]
    missingIdx = 0
    totObs = len(varYall)
    lstMissingIdx = list()
    while True:
        missingIdx = missingIdx + np.ceil(np.random.exponential(1 / missingRate))
        if missingIdx < totObs:
            lstMissingIdx.append(int(missingIdx))
        elif len(lstMissingIdx) == 0:
            missingIdx = 0
            continue
        else:
            break
    trainY = varYall.drop(index=varYall.index[lstMissingIdx])
    testY = varYall.iloc[lstMissingIdx]
    varX = scaledDf[scaledDf.columns[scaledDf.columns != tsCode]]
    return tsCode, varX, trainY, testY, lstMissingIdx


def genChromosome(varX, mapChrom, numChromPerPop, numGenePerChrom, lstChromLog, uniqNumChrom):
    whileCnt = 0  # Max while loop
    if len(mapChrom) > 0:  # In this case, it needs mutation process
        lstGene = list()
        for i in mapChrom.keys():
            lstGene.extend(mapChrom[i])
        setGene = set(lstGene)
        for i in range(100):
            if int(numChromPerPop * 0.9) <= len(mapChrom):
                break  # because mapChrom has enough chroms.
            else:
                chrom = random.sample(setGene, numGenePerChrom)
                if chrom not in lstChromLog:
                    mapChrom[uniqNumChrom] = chrom
                    lstChromLog.append(chrom)
                    uniqNumChrom = uniqNumChrom + 1
    while len(mapChrom) != numChromPerPop and whileCnt < numChromPerPop * 100:
        whileCnt = whileCnt + 1
        chrom = random.sample(range(varX.shape[1]), numGenePerChrom)
        chrom.sort()
        if chrom not in lstChromLog:
            mapChrom[uniqNumChrom] = chrom
            lstChromLog.append(chrom)
            uniqNumChrom = uniqNumChrom + 1
    if len(mapChrom) != numChromPerPop:
        logger.error('number of chromosome is less than ' + str(numChromPerPop))
        sys.exit(-1)
    return mapChrom, uniqNumChrom


def calcLinearModel(trainX, trainY, testX, testY):
    model = linear_model.LinearRegression()
    model.fit(trainX, trainY)
    predY = model.predict(testX)
    valMse = mean_squared_error(testY, predY)
    return valMse


def calcSvmModel(trainX, trainY, testX, testY):
    model = svm.SVR()
    model.fit(trainX, trainY)
    predY = model.predict(testX)
    valMse = mean_squared_error(testY, predY)
    return valMse


def removeRecessive(mapMse, mapChrom, cutoffRate):
    mapMse = sorted(mapMse.items(), key=lambda kv: kv[1], reverse=True)
    numCutoff = int(len(mapMse) * cutoffRate)
    for x in range(numCutoff):
        del mapChrom[mapMse[x][0]]


def compSimpleMethod(scaledDf, tsCode, lstMissingIdx):
    df2 = scaledDf.drop(index=scaledDf.index[lstMissingIdx])
    corr = df2.corr()
    highlyCorrTs = corr.loc[tsCode, :].sort_values(ascending=False)[:2].index
    corrVals = scaledDf.loc[:, highlyCorrTs].iloc[lstMissingIdx, :].values
    logger.debug('corrVal: %s', str(corrVals.round(2).tolist()))
    corrMse = mean_squared_error(corrVals[:, 0], corrVals[:, 1])
    return corrMse


def exeGeneticAlgorithm(varX, trainY, testY, numGenePerChrom, numChromPerPop, numPopulation, numPredPeriod, cutoffRate,
                        lstMissingIdx):
    lstReport = list()  # nGen, nChrom,
    lstChromLog = list()  # all chromosomes for all generations
    uniqNumChrom = 0
    for iGen in range(numPopulation):
        if iGen == 0: mapChrom = dict()  # all chromosomes for one population
        # else: #Create mutation
        mapChrom, uniqNumChrom = genChromosome(varX, mapChrom, numChromPerPop, numGenePerChrom, lstChromLog,
                                               uniqNumChrom)
        mapMse = dict()
        for iChrom, mapChromKey in enumerate(mapChrom):
            # trainX = varX.iloc[:-numPredPeriod, mapChrom[mapChromKey]].values
            # testX = varX.iloc[-numPredPeriod:,mapChrom[mapChromKey]].values
            trainX = varX.drop(index=varX.index[lstMissingIdx])
            trainX = trainX.iloc[:, mapChrom[mapChromKey]].values
            testX = varX.iloc[lstMissingIdx, mapChrom[mapChromKey]].values
            # mapMse[mapChromKey] = calcLinearModel(trainX, trainY, testX, testY)
            mapMse[mapChromKey] = calcSvmModel(trainX, trainY, testX, testY)
            lstReport.append(
                [iGen, iChrom, mapChromKey, mapChrom[mapChromKey], len(lstMissingIdx), mapMse[mapChromKey]])
        logging.debug('%d generation', iGen)
        removeRecessive(mapMse, mapChrom, cutoffRate)
    dfRpt = DataFrame(lstReport, columns=['nGen', 'nChrom', 'uChromKey', 'chromosome', 'nMissingValues', 'mseVal'])
    return dfRpt


def exeConcurrentFunc(aSeriesIdx, scaledDf):
    corrVal = 0.97
    numPredPeriod = 12
    # missingRate = 3
    numGenePerChrom = 6
    numChromPerPop = 10
    numPopulation = 100
    cutoffRate = 0.5
    lstSeed = [1, 2, 3]
    lstRes = list()
    for numSeed in lstSeed:
        random.seed(numSeed)
        for missingRate in [0.1, 0.4, 0.7]:
            tsCode, varX, trainY, testY, lstMissingIdx = genVariablesWithMissingVals(scaledDf, missingRate, aSeriesIdx)
            logger.info('Variable Y: %s, missingRate: %f', scaledDf.columns[aSeriesIdx], missingRate)
            dfRpt = exeGeneticAlgorithm(varX, trainY, testY, numGenePerChrom, numChromPerPop, numPopulation,
                                        numPredPeriod, cutoffRate, lstMissingIdx)
            aggFunc = {'mseVal': ['min', 'median', 'mean', 'max']}
            res = {'Yseries': scaledDf.columns[aSeriesIdx], 'numSeed': numSeed, 'corrCutoff': corrVal,
                   'cutoffRate': cutoffRate, 'missingRate': missingRate, 'numMissingValue': len(lstMissingIdx),
                   'numChromPerPop': numChromPerPop,
                   'numGenePerChrom': numGenePerChrom, 'corrMSE': compSimpleMethod(scaledDf, tsCode, lstMissingIdx),
                   'msePerGen': dfRpt.groupby(by='nGen').agg(aggFunc)}
            lstRes.append(res)
    return lstRes


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
        preprocDf = preprocessDf(df)
        disctinctDf = removeSimilarSeries(preprocDf, corrVal)
        scaledDf = genNormalizedVariables(disctinctDf)
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
                for res in executor.map(exeConcurrentFunc, lstSeriesIdx, repeat(scaledDf)):
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
                        tsCode, varX, trainY, testY, lstMissingIdx = genVariablesWithMissingVals(scaledDf, missingRate,
                                                                                                 i)
                        logger.info('Variable Y: %s, missingRate: %f', scaledDf.columns[i], missingRate)
                        dfRpt = exeGeneticAlgorithm(varX, trainY, testY, numGenePerChrom, numChromPerPop, numPopulation,
                                                    numPredPeriod, cutoffRate, lstMissingIdx)
                        aggFunc = {'mseVal': ['min', 'median', 'mean', 'max']}
                        res = {'Yseries': scaledDf.columns[i], 'numSeed': numSeed, 'corrCutoff': corrVal,
                               'cutoffRate': cutoffRate, 'missingRate': missingRate,
                               'numMissingValue': len(lstMissingIdx), 'numChromPerPop': numChromPerPop,
                               'numGenePerChrom': numGenePerChrom,
                               'corrMSE': compSimpleMethod(scaledDf, tsCode, lstMissingIdx),
                               'msePerGen': dfRpt.groupby(by='nGen').agg(aggFunc)}
                        finalRes.append(res)

    return finalRes


def main():
    # scaledDf = readMacroH5(r'C:\Users\by003457\workspace\data\ifc\bismacro.h5', 'scaledDf')
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

