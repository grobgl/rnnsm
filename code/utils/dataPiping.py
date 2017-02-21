import datetime
import time
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from dateutil import parser

def makeunixtime(val):
    try:
        return int(time.mktime(parser.parse(val).timetuple()))
    except (OverflowError, AttributeError, ValueError):
        return None

def unixtimetostr(val):
    return datetime.datetime.fromtimestamp(int(val)).strftime('%Y-%m-%d %H:%M:%S')


def _discardNanStartUserTime(df,targets):
    nanStartUserTimeCust = df[df.startUserTime.isnull()].customerId.unique()
    return df[~df.customerId.isin(nanStartUserTimeCust)], targets[~targets.index.isin(nanStartUserTimeCust)]

def _calcStartUserTimeNan(g):
    isNan = g.startUserTime.isnull()
    
    if isNan.all():
        return g
    
    if isNan.any():
        timeOffset = g[~isNan].iloc[0].startUserTime - g[~isNan].iloc[0].startTime
        for i, row in g[isNan].iterrows():
            g.set_value(i, 'startUserTime', row.startTime + timeOffset)
        
    return g

def replaceNanStartUserTime(df, targets):
    df = df.groupby('customerId').apply(_calcStartUserTimeNan)
    return _discardNanStartUserTime(df, targets)

def normaliseData(df,targets):
    trainPeriod = ["2015-02-01", "2016-02-01"]
    trainDF = df.copy()
    trainTargets = targets.copy()
    
    trainDF, trainTargets = replaceNanStartUserTime(trainDF, trainTargets)
    
    trainDF['startTime'] = trainDF['startTime'] - makeunixtime(trainPeriod[0])
    trainDF['startUserTime'] = (trainDF['startUserTime'] - makeunixtime(trainPeriod[0])).astype(int)
    
    uniqueCustomerId = trainDF.customerId.unique()
    customerIdEnc = preprocessing.LabelEncoder().fit(uniqueCustomerId)
    
    trainDF['customerId'] = customerIdEnc.transform(trainDF.customerId)
    trainTargets.index = customerIdEnc.transform(trainTargets.index)
    
    return trainDF, trainTargets

def getTrainTest():
    trainDF = pd.read_pickle('../../data/trainFebToFebCensoredMergedDF.pkl')
    targetDF = pd.read_pickle('../../data/trainFebToFebTargetsMergedDF.pkl')
    
    trainDF.reset_index(inplace=True)
    del trainDF['index']
    del trainDF['sessionId']
    
    targetSeq = targetDF.returnTime
    
    train, targets = normaliseData(trainDF, targetSeq)
    
    train_cust, test_cust, y_train, y_test = train_test_split(targets.keys(), targets, test_size=0.33, random_state=42)
    
    X_train = train[train.customerId.isin(train_cust)]
    X_test = train[train.customerId.isin(test_cust)]
    
    return X_train, X_test, y_train, y_test 