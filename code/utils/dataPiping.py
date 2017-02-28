import datetime
import time
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from dateutil import parser
from sklearn.preprocessing import StandardScaler
import datetime
import time
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from dateutil import parser
from sklearn.preprocessing import StandardScaler

trainPeriod = ["2015-02-01", "2016-02-01"]

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


def trainTestSplitCust(X, y, test_size=0.33, random_state=42):
    train_cust, test_cust, y_train, y_test = X_test_split(y.keys(), y, test_size=test_size, random_state=random_state)
    
    X_train = X[X.customerId.isin(train_cust)]
    X_test = X[X.customerId.isin(test_cust)]
    
    return X_train, X_test, y_train, y_test


def getMergedSessionData():
    trainDF = pd.read_pickle('../../data/trainFebToFebCensoredMergedDF.pkl')
    targetDF = pd.read_pickle('../../data/trainFebToFebTargetsMergedDF.pkl')
    
    trainDF.reset_index(inplace=True)
    del trainDF['index']
    del trainDF['sessionId']
    
    targetSeq = targetDF.returnTime
    
    train, targets = normaliseData(trainDF, targetSeq)
    
    return train, targets


'''
    Aggregate users into categories listed below (use via aggrUsers)
'''
def _aggrUser(user):
    nSessions = len(user)
    period = makeunixtime(trainPeriod[1]) - user.startTime.values[0]
    frequency = 0
    if period > 0:
        frequency = nSessions/period
    recency = user.returnTime.values[-1]
    avgViewOnly = user.viewonly.sum()/nSessions
    avgChangeThumbnail = user.changeThumbnail.sum()/nSessions
    avgImageZoom = user.imageZoom.sum()/nSessions
    avgWatchVideo = user.watchVideo.sum()/nSessions
    avgView360 = user.view360.sum()/nSessions
    mobile = (user.device == 'mobile').sum()/nSessions
    desktop = (user.device == 'desktop').sum()/nSessions
    android = (user.device == 'android').sum()/nSessions
    ios = (user.device == 'ios').sum()/nSessions
    
    return pd.DataFrame({
        'nSessions': nSessions, 
        'period': period,
        'frequency': frequency,
        'recency': recency,
        'avgViewOnly': avgViewOnly,
        'avgChangeThumbnail': avgChangeThumbnail,
        'avgImageZoom': avgImageZoom,
        'avgWatchVideo': avgWatchVideo,
        'avgView360': avgView360,
        'device[mobile]': mobile,
        'device[desktop]': desktop,
        'device[android]': android,
        'device[ios]': ios
    },index=[0])

def aggrUsers(df):
    aggr = df.groupby('customerId').apply(_aggrUser)
    aggr.reset_index(inplace=True)
    del aggr['customerId']
    del aggr['level_1']
    return aggr


def filterNanReturnTimeUsers(X, y):
    nanTrain = y[y.isnull()].index

    return X[~X.customerId.isin(nanTrain)], y[~y.isnull()]


def getAggrSet(replace_nans=True):
    if replace_nans:
        X = pd.read_pickle('../../data/usersAggrNonanFebToFebXDF.pkl')
        y = pd.read_pickle('../../data/usersAggrNonanFebToFebYDF.pkl')
        return X, y
    else:
        X = pd.read_pickle('../../data/usersAggrFebToFebXDF.pkl')
        y = pd.read_pickle('../../data/usersAggrFebToFebYDF.pkl')
        return X, y


def splitAndNormaliseAggr(X, y, test_size=0.33, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    ss = StandardScaler()
    X_train_s = ss.fit_transform(X_train)
    X_test_s = ss.fit_transform(X_test)
    
    return X_train_s, X_test_s, y_train.values, y_test.values
