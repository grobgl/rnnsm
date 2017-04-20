import sys
sys.path.insert(0, '../utils')
from dataPiping import makeunixtime

import pandas as pd
import numpy as np
from multiprocessing import Pool

def createStage1():
    df = loadData()
    df = parallelizeDataframe(df, addTimeIndices)
    df = unifyBrowsers(df)
    df.sort('startTime',inplace=True)
    df = parallelizeDataframe(df, appendReturnTime)

    return df

def pruneLongSessionUsers(df):
    longUsers = df[df.sessionLengthSec > 10e2].customerId.unique()
    return df[~df.customerId.isin(longUsers)]

def createStage1_obs_pred():
    obs_timefr = [pd.datetime(2015,2,1), pd.datetime(2016,2,1)]
    pred_timefr = [pd.datetime(2016,2,1), pd.datetime(2016,6,1)]
    stage1 = pd.read_pickle('../../data/cleaned/stage1_pruned.pkl')

    stage1_obs_pred = stage1[(stage1.startUserTime >= obs_timefr[0]) & \
                             (stage1.startUserTime < pred_timefr[1])].copy()
    obs_cust = stage1_obs_pred[stage1_obs_pred.startUserTime < obs_timefr[1]].customerId.unique()
    stage1_obs_pred = stage1_obs_pred[stage1_obs_pred.customerId.isin(obs_cust)]

    df = parallelizeDataframe(stage1_obs_pred, appendReturnTime)
    return df


def addMissingLocalTimes(df):
    df['localTimeOffset'] = df.startUserTime - df.startTime
    noLocalTimeCust = df[df.startUserTime.isnull()].customerId.unique()
    grouped = df[df.customerId.isin(noLocalTimeCust)].groupby('customerId')

    # use median offset of four (or fewer) closest sessions with given offset
    for cust, sessions in grouped:
        ix = sessions.index
        nanIx = sessions[sessions.startUserTime.isnull()].index
        for i in nanIx:
            remIx = ix.drop(nanIx.drop(i))
            pos = np.where(remIx==i)[0][0]
            values = remIx[_getWindow(len(remIx), pos)]
            offsetMedian = sessions.localTimeOffset.ix[values].median()
            df.loc[i, 'startUserTime'] = sessions.loc[i, 'startTime'] + offsetMedian

    del df['localTimeOffset']
    return df

def _getWindow(arrLen, i, length=4):
    """_getWindow: Creates sliding window in list of length arrLen around index i
    """
    leftCt = i
    rightCt = arrLen - i - 1
    left = min(leftCt, max(2, length-rightCt))
    right = min(rightCt, max(2, length-leftCt))
    return list(range(i-left, i)) + list(range(i+1, i+right+1))

# read raw data
def loadData():
    """Loads data into df
    """
    df = pd.read_pickle('../../data/sessionDF.pkl')

    return df

def addTimeIndices(df):
    # convert session start time to pandas time index
    df.startTime = pd.DatetimeIndex(df.startTime.apply(makeunixtime) * 1000000000)
    df.startUserTime = pd.DatetimeIndex(df.startUserTime.apply(makeunixtime) * 1000000000)
    df.sessionLength = pd.TimedeltaIndex(df.sessionLength * 1000000000)

    df['sessionLengthSec'] = df.sessionLength / np.timedelta64(1,'s')
    df['startTimeDelta'] = pd.DatetimeIndex(df.startTime) - pd.Timestamp('2015-01-01')
    df['startUserTimeDelta'] = pd.DatetimeIndex(df.startUserTime) - pd.Timestamp('2015-01-01')
    df['endTime'] = df.startTime + df.sessionLength
    df['endUserTime'] = df.startUserTime + df.sessionLength

    return df


def unifyBrowsers(df):
    """Cleans browser column
    """
    df.loc[df.device.isnull(), 'device'] = 'unknown'
    df.loc[df.device.str.contains('desktop', case=False), 'device'] = 'desktop'
    df.loc[df.device.str.contains('tesktop', case=False), 'device'] = 'desktop'
    df.loc[df.device.str.contains('mobile', case=False), 'device'] = 'mobile'
    df.loc[df.device.str.contains('android', case=False), 'device'] = 'android'
    df.loc[df.device.str.contains('ios', case=False), 'device'] = 'ios'
    return df


def appendReturnTime(df):
    df['deltaPrev'] = 0
    df['deltaNext'] = 0
    return df.groupby('customerId').apply(_appendReturnTime)

def _appendReturnTime(group):
    group['deltaNext'] = group.startTime.shift(-1) - group.endTime
    group['deltaPrev'] = -(group.endTime.shift(1) - group.startTime)
    return group

def parallelizeDataframe(df, func, num_cores=8):
    df['partition'] = ((df.customerId // 100) % 16) // 2

    pt = df.groupby('partition')
    df_split = [pt.get_group(x) for x in pt.groups]

    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))

    pool.close()
    pool.join()
    return df


