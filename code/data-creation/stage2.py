import numpy as np
import pandas as pd
from multiprocessing import Pool

obsPeriod = {
    'start': pd.Timestamp('2015-02-02'),
    'end': pd.Timestamp('2016-02-02')
}

def createStage2():
    df = pd.read_pickle('../../data/cleaned/stage1_startUserTimes.pkl')
    df = df[~df.customerId.isin(df[df.startUserTime.isnull()].customerId.unique())]
    df = appendSessionTimeMetrics(df)
    df = combineDailySessionsPar(df)
    return df

def createStage2_pruned():
    df = pd.read_pickle('../../data/cleaned/stage1_pruned.pkl')
    df = df[~df.customerId.isin(df[df.startUserTime.isnull()].customerId.unique())]
    df = appendSessionTimeMetrics(df)
    df = combineDailySessionsPar(df)
    return df

def createStage2_obs_pred():
    df = pd.read_pickle('../../data/cleaned/stage1_obs_pred.pkl')
    df = df[~df.customerId.isin(df[df.startUserTime.isnull()].customerId.unique())]
    df = appendSessionTimeMetrics(df)
    df = combineDailySessionsPar(df)
    return df

def appendSessionTimeMetrics(df):
    startUserTimeIdx = pd.DatetimeIndex(df.startUserTime)
    df['startUserDate'] = startUserTimeIdx.date
    df['recency'] = obsPeriod['end'] - df.startUserTime
    df['hourOfDay'] = startUserTimeIdx.hour
    df['dayOfWeek'] = startUserTimeIdx.dayofweek
    df['dayOfMonth'] = startUserTimeIdx.day
    df['weekOfYear'] = startUserTimeIdx.week
    df['month'] = startUserTimeIdx.month
    df.sessionLength = pd.TimedeltaIndex(pd.to_timedelta(df.sessionLength))
    df.startUserTimeDelta = pd.to_timedelta(df.startUserTimeDelta)
    return df


def combineDailySessions(df):
    df['numSessions'] = 0
    df = df.groupby('customerId').apply(_combineDatesUser)
    df = df.groupby('customerId').apply(_addReturnTimeUser)
    # return df.reset_index(drop=True)
    return df

def _combineDatesUser(group):
    return group.groupby('startUserDate').apply(_combineDatesDate)

def _combineDatesDate(group):
    longestSessionIdx = group.sessionLength.idxmax()

    group.numSessions.values[0] = len(group)
    group.hourOfDay.values[0] = group.loc[longestSessionIdx].hourOfDay
    group.device.values[0] = group.loc[longestSessionIdx].device
    group.startUserTime.values[0] = group.startUserTime.min()
    group.endTime.values[0] = group.endTime.max()
    group.endUserTime.values[0] = group.endUserTime.max()
    group.sessionLength.values[0] = np.timedelta64(group.sessionLength.sum(),'ns')
    group.sessionLengthSec.values[0] = group.sessionLengthSec.sum()
    group.recency.values[0] = np.timedelta64(group.recency.min(),'ns')
    group.changeThumbnail.values[0] = group.changeThumbnail.sum()
    group.imageZoom.values[0] = group.imageZoom.sum()
    group.watchVideo.values[0] = group.watchVideo.sum()
    group.view360.values[0] = group.view360.sum()
    group.numberdivisions.values[0] = group.numberdivisions.max()
    group.avgPrice[0] = group.loc[longestSessionIdx].avgPrice
    group.viewonly[0] = group.viewonly.max()

    return group.head(1)

def _addReturnTimeUser(group):
    group = group.sort_values('startUserDate')
    group['deltaNext'] = group.startUserDate.shift(-1) - group.startUserDate
    group['deltaPrev'] = -(group.startUserDate.shift(1) - group.startUserDate)

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

def combineDailySessionsPar(df):
    df['numSessions'] = 0
    return parallelizeDataframe(df, combineDailySessions)
