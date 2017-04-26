import numpy as np
import pandas as pd
from multiprocessing import Pool

obsPeriod = {
    'start': pd.Timestamp('2015-02-01'),
    'end': pd.Timestamp('2016-02-01')
}

actPeriod = { 'start': pd.Timestamp('2015-10-01'),
    'end': pd.Timestamp('2016-02-01')
}

predPeriod = {
    'start': pd.Timestamp('2016-02-01'),
    'end': pd.Timestamp('2016-06-01')
}


features = ['numSessions', 'deviceAndroid', 'deviceIos', 'deviceDesktop', 'deviceMobile', 'deviceUnknown',
            'deviceAndroid_wght', 'deviceIos_wght', 'deviceDesktop_wght', 'deviceMobile_wght', 'deviceUnknown_wght',
            'recency', 'deltaPrev_avg', 'deltaPrev_wght_avg', 'dayOfMonth_avg',
            'dayOfMonth_wght_avg', 'dayOfWeek_avg', 'dayOfWeek_wght_avg', 'hourOfDay_avg',
            'hourOfDay_wght_avg', 'sessionLen_avg', 'sessionLen_wght_avg', 'price_avg',
            'price_wght_avg', 'numInteractions_avg', 'numInteractions_wght_avg', 'numItemsViewed_avg',
            'numItemsViewed_wght_avg', 'numDivisions_avg', 'numDivisions_wght_avg', 'deltaNextHours', 'observed']


def createCoxDS():
    df = pd.read_pickle('../../data/cleaned/stage1_obs_pred.pkl')

    # only look at sessions in obs period
    df = df[df.startUserTime < obsPeriod['end']]

    # customers with session in act period
    actCust = df[(df.startUserTime >= actPeriod['start']) & (df.startUserTime < actPeriod['end'])].customerId.unique()

    df = df[df.customerId.isin(actCust)]

    df = appendSessionTimeMetrics(df)

    df = parallelizeDataframe(df, aggregateCust)

    df = df[['customerId'] + features]

    # add log features
    df['logNumSessions'] = np.log(df.numSessions)
    df['logDeltaPrev_avg'] = np.log(df.deltaPrev_avg + 1)
    df['logDeltaPrev_wght_avg'] = np.log(df.deltaPrev_wght_avg + 1)
    f['logSessionLen_avg'] = np.log(df.sessionLen_avg + 1)
    df['logSessionLen_wght_avg'] = np.log(df.sessionLen_wght_avg + 1)
    df['logPrice_avg'] = np.log(df.price_avg)
    df['logPrice_wght_avg'] = np.log(df.price_wght_avg)
    df['logNumDivisions_avg'] = np.log(df.numDivisions_avg)
    df['logNumDivisions_wght_avg'] = np.log(df.numDivisions_wght_avg)
    df['logNumInteractions_avg'] = np.log(df.numInteractions_avg + 1)
    df['logNumInteractions_wght_avg'] = np.log(df.numInteractions_wght_avg + 1)
    df['logNumItemsViewed_avg'] = np.log(df.numItemsViewed_avg + 1)
    df['logNumItemsViewed_wght_avg'] = np.log(df.numItemsViewed_wght_avg + 1)

    return df


def appendSessionTimeMetrics(df):
    for feat in features:
        df[feat] = 0.

    # recency in hours
    df['recency'] = (obsPeriod['end'] - df.startUserTime) / np.timedelta64(1,'h')
    df['deltaPrevHours'] = df.deltaPrev/np.timedelta64(1,'h')
    df['hourOfDay'] = df.startUserTime.dt.hour
    df['dayOfWeek'] = df.startUserTime.dt.dayofweek
    df['dayOfMonth'] = df.startUserTime.dt.day
    df['numInteractions'] = df.changeThumbnail + df.imageZoom + df.watchVideo + df.view360

    return df


def aggregateCust(df):
    df = df.groupby('customerId').apply(_aggregateCust)
    return df

def _aggregateCust(sess):
    n = len(sess)
    invRecency = 1/sess.recency
    longestSessionIdx = sess.sessionLength.idxmax()

    sess['numSessions'].values[0] = n
    sess['deviceAndroid'].values[0] = (sess.device=='android').sum() / n
    sess['deviceAndroid_wght'].values[0] = invRecency[sess.device=='android'].sum() / invRecency.sum()
    sess['deviceIos'].values[0] = (sess.device=='ios').sum() / n
    sess['deviceIos_wght'].values[0] = invRecency[sess.device=='ios'].sum() / invRecency.sum()
    sess['deviceDesktop'].values[0] = (sess.device=='desktop').sum() / n
    sess['deviceDesktop_wght'].values[0] = invRecency[sess.device=='desktop'].sum() / invRecency.sum()
    sess['deviceMobile'].values[0] = (sess.device=='mobile').sum() / n
    sess['deviceMobile_wght'].values[0] = invRecency[sess.device=='mobile'].sum() / invRecency.sum()
    sess['deviceUnknown'].values[0] = (sess.device=='unknown').sum() / n
    sess['deviceUnknown_wght'].values[0] = invRecency[sess.device=='unknown'].sum() / invRecency.sum()
    sess['recency'].values[0] = sess.recency.values[-1]
    sess['dayOfMonth_avg'].values[0] = sess.dayOfMonth.mean()
    sess['dayOfMonth_wght_avg'].values[0] = np.average(sess.dayOfMonth, weights=invRecency)
    sess['dayOfWeek_avg'].values[0] = sess.dayOfWeek.mean()
    sess['dayOfWeek_wght_avg'].values[0] = np.average(sess.dayOfWeek, weights=invRecency)
    sess['hourOfDay_avg'].values[0] = sess.hourOfDay.mean()
    sess['hourOfDay_wght_avg'].values[0] = np.average(sess.hourOfDay, weights=invRecency)
    sess['sessionLen_avg'].values[0] = sess.sessionLengthSec.mean()
    sess['sessionLen_wght_avg'].values[0] = np.average(sess.sessionLengthSec, weights=invRecency)
    sess['price_avg'].values[0] = sess.avgPrice.mean()
    sess['price_wght_avg'].values[0] = np.average(sess.avgPrice, weights=invRecency)

    if not sess.deltaNext.tail(1).isnull().values[0]:
        sess['deltaNextHours'].values[0] = max(0,sess.deltaNext.values[0] / np.timedelta64(1,'h'))
    else:
        sess['deltaNextHours'].values[0] = (predPeriod['end'] - sess.startUserTime.values[-1]) / np.timedelta64(1,'h')

    sess['numItemsViewed_avg'].values[0] = sess.viewonly.mean()
    sess['numItemsViewed_wght_avg'].values[0] = np.average(sess.viewonly, weights=invRecency)
    sess['numDivisions_avg'].values[0] = sess.numberdivisions.mean()
    sess['numDivisions_wght_avg'].values[0] = np.average(sess.numberdivisions, weights=invRecency)

    # leave deltaPrev at 0 if only a single session
    if n > 1:
        sess['deltaPrev_avg'].values[0] = sess.deltaPrevHours.mean()
        sess['deltaPrev_wght_avg'].values[0] = np.average(sess.deltaPrevHours.dropna(), weights=invRecency[~sess.deltaPrev.isnull()])

    # leave numInteractions at 0 if no web sessions
    webFilter = sess.device.isin(['desktop','mobile'])
    if webFilter.sum() > 0:
        sess['numInteractions_avg'].values[0] = sess[webFilter].numInteractions.mean()
        sess['numInteractions_wght_avg'].values[0] = np.average(sess[webFilter].numInteractions, weights=invRecency[webFilter])

    # label customer
    sess['observed'].values[0] = int(sess.deltaNext.tail(1).isnull().values[0])

    return sess.head(1)


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

