import numpy as np
import pandas as pd
import patsy as pt

obsPeriod = {
    'start': pd.Timestamp('2015-02-01'),
    'end': pd.Timestamp('2016-02-01')
}

actPeriod = {
    'start': pd.Timestamp('2015-10-01'),
    'end': pd.Timestamp('2016-02-01')
}

predPeriod = {
    'start': pd.Timestamp('2016-02-01'),
    'end': pd.Timestamp('2016-06-01')
}

PATH = '../../data/rnn/first/'


def createChurnRnnDS():
    """
    Creates dataset using same time frames as churn DS, without summarizing sessions. Used for RNN models.
    """
    df = pd.read_pickle('../../data/cleaned/stage2_obs_pred.pkl')

    # only look at sessions in obs period
    df_obs = df[df.startUserTime < obsPeriod['end']]
    cust = df_obs.customerId.unique()
    df = df_obs

    # customers with session in act period
    actCust = df[(df.startUserTime >= actPeriod['start']) &
                 (df.startUserTime < actPeriod['end'])].customerId.unique()
    df = df[df.customerId.isin(actCust)]

    # extract temporal features
    startUserTimeIdx = pd.DatetimeIndex(df.startUserTime)
    df['hourOfDay'] = startUserTimeIdx.hour
    df['dayOfWeek'] = startUserTimeIdx.dayofweek
    df['dayOfMonth'] = startUserTimeIdx.day

    # aggregate interactions
    df['numInteractions'] = df.changeThumbnail + df.imageZoom + df.watchVideo + df.view360

    # set type for timestamp values
    # df['deltaNextHours'] = df.deltaNext / np.timedelta64(1,'h')
    # df['deltaPrevHours'] = df.deltaPrev / np.timedelta64(1,'h')
    df['deltaNextHours'] = -1
    df.loc[~df.deltaNext.isnull(), 'deltaNextHours'] = df.loc[~df.deltaNext.isnull(), 'deltaNext'].apply(lambda x: x.days * 24)
    df.loc[df.deltaNext.isnull(), 'deltaNextHours'] = pd.NaT
    df['deltaPrevHours'] = -1
    df.loc[~df.deltaPrev.isnull(), 'deltaPrevHours'] = df.loc[~df.deltaPrev.isnull(), 'deltaPrev'].apply(lambda x: x.days * 24)
    df.loc[df.deltaPrev.isnull(), 'deltaPrevHours'] = pd.NaT


    # set deltaNextHours to time until end of observation window
    df['churned'] = (df.deltaNextHours.isnull() & (df.startUserTime < obsPeriod['end'])).astype('int')
    df.startUserDate = df.startUserDate.apply(pd.Timestamp)
    df.loc[df.churned.astype('bool'), 'deltaNextHours'] = (predPeriod['end'] - df.startUserDate[df.churned.astype('bool')]) / np.timedelta64(1, 'h')

    # set deltaNextHours to 0 for sessions in prediction window
    df.loc[df.startUserTime >= predPeriod['start'], 'deltaNextHours'] = 0

    df.loc[df.deltaPrevHours < 0, 'deltaPrevHours'] = 0
    df.loc[df.deltaPrevHours.isnull(), 'deltaPrevHours'] = 0

    return df

    # select features
    features = ['customerId', 'churned', 'deltaNextHours', 'deltaPrevHours', 'numInteractions',
                'numberdivisions', 'avgPrice', 'viewonly', 'changeThumbnail',
                'imageZoom', 'watchVideo', 'view360', 'device', 'sessionLengthSec',
                'hourOfDay', 'dayOfWeek', 'dayOfMonth']

    df_onehot = pt.dmatrix('+'.join(features)+'-1', df, return_type='dataframe', NA_action='raise')
    df_onehot['startUserTime'] = df.startUserTime
    df_onehot['startUserTimeHours'] = (df_onehot['startUserTime'] - obsPeriod['start']) / np.timedelta64(1,'h')
    df_onehot['churned'] = df.churned.astype('bool')
    df_onehot['device'] = df.device


    return df_onehot

