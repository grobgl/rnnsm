import numpy as np
import pandas as pd

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
    'mid': pd.Timestamp('2016-04-01'),
    'end': pd.Timestamp('2016-06-01')
}


def createChurnDS():
    df = pd.read_pickle('../../data/churn/churn.pkl')
    df['returnTime'] = ((df.deltaNextHours - df.recency) * np.timedelta64(1,'h')) + obsPeriod['end']
    df['deltaNextHoursFull'] = df.deltaNextHours
    df['churnedFull'] = df.churned

    df['churned'] = (df['returnTime'] > predPeriod['mid'])
    df.loc[df.churned, 'deltaNextHours'] = df.loc[df.churned, 'deltaNextHours'] - \
            (df.loc[df.churned, 'returnTime'] - predPeriod['mid'])/np.timedelta64(1,'h')
    df['churned'] = df.churned.astype('float64')

    del df['returnTime']

    return df

