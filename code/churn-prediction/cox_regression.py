from churn_data import ChurnData
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

import sys
sys.path.insert(0, '../utils')


predPeriod = {
    'start': pd.Timestamp('2016-02-01'),
    'end': pd.Timestamp('2016-06-01')
}
predPeriodHours = (predPeriod['end'] - predPeriod['start']) / np.timedelta64(1, 'h')

class CoxChurnModel:
    def __init__(self):
        self.cf = CoxPHFitter()

    def fit(self, dataset, pred_col='deltaNextHours', event_col='observed'):
        self.cf.fit(dataset, pred_col, event_col=event_col)

    def predict(self, df):
        pred = self.cf.predict_expectation(df)
        churned = (pred - df.recency.values.reshape((-1,1))) > predPeriodHours
        return churned.values.reshape(-1)

    def predict_proba(self, df):
        return np.zeros(len(df))

model = CoxChurnModel()
data = ChurnData(dataset='cox')
