from churn_data import ChurnData
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

import sys
sys.path.insert(0, '../utils')
from plot_format import *


predPeriod = {
    'start': pd.Timestamp('2016-02-01'),
    'end': pd.Timestamp('2016-06-01')
}
predPeriodHours = (predPeriod['end'] - predPeriod['start']) / np.timedelta64(1, 'h')

class CoxChurnModel:
    def __init__(self, data):
        self.cf = CoxPHFitter()
        self.data = data

    def fit(self, dataset, pred_col='deltaNextHours', event_col='observed'):
        self.cf.fit(dataset, pred_col, event_col=event_col)

    def predict(self, df):
        # recency is scaled!
        # pred = self.cf.predict_expectation(df)
        pred = self.cf.predict_expectation(self.data.split_val_df)
        churned = (pred.values.reshape(-1) - self.data.split_val_unscaled_df.recency.values.reshape(-1)) > predPeriodHours
        return churned.reshape(-1)

    def predict_proba(self, df):
        return np.zeros(len(df)*2).reshape((-1,2))


data = ChurnData(predict='deltaNextHours', features=['recency', 'logDeltaPrev_avg', 'logNumSessions'])
model = CoxChurnModel(data)

def scoreChurn(model, data):
    dataChurn = ChurnData()
    data.printScores(model, X=dataChurn.split_val['X'], y=dataChurn.split_val['y'])
