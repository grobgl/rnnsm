import pickle
from churn_data import ChurnData
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, '../utils')
from plot_format import *
from seaborn import apionly as sns

_RESULT_PATH = '../../results/churn/cox_regression_log/'

predPeriod = {
    'start': pd.Timestamp('2016-02-01'),
    'end': pd.Timestamp('2016-06-01')
}
predPeriodHours = (predPeriod['end'] - predPeriod['start']) / np.timedelta64(1, 'h')
# predPeriodHours = 480


class CoxLogChurnModel:
    def __init__(self, data):
        self.cf = CoxPHFitter()
        self.data = data

    def fit(self, dataset, pred_col='deltaNextHours', event_col='observed'):
        dataset.deltaNextHours = np.log(dataset.deltaNextHours + 1)
        self.cf.fit(dataset, pred_col, event_col=event_col)

    def predict(self, df):
        # recency is scaled!
        # pred = self.cf.predict_expectation(df)
        pred = np.exp(self.cf.predict_expectation(self.data.split_val_df)) - 1
        churned = (pred.values.reshape(-1) - self.data.split_val_unscaled_df.recency.values.reshape(-1)) > predPeriodHours
        return churned.reshape(-1)

    def predict_proba(self, df):
        return np.zeros(len(df)*2).reshape((-1,2))


def storeModels():
    data = ChurnData(predict='deltaNextHours')#, features=['recency', 'logDeltaPrev_avg', 'logNumSessions'])
    model = CoxLogChurnModel(data)
    model.fit(data.split_train_df)

    with open(_RESULT_PATH+'model.pkl', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pred_val = model.cf.predict_expectation(data.split_val_df).values.reshape(-1)

    with open(_RESULT_PATH+'pred_val.pkl', 'wb') as handle:
        pickle.dump(pred_val, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 0.702
def scoreChurn(model):
    dataChurn = ChurnData()
    data.printScores(model, X=dataChurn.split_val['X'], y=dataChurn.split_val['y'])

def getMse():
    data = ChurnData(predict='deltaNextHours')#, features=['recency', 'logDeltaPrev_avg', 'logNumSessions'])
    observed = data.split_val_df.observed.values.astype('bool').reshape(-1)

    pred = np.exp(pred_val[observed]) - 1
    act = data.split_val['y'][observed]
    rmse = np.sqrt(((pred - act)**2).mean()) / 24

    print('RMSE: {:.2f}'.format(rmse))
    return rmse


def getAuc():
    data = ChurnData()
    roc_auc = roc_auc_score(data.split_val['y'], pred_val)

    print("AUC = {:.3f}".format(roc_auc))
    return roc_auc


def showJointPlot():
    data = ChurnData(predict='deltaNextHours')#, features=['recency', 'logDeltaPrev_avg', 'logNumSessions'])
    observed = data.split_val_df.observed.values.astype('bool').reshape(-1)
    pred_val = pickle.load(open(_RESULT_PATH+'pred_val.pkl', 'rb'))

    df = pd.DataFrame()
    df['predicted'] = pred_val[observed]
    df['actual'] = np.log(data.split_val['y'][observed] + 1)

    sns.jointplot('actual', 'predicted', data=df, kind='kde', size=figsize(.5,.5)[0])
    plt.show()

def showChurnedPred(width=1, height=None):
    data = ChurnData(predict='deltaNextHours')#, features=['recency', 'logDeltaPrev_avg', 'logNumSessions'])
    observed = data.split_val_df.observed.values.astype('bool').reshape(-1)
    pred_val = np.exp(pickle.load(open(_RESULT_PATH+'pred_val.pkl', 'rb'))) - 1

    returnDate = pred_val - data.split_val_unscaled_df.recency.values.reshape(-1)

    fig, ax = newfig(width, height)

    ax.hist(returnDate[observed], bins=50, label='not churned', alpha=.5)
    ax.hist(returnDate[~observed], bins=50, label='churned', alpha=.5)
    xDates = [pd.datetime(2015,i,1) for i in range(11,13)]+[pd.datetime(2016,i,1) for i in range(1,12)]
    xDatesHours = [(d - predPeriod['start']).to_timedelta64()/np.timedelta64(1,'h') for d in xDates]
    xDatesStr = [d.strftime('%Y-%m') for d in xDates]
    ax.set_xticks(xDatesHours)
    ax.set_xticklabels(xDatesStr)
    ax.axvline(x=predPeriodHours, label='prediction threshold' )
    ax.legend()

    fig.tight_layout()
    fig.show()



data = ChurnData(predict='deltaNextHours')#, features=['recency', 'logDeltaPrev_avg', 'logNumSessions'])
model = pickle.load(open(_RESULT_PATH+'model.pkl', 'rb'))
pred_val = pickle.load(open(_RESULT_PATH+'pred_val.pkl', 'rb'))


