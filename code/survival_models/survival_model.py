import pickle
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from GPyOpt.methods import BayesianOptimization

from multiprocessing import Pool
from functools import partial
import sys
sys.path.insert(0, '../utils')
sys.path.insert(0, '../churn-prediction')
from churn_data import ChurnData, getChurnScores
from plot_format import *
import seaborn as sns
from seaborn import apionly as sns


predPeriod = {
    'start': pd.Timestamp('2016-02-01'),
    'end': pd.Timestamp('2016-06-01')
}
predPeriodHours = (predPeriod['end'] - predPeriod['start']) / np.timedelta64(1, 'h')

data = ChurnData(predict='deltaNextHours')

class SurvivalModel:
    def __init__(self):
        self.data = data

    def fit(self, dataset, indices=None):
        if indices is not None:
            dataset = dataset.iloc[indices]

        dataset = dataset.copy()
        dataset.deltaNextHours = self.transformTargets(dataset.deltaNextHours)
        self.cf.fit(dataset, 'deltaNextHours', event_col='observed')


    def transformTargets(self, targets):
        return targets


    def reverseTransformTargets(self, targets):
        return targets


    def predict_expectation(self, indices=None, dataset='train'):
        df = self.data.train_df

        if dataset=='test':
            df = self.data.test_df

        if indices is None:
            indices = self.data.split_val_ind

        x_df = df.iloc[indices]

        pred = self.reverseTransformTargets(self.cf.predict_expectation(x_df))

        return pred.values.reshape(-1)


    def predict_churn(self, pred_durations, indices=None, dataset='train'):
        df_unscaled = self.data.train_unscaled_df

        if dataset=='test':
            df_unscaled = self.data.test_unscaled_df

        if indices is None:
            indices = self.data.split_val_ind

        x_df_unscaled = df_unscaled.iloc[indices]

        churned = (pred_durations - x_df_unscaled.recency.values.reshape(-1)) > predPeriodHours

        return churned.reshape(-1)


    def getScores(self, indices=None, dataset='train'):
        df = self.data.train_df

        if dataset=='test':
            df = self.data.test_df

        if indices is None:
            indices = self.data.split_val_ind

        df = df.iloc[indices]

        pred_durations = self.predict_expectation(indices, dataset)
        pred_churn = self.predict_churn(pred_durations, indices, dataset)

        return {'churn': getChurnScores(~df.observed, pred_churn, pred_durations),
                'rmse_days': np.sqrt(mean_squared_error(df.deltaNextHours, pred_durations)) / 24,
                'concordance': concordance_index(df.deltaNextHours, pred_durations, df.observed)}


def runParameterSearch(model):
    """
    Cross-validated search for parameters

    """
    nFolds = 2
    nPools = 8
    bounds = [(0,20000)]
    max_iter = 10

    print(model.RESULT_PATH)

    # load churn data for splitting fold stratas
    churnData = ChurnData()

    pool = Pool(nPools)

    cv = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=42)
    splits = np.array(list(cv.split(**churnData.train)))

    f = partial(_evaluatePenalizer, model=model, splits=splits, pool=pool)
    bOpt = BayesianOptimization(f=f, bounds=bounds)

    bOpt.run_optimization(max_iter=max_iter)

    pool.close()

    with open(model.RESULT_PATH+'bayes_opt.pkl', 'wb') as handle:
        pickle.dump(bOpt, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return bOpt

def _evaluatePenalizer(penalizer, model=None, splits=None, pool=None):
    scores = pool.map(
            partial(_runParameterSearch, model=model, penalizer=penalizer),
            splits)
    # scores = list(map(
    #         partial(_runParameterSearch, model=model, penalizer=penalizer),
    #         splits))

    return np.mean(scores)

def _runParameterSearch(splits, model=None, penalizer=None):
    train_ind, test_ind = splits
    model = model(penalizer=penalizer[0][0])
    model.fit(model.data.train_df, indices=train_ind)

    return model.getScores(test_ind)['concordance']


def storeModel(model, **model_params):
    data = ChurnData(predict='deltaNextHours')
    m = model(data, **model_params)
    m.fit(data.split_train_df)

    with open(model.RESULT_PATH+'model.pkl', 'wb') as handle:
        pickle.dump(m, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pred_val = m.cf.predict_expectation(data.split_val_df).values.reshape(-1)

    with open(model.RESULT_PATH+'pred_val.pkl', 'wb') as handle:
        pickle.dump(pred_val, handle, protocol=pickle.HIGHEST_PROTOCOL)


def showJointPlot(model, width=1, height=None):
    observed = model.data.split_val_df.observed.values.astype('bool').reshape(-1)
    pred_val = model.predict_expectation()

    df = pd.DataFrame()
    df['predicted'] = pred_val[observed] / 24
    df['actual'] = data.split_val['y'][observed] / 24

    jointgrid = sns.jointplot('actual', 'predicted', data=df, kind='resid', size=figsize(.5,.5)[0])

    plt.show()


# up to acc 0.712
def showChurnedPred(model, width=1, height=None):
    observed = model.data.split_val_df.observed.values.astype('bool').reshape(-1)
    pred_val = model.predict_expectation()

    returnDate = pred_val - data.split_val_unscaled_df.recency.values.reshape(-1)

    fig, ax = newfig(width, height)

    ax.hist(returnDate[observed], bins=50, label='not churned', alpha=.5)
    ax.hist(returnDate[~observed], bins=50, label='churned', alpha=.5)
    xDates = [pd.datetime(2016,i,1) for i in range(1,12)]
    xDatesHours = [(d - predPeriod['start']).to_timedelta64()/np.timedelta64(1,'h') for d in xDates]
    xDatesStr = [d.strftime('%Y-%m') for d in xDates]
    ax.set_xticks(xDatesHours)
    ax.set_xticklabels(xDatesStr)
    ax.axvline(x=predPeriodHours, label='prediction threshold' )
    ax.legend()

    fig.tight_layout()
    fig.show()


