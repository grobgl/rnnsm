import pickle
import pandas as pd
import numpy as np
from scipy.integrate import trapz
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index, _get_index, qth_survival_times
from sklearn.metrics import roc_auc_score, mean_squared_error, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.gaussian_process.kernels import Matern
from bayes_opt import BayesianOptimization

from multiprocessing import Pool
from functools import partial
from churn_data import ChurnData, getChurnScores
import sys
sys.path.insert(0, '../utils')
# from plot_format import *
# from seaborn import apionly as sns


predPeriod = {
    'start': pd.Timestamp('2016-02-01'),
    'end': pd.Timestamp('2016-06-01')
}
predPeriodHours = (predPeriod['end'] - predPeriod['start']) / np.timedelta64(1, 'h')
hours_year = np.timedelta64(pd.datetime(2017,2,1) - pd.datetime(2016,2,1)) / np.timedelta64(1,'h')
# predPeriodHours = 2700


class SurvivalModel:
    def __init__(self, include_recency=False):
        self.data = ChurnData(predict='deltaNextHours')#, features=['recency', 'logNumSessions'])
        self.include_recency = include_recency

    def fit(self, dataset, indices=None):
        if indices is not None:
            dataset = dataset.iloc[indices]

        dataset = dataset.copy()
        dataset.deltaNextHours = self.transformTargets(dataset.deltaNextHours)
        self.cf.fit(dataset, 'deltaNextHours', event_col='observed', show_progress=False)


    def transformTargets(self, targets):
        return targets


    def reverseTransformTargets(self, targets):
        return targets


    def predict_expectation(self, indices=None, dataset='train'):
        if self.include_recency:
            return self._predict_expectation_recency_C(indices, dataset)

        df = self.data.train_df

        if dataset=='test':
            df = self.data.test_df

        if indices is None:
            indices = self.data.split_val_ind

        x_df = df.iloc[indices]
        index = _get_index(x_df)

        # pred = self.reverseTransformTargets(self.cf.predict_median(x_df))
        # pred[np.isinf(pred)] = 2*predPeriodHours
        # pred = qth_survival_times(.5, self._predict_survival_function(indices, dataset)[index])
        # pred = self.reverseTransformTargets(pred)
        pred = self.reverseTransformTargets(self.cf.predict_expectation(x_df))

        return pred.values.reshape(-1)


    def _predict_expectation_recency_A(self, indices=None, dataset='train'):
        df = self.data.train_df
        df_unscaled = self.data.train_unscaled_df

        if dataset=='test':
            df = self.data.test_df
            df_unscaled = self.data.test_unscaled_df

        if indices is None:
            indices = self.data.split_val_ind

        x_df = df.iloc[indices]
        x_df_unscaled = df_unscaled.iloc[indices]
        recency = self.transformTargets(x_df_unscaled.recency)

        index = _get_index(x_df)
        v = self.cf.predict_survival_function(x_df)[index]

        # set all values in predicted survival function at position lower than recency to 0
        for i,j in enumerate(v.columns):
            v[j][v.index < recency[j]] = 0

        targets = pd.DataFrame(recency + trapz(v.values.T, v.index), index=index)

        pred = self.reverseTransformTargets(targets)

        return pred.values.reshape(-1)

    def _predict_expectation_recency_B(self, indices=None, dataset='train'):
        df = self.data.train_df
        # df_unscaled = self.data.train_unscaled_df

        if dataset=='test':
            df = self.data.test_df

        if indices is None:
            indices = self.data.split_val_ind

        x_df = df.iloc[indices]

        index = _get_index(x_df)

        v = self._predict_survival_function(indices, dataset)

        targets = pd.DataFrame(trapz(v.values.T, v.index), index=index)

        pred = self.reverseTransformTargets(targets)

        return pred.values.reshape(-1)

    def _predict_expectation_recency_C(self, indices=None, dataset='train'):
        df = self.data.train_df
        df_unscaled = self.data.train_unscaled_df

        if dataset=='test':
            df = self.data.test_df
            df_unscaled = self.data.test_unscaled_df

        if indices is None:
            indices = self.data.split_val_ind

        x_df = df.iloc[indices]
        x_df_unscaled = df_unscaled.iloc[indices]
        recency = self.transformTargets(x_df_unscaled.recency)

        index = _get_index(x_df)
        survival = self.cf.predict_survival_function(x_df)[index]

        # set all values in predicted survival function at position lower than recency to 0
        # S_ts = np.zeros(len(index)) # survival at time ts
        s_df = pd.DataFrame(index=index, columns=['S_ts','int_full', 'int_from_ts', 'int_to_ts', 'E_T'])
        s_df['int_full'] = trapz(survival.values.T, survival.index)
        for i in survival.columns:
            s_filter = survival[i][survival.index < recency[i]]
            s_df.loc[i,'S_ts'] = 1
            if s_filter.any():
                s_df.loc[i,'S_ts'] = s_filter.values[-1] # set survival at time ts
                survival.loc[survival.index <= recency[i],i] = 0

        s_df['int_from_ts'] = trapz(survival.values.T, survival.index)
        s_df['int_to_ts'] = s_df['int_full'] - s_df['int_from_ts']

        s_df['E_T'] = s_df['int_from_ts'] / s_df['S_ts'] + s_df['int_to_ts']

        pred = self.reverseTransformTargets(s_df['E_T'].values.reshape(-1))

        return pred

    def _predict_survival_function(self, indices=None, dataset='train'):
        df = self.data.train_df
        df_unscaled = self.data.train_unscaled_df

        if dataset=='test':
            df = self.data.test_df
            df_unscaled = self.data.test_unscaled_df

        if indices is None:
            indices = self.data.split_val_ind

        x_df = df.iloc[indices]
        x_df_unscaled = df_unscaled.iloc[indices]
        recency = self.transformTargets(x_df_unscaled.recency)

        index = _get_index(x_df)
        cum_hazard = self.cf.predict_cumulative_hazard(x_df)

        # set all values in hazard function at position lower than recency to 0
        for i in cum_hazard.columns:
            # s = cum_hazard[i][cum_hazard.index < recency[i]].sum()
            cum_hazard[i] -= cum_hazard[i][cum_hazard.index <= recency[i]].values[-1]
            cum_hazard[i][cum_hazard.index <= recency[i]] = 0

        # survival function
        v = np.exp(-cum_hazard)

        return v

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

        pred_durations[pred_durations==np.inf] = hours_year
        pred_durations[pred_durations>hours_year] = hours_year
        pred_durations[pred_durations==np.nan] = hours_year

        churn_err = getChurnScores(~df.observed, pred_churn, pred_durations)

        return {'churn_acc': churn_err['accuracy'],
                'churn_auc': churn_err['auc'],
                'rmse_days': np.sqrt(mean_squared_error(df.deltaNextHours, pred_durations)) / 24,
                'concordance': concordance_index(df.deltaNextHours, pred_durations, df.observed)}


def runGridSearch(model, include_recency=False):
    """
    Cross-validated search for parameters

    """
    nFolds = 10
    nPools = 10
    bounds = (2000,3000)
    n_iter = 21
    space = np.linspace(bounds[0],bounds[1],n_iter)

    print(model.RESULT_PATH)

    # load churn data for splitting fold stratas
    churnData = ChurnData()

    cv = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=42)
    splits = np.array(list(cv.split(**churnData.train)))

    scores = []
    for p in space: _evaluatePenalizer(p, model=model, splits=splits, nPools=nPools, include_recency=include_recency, error=None)

    res = {k: [d[k] for d in scores] for k in scores[0]}

    with open(model.RESULT_PATH+'grid_search{}.pkl'.format('_rec' if include_recency else ''), 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return res


def runBayesOpt(model, include_recency=False, error='concordance', maximise=True):
    """
    Cross-validated search for parameters

    """
    nFolds = 10
    nPools = 10
    bounds = {'penalizer': (2000,3000)}
    n_iter = 20

    print(model.RESULT_PATH)

    # load churn data for splitting fold stratas
    churnData = ChurnData()

    cv = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=42)
    splits = np.array(list(cv.split(**churnData.train)))

    f = partial(_evaluatePenalizer, model=model, splits=splits, nPools=nPools, include_recency=include_recency, error=error, maximise=maximise)
    bOpt = BayesianOptimization(f, bounds)

    bOpt.maximize(init_points=2, n_iter=n_iter, acq='ucb', kappa=5, kernel=Matern())

    with open(model.RESULT_PATH+'bayes_opt_{}{}.pkl'.format(error, '_rec' if include_recency else ''), 'wb') as handle:
        pickle.dump(bOpt, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return bOpt

def _evaluatePenalizer(penalizer, model=None, splits=None, nPools=None, include_recency=False, error='concordance', maximise=False):
    pool = Pool(nPools)

    scores = pool.map(
            partial(
                _runParameterSearch,
                model=model,
                penalizer=penalizer,
                include_recency=include_recency,
                error=error,
                maximise=maximise),
            splits)

    pool.close()
    pool.join()

    # scores = list(map(
    #         partial(
    #             _runParameterSearch,
    #             model=model,
    #             penalizer=penalizer,
    #             include_recency=include_recency,
    #             error=error,
    #             maximise=maximise),
    #         splits))
    return np.mean(scores)

def _runParameterSearch(splits, model=None, penalizer=None, include_recency=False, error='concordance', maximise=False):
    train_ind, test_ind = splits
    model = model(penalizer=penalizer, include_recency=include_recency)
    model.fit(model.data.train_df, indices=train_ind)

    if error is None:
        return score

    score = model.getScores(test_ind)[error]

    if not maximise:
        score = -score

    return score

def crossValidate(model, penalizer=2045, include_recency=False, nFolds=10):
    churnData = ChurnData()
    cv = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=42)
    splits = np.array(list(cv.split(**churnData.train)))


    pool = Pool(nFolds)

    scores = pool.map(
            partial(_scoreModel, model=model, penalizer=penalizer, include_recency=include_recency),
            splits)

    res = {key: np.mean([score[key] for score in scores]) for key in scores[0].keys()}

    pool.close()
    pool.join()

    return res

def _scoreModel(split, model=None, penalizer=0, include_recency=False):
    train_ind, test_ind = split
    model = model(penalizer=penalizer, include_recency=include_recency)
    model.fit(model.data.train_df, indices=train_ind)

    return model.getScores(test_ind)


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
    df['actual'] = model.data.split_val['y'][observed] / 24

    jointgrid = sns.jointplot('actual', 'predicted', data=df, kind='kde', size=figsize(.5,.5)[0])

    plt.show()


# up to acc 0.712
def showChurnedPred(model, width=1, height=None):
    observed = model.data.split_val_df.observed.values.astype('bool').reshape(-1)
    pred_val = model.predict_expectation()

    returnDate = pred_val - model.data.split_val_unscaled_df.recency.values.reshape(-1)

    fig, ax = newfig(width, height)

    ax.hist(returnDate[observed], bins=50, label='not churned', alpha=.5)
    ax.hist(returnDate[~observed], bins=50, label='churned', alpha=.5)
    xDates = [pd.datetime(2016,i,1) for i in range(1,12)]
    xDatesHours = [(d - predPeriod['start']).to_timedelta64()/np.timedelta64(1,'h') for d in xDates]
    xDatesStr = [d.strftime('%Y-%m') for d in xDates]
    # ax.set_xticks(xDatesHours)
    # ax.set_xticklabels(xDatesStr)
    ax.axvline(x=predPeriodHours, label='prediction threshold' )
    ax.legend()

    fig.tight_layout()
    fig.show()


