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
from plot_format import *
from seaborn import apionly as sns


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


    def predict_expectation(self, df, df_unscaled):
        if self.include_recency:
            return self._predict_expectation_recency(df, df_unscaled)

        index = _get_index(df)

        pred = self.reverseTransformTargets(self.cf.predict_expectation(df))

        return pred.values.reshape(-1)


    def _predict_expectation_recency(self, df, df_unscaled):
        x_df = df
        x_df_unscaled = df_unscaled

        recency = self.transformTargets(x_df_unscaled.recency)

        index = _get_index(x_df)
        recency.index = index
        survival = self.cf.predict_survival_function(x_df)[index]

        # set all values in predicted survival function at position lower than recency to 0
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


    def predict_churn(self, pred_durations, df_unscaled):
        churned = (pred_durations - df_unscaled.recency.values.reshape(-1)) > predPeriodHours

        return churned.reshape(-1)


    def getScores(self, indices=None, dataset='train'):
        df = self.data.train_df
        df_unscaled = self.data.train_unscaled_df

        if dataset=='test':
            df = self.data.test_df
            df_unscaled = self.data.test_unscaled_df

        if indices is None:
            indices = self.data.split_val_ind

        df = df.iloc[indices]
        df_unscaled = df_unscaled.iloc[indices]

        pred_durations = self.predict_expectation(df, df_unscaled)
        pred_churn = self.predict_churn(pred_durations, df_unscaled)

        pred_durations[pred_durations==np.inf] = hours_year
        pred_durations[pred_durations>hours_year] = hours_year
        pred_durations[pred_durations==np.nan] = hours_year

        churn_err = getChurnScores(~df.observed, pred_churn, pred_durations)

        return {'churn_acc': churn_err['accuracy'],
                'churn_auc': churn_err['auc'],
                'churn_prec': churn_err['precision'][1],
                'churn_recall': churn_err['recall'][1],
                'churn_f1': churn_err['f1'][1],
                'rmse_days': np.sqrt(mean_squared_error(df.deltaNextHours[df.observed]/24, pred_durations[df.observed]/24)),
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
    for p in space:
        print(p)
        scores.append(_evaluatePenalizer(p, model=model, splits=splits, nPools=nPools, include_recency=include_recency, error=None))

    res = {'penalties': space, 'scores': {k: [d[k] for d in scores] for k in scores[0]}}

    with open(model.RESULT_PATH+'grid_search{}_{}.pkl'.format('_rec' if include_recency else '', n_iter), 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return res


def runBayesOpt(model, include_recency=False, error='concordance', maximise=True):
    """
    Cross-validated search for parameters

    """
    nFolds = 10
    nPools = 10
    bounds = {'penalizer': (1000,5000)}
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

    if error is None:
        return {k: np.mean([d[k] for d in scores]) for k in scores[0]}

    return np.mean(scores)

def _runParameterSearch(splits, model=None, penalizer=None, include_recency=False, error='concordance', maximise=False):
    train_ind, test_ind = splits
    model = model(penalizer=penalizer, include_recency=include_recency)
    model.fit(model.data.train_df, indices=train_ind)

    score = model.getScores(test_ind)

    if error is None:
        return score

    score = score[error]

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


def showResidPlot(model, pred_val, width=1, height=None):
    observed = model.data.split_val_df.observed.values.astype('bool').reshape(-1)

    df = pd.DataFrame()
    df['predicted (days)'] = pred_val[observed] / 24
    # df['actual (days)'] = model.data.split_val['y'][observed] / 24
    df['actual (days)'] = model.data.split_val_unscaled_df[observed].deltaNextHoursFull.values / 24
    df['residual (days)'] = df['predicted (days)'] - df['actual (days)']

    grid = sns.JointGrid('actual (days)', 'residual (days)', data=df, size=figsize(.5,.5)[0], xlim=(0,250), ylim=(-175,175))
    grid = grid.plot_marginals(sns.distplot, kde=False)#, shade=True)
    # grid = grid.plot_joint(sns.kdeplot, shade=True, n_levels=15, cmap='Blues', shade_lowest=False, cut=0)
    grid = grid.plot_joint(plt.scatter, alpha=.1, s=6, lw=0)

    plt.show()

def showResidPlot_short(model, pred_val, width=1, height=None):
    # observed = model.data.split_val_df.observed.values.astype('bool').reshape(-1)
    # observed = ~model.data.split_val_df.churnedFull.values.astype('bool').reshape(-1)

    df = pd.DataFrame()
    df['predicted (days)'] = pred_val / 24
    df['actual (days)'] = model.data.split_val_unscaled_df.deltaNextHoursFull.values / 24
    df['residual (days)'] = df['predicted (days)'] - df['actual (days)']

    grid = sns.JointGrid('actual (days)', 'residual (days)', data=df[~model.data.split_val_df.churnedFull.astype('bool')], size=figsize(.5,.5)[0], xlim=(0,250), ylim=(-175,175))
    grid = grid.plot_marginals(sns.distplot, kde=False, color='k')#, shade=True)
    # grid = grid.plot_joint(sns.kdeplot, shade=True, n_levels=15, cmap='Blues', shade_lowest=False, cut=0)
    grid = grid.plot_joint(plt.scatter, alpha=.1, s=6, lw=0)
    grid.ax_joint.clear()

    obsUncensScat = grid.ax_joint.scatter(df.loc[model.data.split_val_df.observed, 'actual (days)'], df.loc[model.data.split_val_df.observed, 'residual (days)'], alpha=.1, s=6, lw=0, color='C0', label='Ret. user (uncens.)')

    obs_cens = ~model.data.split_val_df.observed & ~model.data.split_val_df.churnedFull.astype('bool')
    obsCensScat = grid.ax_joint.scatter(df.loc[obs_cens, 'actual (days)'], df.loc[obs_cens, 'residual (days)'], alpha=.1, s=6, lw=0, color='C4', label='Ret. user (cens.)')

    grid.ax_joint.legend(handles=[obsUncensScat, obsCensScat], loc=1, labelspacing=0.02, handlelength=0.5)
    grid.ax_joint.set_xlabel('actual return time (days)')
    grid.ax_joint.set_ylabel('residual (days)')

    plt.show()

def showResidPlot_short_date(model, pred_val, width=1, height=None):
    # observed = model.data.split_val_df.observed.values.astype('bool').reshape(-1)
    observed = ~model.data.split_val_df.churnedFull.values.astype('bool').reshape(-1)

    df = pd.DataFrame()
    df['predicted (days)'] = pred_val / 24
    # df['actual (days)'] = model.data.split_val['y'] / 24
    df['actual (days)'] = model.data.split_val_unscaled_df.deltaNextHoursFull.values / 24
    df['hoursInPred'] = (model.data.split_val_unscaled_df.deltaNextHoursFull.values - model.data.split_val_unscaled_df.recency.values)
    df['date'] = df['hoursInPred'] * np.timedelta64(1,'h') + predPeriod['start']
    df['residual (days)'] = df['predicted (days)'] - df['actual (days)']

    grid = sns.JointGrid('hoursInPred', 'residual (days)', data=df[observed], size=figsize(.5,.5)[0], xlim=(0,3000), ylim=(-110,110))
    grid = grid.plot_marginals(sns.distplot, kde=False, color='k')#, shade=True)
    # grid = grid.plot_joint(sns.kdeplot, shade=True, n_levels=12, cmap='Blues', shade_lowest=False, cut=0)
    grid = grid.plot_joint(plt.scatter, alpha=.1, s=6, lw=0)
    # grid = grid.plot_joint(sns.kdeplot, shade=True, n_levels=15, cmap='Blues', shade_lowest=False, cut=0)
    # grid = grid.plot(df.hoursInPred, df['residual (days)'])
    # grid.ax_joint.clear()
    # grid.ax_joint.scatter(df.hoursInPred, df['residual (days)'])
    grid.ax_joint.clear()

    retUnc = grid.ax_joint.scatter(df.loc[model.data.split_val_df.observed, 'hoursInPred'], df.loc[model.data.split_val_df.observed, 'residual (days)'], alpha=.1, s=6, lw=0, color='C0', label='Ret. user (uncens.)')

    obs_cens = ~model.data.split_val_df.observed & ~model.data.split_val_df.churnedFull.astype('bool')
    retCens = grid.ax_joint.scatter(df.loc[obs_cens, 'hoursInPred'], df.loc[obs_cens, 'residual (days)'], alpha=.1, s=6, lw=0, color='C4', label='Ret. user (cens.)')

    xDates = [pd.datetime(2016,i,1) for i in [2,4,6]]
    xDatesHours = [(d - predPeriod['start']).to_timedelta64()/np.timedelta64(1,'h') for d in xDates]
    xDatesStr = [d.strftime('%Y-%m') for d in xDates]
    grid.ax_joint.axvline(x=xDatesHours[1], ls=':', color='k', label='prediction threshold', lw=1)
    grid.ax_joint.set_xticks(xDatesHours)
    grid.ax_joint.set_xticklabels(xDatesStr)
    grid.ax_joint.set_xlabel('actual return date')
    grid.ax_joint.set_ylabel('residual (days)')
    grid.ax_joint.legend(handles=[retUnc, retCens], loc=1, labelspacing=0.02, handlelength=0.5)
    # grid = grid.plot_joint(plt.scatter)#, shade=True, n_levels=10, cmap='Blues', shade_lowest=False, cut=0)

    plt.show()


def showMseOverTime(model, pred_val, width=1, height=None):
    observed = ~model.data.split_val_df.churnedFull.values.astype('bool').reshape(-1)
    pred_val = pred_val.astype('float')

    df = pd.DataFrame(dtype=float)
    df['predicted_days'] = pred_val[observed] / 24
    df['actual_days'] = model.data.split_val_unscaled_df.deltaNextHoursFull[observed] / 24
    df['hoursInPred'] = (model.data.split_val_unscaled_df[observed].deltaNextHoursFull.values - model.data.split_val_unscaled_df[observed].recency.values)
    df['date'] = pd.DatetimeIndex(df['hoursInPred'] * np.timedelta64(1,'h') + predPeriod['start']).dayofyear
    df['dateInd'] = (df.date - df.date.min()) // 7
    df['error'] = df['predicted_days'] - df['actual_days']
    df['squared_err'] = df['error'] ** 2

    # return np.sqrt(mean_squared_error(model.data.split_val['y'][observed] / 24, pred_val[observed]/24))
    # return np.sqrt(mean_squared_error(model.data.split_val['y'][retCens] / 24, pred_val[retCens]/24))
    # return np.sqrt(df.squared_err.mean())

    rmse_by_date = np.sqrt(df.groupby('dateInd').squared_err.mean())
    variance_by_date = df.groupby('dateInd').error.var()

    fig, ax = newfig(width, height)
    ax2 = ax.twinx()

    xDates = [pd.datetime(2016,i,1) for i in [2,4,6]]
    xDatesInd = (pd.DatetimeIndex(xDates).dayofyear - df.date.min()) / 7
    xDatesStr = [d.strftime('%Y-%m') for d in xDates]

    ax.set_xticks(xDatesInd)
    ax.set_xticklabels(xDatesStr)
    ax.set_ylabel('RMSE')#, color='C0')
    ax.tick_params('y')#, colors='C0')
    ax.set_xlabel('Actual return date')

    rmse_plot = ax.plot(rmse_by_date.index[:-1], rmse_by_date.values[:-1], zorder=0, label='RMSE')[0]
    var_plot = ax2.plot(variance_by_date.index[:-1], variance_by_date.values[:-1], color='C1', zorder=-1, label='Error variance')[0]
    ax2.set_ylabel('Error variance')#, color='C1')
    ax2.tick_params('y')#, colors='C1')
    ax.axvline(x=xDatesInd[1], color='k', zorder=1, ls=':', lw=1)
    ax.legend(handles=[rmse_plot, var_plot])
    ax.set_xlim((xDatesInd[0], xDatesInd[-1]))
    # ax.set_ylim((60,120))
    fig.tight_layout()
    fig.show()


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


