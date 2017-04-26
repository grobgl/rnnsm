from churn_data import ChurnData
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, RFECV
import pandas as pd
import numpy as np
import itertools
from multiprocessing import Pool
from functools import partial
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

import pickle
import os
import sys
sys.path.insert(0, '../utils')
from plot_format import *

_RESULT_PATH = '../../results/churn/logistic_regression/'

def getOptL1Model():
    res = pickle.load(open('{}logRegL1vsL2.pkl'.format(_RESULT_PATH), 'rb'))
    accL1 = [r['accuracy'] for r in res['L1']]
    modelL1 = LogisticRegression(penalty='l1', C=np.logspace(np.log10(2e-4),0,800)[accL1.index(max(accL1))])
    return modelL1


def getOptL2Model():
    res = pickle.load(open('{}logRegL1vsL2.pkl'.format(_RESULT_PATH), 'rb'))
    accL2 = [r['accuracy'] for r in res['L2']]
    modelL2 = LogisticRegression(penalty='l2', C=np.logspace(np.log10(2e-4),0,800)[accL2.index(max(accL2))])
    return modelL2


def runFeatureElimination(includeFeat='all'):
    """
    Performs feature elimination
    Run RFE for each fold, find average scores for AUC and accuracy for each step

    :includeFeat: 'avg' or 'wght' -- include wght avg or avg only
    """
    # load data
    pool = Pool(64)

    # all features
    data = ChurnData()
    features = data.features

    if includeFeat=='avg':
        # only avg deltaPrev
        features = list(set(features) - set(['logDeltaPrev_wght_avg', 'deltaPrev_wght_avg']))
    elif includeFeat=='wght':
        # only weighted deltaPrev
        features = list(set(features) - set(['logDeltaPrev_avg', 'deltaPrev_avg']))

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = [0] * 10

    for i, (train_ind, test_ind) in enumerate(cv.split(**data.train)):
        print('Fold: {} out of 10'.format(i+1))
        scores[i] = pool.map(
                partial(_runFeatureElimination, features=features, train=train_ind, test=test_ind),
                range(1, len(features)+1))

    pool.close()

    features = [[s['features'] for s in ss] for ss in scores]
    accuracy = np.array([[s['accuracy'] for s in ss] for ss in scores]).mean(0)
    roc_auc = np.array([[s['roc_auc'] for s in ss] for ss in scores]).mean(0)
    # accuracy = np.array([s['accuracy'] for s in scores]).mean(0)
    # roc_auc = np.array([s['roc_auc '] for s in scores]).mean(0)

    res = {'features': features, 'accuracy': accuracy, 'roc_auc': roc_auc}

    with open('{}logReg_rfe_{}.pkl'.format(_RESULT_PATH, includeFeat), 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return res


def _runFeatureElimination(numFeatures, features, train=None, test=None):
    model = LogisticRegression()
    data = ChurnData(features)

    rfe = RFE(model, numFeatures)
    fit = rfe.fit(data.train['X'][train], data.train['y'][train])

    features = data.features[fit.support_]
    data = ChurnData(features)
    model.fit(data.train['X'][train], data.train['y'][train])
    scores = data.getScores(model, X=data.train['X'][test], y=data.train['y'][test])

    return {'features': features, 'accuracy': scores['accuracy'], 'roc_auc': scores['auc']}


def runL1GridSearch():
    """
    Runs grid search logistic regression model with L1 penalty
    Uses 10-fold cross validation
    """

    param_grid = {
        'penalty': ['l1'],
        'C': np.logspace(np.log10(2e-4),0,800)
    }

    data = ChurnData()
    model = LogisticRegression(penalty='l1')

    # fixed random state for cross validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # default scoring is accuracy
    grid_acc = GridSearchCV(estimator=model, param_grid=param_grid, verbose=1, n_jobs=64, cv=cv, scoring='accuracy')
    grid_acc.fit(**data.train)

    grid_auc = GridSearchCV(estimator=model, param_grid=param_grid, verbose=1, n_jobs=64, cv=cv, scoring='roc_auc')
    grid_auc.fit(**data.train)

    res = {'accuracy': grid_acc, 'roc_auc': grid_auc}

    with open('{}logRegL1_grid.pkl'.format(_RESULT_PATH), 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return res


def runL2GridSearch():
    """
    Runs grid search logistic regression model with L2 penalty
    Uses 10-fold cross validation
    """

    param_grid = {
        'penalty': ['l2'],
        'C': np.logspace(-6,0,800)
    }

    data = ChurnData()
    model = LogisticRegression(penalty='l2')

    # fixed random state for cross validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # default scoring is accuracy
    grid_acc = GridSearchCV(estimator=model, param_grid=param_grid, verbose=1, n_jobs=64, cv=cv, scoring='accuracy')
    grid_acc.fit(**data.train)

    grid_auc = GridSearchCV(estimator=model, param_grid=param_grid, verbose=1, n_jobs=64, cv=cv, scoring='roc_auc')
    grid_auc.fit(**data.train)

    res = {'accuracy': grid_acc, 'roc_auc': grid_auc}

    with open('{}logRegL2_grid.pkl'.format(_RESULT_PATH), 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return res


def pearsonAllIter():
    pearsonFeat = ['logNumSessions', 'recency', 'logSessionLen_avg', 'dayOfMonth_wght_avg',
                   'logDeltaPrev_avg', 'logNumInteractions_avg', 'logNumItemsViewed_avg',
                   'deviceIos_wght', 'logNumDivisions_wght_avg', 'dayOfWeek_wght_avg',
                   'price_avg', 'deltaLogDeltaPrev', 'hourOfDay_wght_avg']

    pool = Pool(8)
    results = pool.map(pearsonComb, range(1,len(pearsonFeat)+1))
    pool.close()

    return results

def pearsonComb(n):
    pearsonFeat = ['logNumSessions', 'recency', 'logSessionLen_avg', 'dayOfMonth_wght_avg',
                   'logDeltaPrev_avg', 'logNumInteractions_avg', 'logNumItemsViewed_avg',
                   'deviceIos_wght', 'logNumDivisions_wght_avg', 'dayOfWeek_wght_avg',
                   'price_avg', 'deltaLogDeltaPrev', 'hourOfDay_wght_avg']
    featComb = itertools.permutations(pearsonFeat, n)
    return [fitAndScore(feat) for feat in featComb]


def fitAndScore(features):
    data = ChurnData(features)
    model = LogisticRegression()
    model.fit(**data.split_train)
    scores = data.getScores(model, 'split_val')

    return {'model': model, 'scores': scores, 'features': features}


def findPearsonCor():
    data = ChurnData()
    df = pd.DataFrame(data.X_split_train)
    df.columns = data.features
    df['churned'] = data.y_split_train
    corr = df.corr().churned
    keys = corr.keys()

    feat_noWght = ['numSessions', 'recency']
    feat = ['deltaPrev', 'dayOfMonth', 'dayOfWeek', 'hourOfDay', 'sessionLen',
            'price', 'numDivisions', 'numInteractions', 'numItemsViewed']
    feat_dev = ['Desktop', 'Mobile', 'Ios', 'Android', 'Unknown']

    corrs_feat_noWght = pd.DataFrame(columns=['feature','plain','log'])
    corrs_feat_noWght.feature = feat_noWght
    corrs_feat_noWght.plain = [corr[f] for f in feat_noWght]
    corrs_feat_noWght.log = [corr['log'+upperfirst(f)] for f in feat_noWght]

    corrs_feat = pd.DataFrame(columns=['feature','avg','log_avg','wght_avg','log_wght_avg'])
    corrs_feat.feature = feat
    corrs_feat.avg= [corr[f+'_avg'] for f in feat]
    corrs_feat.log_avg = [corr['log'+upperfirst(f)+'_avg'] if 'log'+upperfirst(f)+'_avg' in keys else np.nan for f in feat]
    corrs_feat.wght_avg= [corr[f+'_wght_avg'] for f in feat]
    corrs_feat.log_wght_avg= [corr['log'+upperfirst(f)+'_wght_avg'] if 'log'+upperfirst(f)+'_wght_avg' in keys else np.nan for f in feat]

    corrs_dev = pd.DataFrame(columns=['feature','plain','wght'])
    corrs_dev.feature = feat_dev
    corrs_dev.plain = [corr['device'+f] for f in feat_dev]
    corrs_dev.wght = [corr['device'+upperfirst(f)+'_wght'] for f in feat_dev]

    return corrs_feat_noWght, corrs_feat, corrs_dev


def upperfirst(x):
        return x[0].upper() + x[1:]


def plotRfeRes(width=1, height=None):
    # 'logReg_rfe_all'
    res_all = pickle.load(open('{}logReg_rfe_all.pkl'.format(_RESULT_PATH), 'rb'))
    # 'logReg_rfe_avg'
    res_noWghtRet = pickle.load(open('{}logReg_rfe_avg.pkl'.format(_RESULT_PATH), 'rb'))
    # 'logReg_rfe_wght'
    res_onlyWghtRet = pickle.load(open('{}logReg_rfe_wght.pkl'.format(_RESULT_PATH), 'rb'))

    numFeat = len(res_all['accuracy'])
    auc_all = res_all['roc_auc']
    auc_noWghtRet = res_noWghtRet['roc_auc']
    auc_onlyWghtRet = res_onlyWghtRet['roc_auc']
    acc_all = res_all['accuracy']
    acc_noWghtRet = res_noWghtRet['accuracy']
    acc_onlyWghtRet = res_onlyWghtRet['accuracy']

    fig, ax1 = newfig(width, height, ax_pos=121)
    ax2 = fig.add_subplot(122)

    ax1.plot(range(1, numFeat+1), auc_all, label='AUC')
    ax1.plot(range(1, numFeat+1), acc_all, label='Classification accuracy')
    ax2.plot(range(1, numFeat+1), acc_all, label='Classification accuracy\n(i) all features')
    ax2.plot(range(1, len(acc_noWghtRet) + 1), acc_noWghtRet, label='Classification accuracy\n(ii) avg. return time')
    ax2.plot(range(1, len(acc_onlyWghtRet) + 1), acc_onlyWghtRet, label='Classification accuracy\n(iii) wt. avg. return time')

    ax1.axvline(x=4, linewidth=1, linestyle='--', color='grey', zorder=-1)
    ax1.set_xticks([0,4,10,20,30,40])
    ax1.set_xlabel(r'Number of features used')
    ax2.set_xlabel(r'Number of features used')
    ax1.legend()
    ax2.legend()

    fig.tight_layout()
    fig.show()


def plotL1L2GridRes(width=1, height=None):
    """
    Plots grid search results comparing L1 and L2 penalty terms
    """
    gridL1 = pickle.load(open(_RESULT_PATH+'logRegL1_grid.pkl','rb'))
    gridL2 = pickle.load(open(_RESULT_PATH+'logRegL2_grid.pkl','rb'))

    cValuesL1 = np.logspace(np.log10(2e-4),0,800)
    cValuesL2 = np.logspace(-6,0,800)

    accL1 = gridL1['accuracy'].cv_results_['mean_test_score']
    accL2 = gridL2['accuracy'].cv_results_['mean_test_score']
    aucL1 = gridL1['roc_auc'].cv_results_['mean_test_score']
    aucL2 = gridL2['roc_auc'].cv_results_['mean_test_score']

    fig, ax = newfig(width, height)

    ax.plot(cValuesL1, accL1, label='L1 accuracy')
    ax.plot(cValuesL2, accL2, label='L2 accuracy')
    ax.plot(cValuesL1, aucL1, label='L1 AUC')
    ax.plot(cValuesL2, aucL2, label='L2 AUC')

    ax.set_xlabel('Regularisation parameter')
    ax.legend(loc='center right', bbox_to_anchor=(1,.68))
    ax.set_xscale('log', nonposy='clip')
    fig.tight_layout()

    fig.show()

