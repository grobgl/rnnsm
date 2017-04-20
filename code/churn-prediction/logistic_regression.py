from churn_data import ChurnData
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np
import itertools
from multiprocessing import Pool
from functools import partial

import pickle
import os
import sys
sys.path.insert(0, '../utils')
from plot_format import *


def run():
    # load data
    pool = Pool(8)
    data = ChurnData(pearsonFeat)
    feat = data.features

    # data = ChurnData(list(set(feat) - set(['dayOfMonth_wght_avg','dayOfMonth_avg','deltaLogDeltaPrev', 'deltaDeltaPrev', 'logDeltaPrev_wght_avg', 'deltaPrev_wght_avg'])))
    # data = ChurnData(list(set(feat) - set(['deltaLogDeltaPrev', 'deltaDeltaPrev', 'logDeltaPrev_wght_avg', 'deltaPrev_wght_avg', 'deltaPrev_avg'])))#, 'logDeltaPrev_avg'])))
    # data = ChurnData(list(set(feat) - set(['deltaLogDeltaPrev', 'deltaDeltaPrev', 'logDeltaPrev_wght_avg', 'deltaPrev_wght_avg'])))
    # data = ChurnData(list(set(feat) - set(['logDeltaPrev_wght_avg', 'deltaPrev_wght_avg'])))
    results = pool.map(partial(runFeatElim, data=data), range(1, len(data.features)+1))
    pool.close()

    return results

def regComp():
    pool = Pool(8)
    data = ChurnData()
    cValuesL1 = np.logspace(np.log10(2e-4),-1,400)
    cValuesL2 = np.logspace(-6,-1,400)
    resultsL1 = pool.map(partial(scoreModel, data), [{'C': i, 'penalty': 'l1'} for i in cValuesL1])
    resultsL2 = pool.map(partial(scoreModel, data), [{'C': i, 'penalty': 'l2'} for i in cValuesL2])
    pool.close()

    res = {'L1': resultsL1, 'L2': resultsL2}

    with open('../../results/logRegL1vsL2.pkl', 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return res

def plotComp(width=1, height=None):
    with open('../../results/logRegL1vsL2.pkl', 'rb') as handle:
        res = pickle.load(handle)

    cValuesL1 = np.logspace(np.log10(2e-4),-1,400)
    cValuesL2 = np.logspace(-6,-1,400)
    aucL1 = [r['auc'] for r in res['L1']]
    aucL2 = [r['auc'] for r in res['L2']]
    f1L1 = [r['f1'] for r in res['L1']]
    f1L2 = [r['f1'] for r in res['L2']]
    accL1 = [r['accuracy'] for r in res['L1']]
    accL2 = [r['accuracy'] for r in res['L2']]

    fig, ax = newfig(width, height)

    ax.plot(cValuesL1, aucL1, label='L1 AUC')
    ax.plot(cValuesL2, aucL2, label='L2 AUC')
    ax.plot(cValuesL1, f1L1, label='L1 F1')
    ax.plot(cValuesL2, f1L2, label='L2 F1')
    ax.plot(cValuesL1, accL1, label='L1 accuracy')
    ax.plot(cValuesL2, accL2, label='L2 accuracy')

    ax.legend()
    ax.set_xscale('log', nonposy='clip')
    fig.tight_layout()

    fig.show()

def scoreModel(data, kwargs):
    model = LogisticRegression(**kwargs)
    model.fit(**data.split_train)
    return data.getScores(model, 'split_val')


def pearsonAllIter():
    pearsonFeat = ['logNumSessions', 'recency', 'logSessionLen_avg', 'dayOfMonth_wght_avg', 'logDeltaPrev_avg', 'logNumInteractions_avg', 'logNumItemsViewed_avg', 'deviceIos_wght', 'logNumDivisions_wght_avg', 'dayOfWeek_wght_avg', 'price_avg', 'deltaLogDeltaPrev', 'hourOfDay_wght_avg']
    # pearsonFeat = ['logNumSessions', 'recency', 'logSessionLen_avg', 'dayOfMonth_wght_avg', 'logDeltaPrev_avg', 'logNumInteractions_avg', 'logNumItemsViewed_avg']

    pool = Pool(8)
    results = pool.map(pearsonComb, range(1,len(pearsonFeat)+1))
    pool.close()

    return results

def pearsonComb(n):
    pearsonFeat = ['logNumSessions', 'recency', 'logSessionLen_avg', 'dayOfMonth_wght_avg', 'logDeltaPrev_avg', 'logNumInteractions_avg', 'logNumItemsViewed_avg', 'deviceIos_wght', 'logNumDivisions_wght_avg', 'dayOfWeek_wght_avg', 'price_avg', 'deltaLogDeltaPrev', 'hourOfDay_wght_avg']
    featComb = itertools.permutations(pearsonFeat, n)
    return [fitAndScore(feat) for feat in featComb]


def fitAndScore(features):
    data = ChurnData(features)
    model = LogisticRegression()
    model.fit(**data.split_train)
    scores = data.getScores(model, 'split_val')

    return {'model': model, 'scores': scores, 'features': features}



def runFeatElim(numFeatures, data):
    model = LogisticRegression()
    rfe = RFE(model, numFeatures)
    fit = rfe.fit(**data.split_train)
    features = data.features[fit.support_]
    prunedData = ChurnData(features)
    model.fit(**prunedData.split_train)
    scores = prunedData.getScores(model, 'split_val')

    return {'model': model, 'fit': fit, 'scores': scores, 'features': features}


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


def plotResults(width=1, height=None):
    res = pickle.load(open('../../results/logisticRegression_rfe.pkl', 'rb'))
    res_deltaDelta = pickle.load(open('../../results/logisticRegression_deltaDelta_rfe.pkl', 'rb'))
    res_noWghtRet = pickle.load(open('../../results/logisticRegression_noWghtRet_rfe.pkl', 'rb'))

    numFeat = len(res)
    f1 = [r['scores']['f1'] for r in res]
    auc = [r['scores']['auc'] for r in res]
    acc = [r['scores']['accuracy'] for r in res]
    f1_deltaDelta = [r['scores']['f1'] for r in res_deltaDelta]
    auc_deltaDelta = [r['scores']['auc'] for r in res_deltaDelta]
    acc_deltaDelta = [r['scores']['accuracy'] for r in res_deltaDelta]
    auc_noWghtRet = [r['scores']['auc'] for r in res_noWghtRet]
    f1_noWghtRet = [r['scores']['f1'] for r in res_noWghtRet]
    acc_noWghtRet = [r['scores']['accuracy'] for r in res_noWghtRet]

    fig, ax1 = newfig(width, height, ax_pos=121)
    ax2 = fig.add_subplot(122)

    ax1.plot(range(1, numFeat+1), auc, label='AUC')
    ax1.plot(range(1, numFeat+1), acc, label='Classification accuracy')
    ax1.plot(range(1, numFeat+1), f1, label='F1 score')
    ax2.plot(range(1, len(f1_deltaDelta)+1), f1_deltaDelta, label='F1')
    ax2.plot(range(1, len(f1_noWghtRet) + 1), f1_noWghtRet, label='F1 (excl. wt. return time)')
    ax2.plot(range(1, len(auc_deltaDelta)+1), auc_deltaDelta, label='AUC')
    ax2.plot(range(1, len(auc_noWghtRet) + 1), auc_noWghtRet, label='AUC (excl. wt. return time)')
    ax2.plot(range(1, len(acc_deltaDelta)+1), acc_deltaDelta, label='Classification accuracy')
    ax2.plot(range(1, len(acc_noWghtRet) + 1), acc_noWghtRet, label='CA (excl. wt. return time)')

    ax1.axvline(x=4, linewidth=1, linestyle='--', color='grey', zorder=-1)
    # ax2.axvline(x=4, linewidth=1, linestyle='--', color='grey', zorder=-1)
    ax1.set_xticks([0,4,10,20,30,40])
    ax1.set_xlabel(r'Number of features used')
    ax2.set_xticks([0,4,10,20,30,40])
    ax2.set_xlabel(r'Number of features used')
    ax1.legend()
    ax2.legend()

    # ax.margins(0,.1)
    fig.tight_layout()
    fig.show()
