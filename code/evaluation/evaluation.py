import pickle
import pandas as pd
import numpy as np
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, roc_auc_score
from bayes_opt import BayesianOptimization
from lifelines.utils import concordance_index
from scipy.stats import binom

import sys
sys.path.insert(0, '../utils')
from plot_format import *
import seaborn as sns
from seaborn import apionly as sns

RESULT_PATH = '../../results/evaluation/'
MODELS = ['cox_ph_abs', 'cox_ph_noabs', 'rmtpp_abs', 'rmtpp_noabs', 'rnn']


def plot_mean_err_by_time(dataset, interval=7, width=1, height=None):
    churned = np.load('../../results/evaluation/{}/churned.npy'.format(dataset))
    deltaNextDays = np.load('../../results/evaluation/{}/deltaNextDays.npy'.format(dataset))[~churned]
    deltaNextDays_rounded = np.floor(deltaNextDays/interval)
    predDays = np.load('../../results/evaluation/{}/predictions_days.npy'.format(dataset))[~churned]
    days = np.arange(0, 34)
    rmses = np.array(list(map(lambda i: _err_by_true_days(deltaNextDays, deltaNextDays_rounded, predDays, i), days)))

    fig, ax = newfig(width, height)
    ax.bar(days, rmses)

    ax.axhline(0, ls='--', color='k', linewidth=1)
    ax.set_xlabel('Actual return time (weeks)')
    ax.set_ylabel('Mean error (days)')
    ax.set_ylim((-100, 100))
    # ax.set_ylim((-150,50))
    fig.tight_layout()
    fig.show()

def _err_by_true_days(deltaNextDays, deltaNextDays_rounded, predDays, true_days):
    filt = deltaNextDays_rounded == true_days
    y_pred = predDays[filt]
    y_true = deltaNextDays[filt]
    err = (((y_pred - y_true)).mean())

    return err


def plot_rmse_by_time(dataset, interval=7, width=1, height=None):
    churned = np.load('../../results/evaluation/{}/churned.npy'.format(dataset))
    deltaNextDays = np.load('../../results/evaluation/{}/deltaNextDays.npy'.format(dataset))[~churned]
    deltaNextDays_rounded = np.floor(deltaNextDays/interval)
    predDays = np.load('../../results/evaluation/{}/predictions_days.npy'.format(dataset))[~churned]
    days = np.arange(0, 34)
    rmses = np.array(list(map(lambda i: _rmse_by_true_days(deltaNextDays, deltaNextDays_rounded, predDays, i), days)))

    fig, ax = newfig(width, height)
    ax.bar(days, rmses)

    ax.axhline(0, ls='--', color='k', linewidth=1)
    ax.set_xlabel('Actual return time (weeks)')
    ax.set_ylabel('RMSE (days)')
    # ax.set_ylim((0,100))
    ax.set_ylim((0,150))
    fig.tight_layout()
    fig.show()

def _rmse_by_true_days(deltaNextDays, deltaNextDays_rounded, predDays, true_days):
    filt = deltaNextDays_rounded == true_days
    y_pred = predDays[filt]
    y_true = deltaNextDays[filt]
    rmse = np.sqrt(((y_pred - y_true)**2).mean())

    return rmse



def plot_rmse_by_numofsess(dataset, interval=7, width=1, height=None):
    churned = np.load('../../results/evaluation/{}/churned.npy'.format(dataset))
    deltaNextDays = np.load('../../results/evaluation/{}/deltaNextDays.npy'.format(dataset))[~churned]
    num_sess = np.load('../../results/evaluation/num_sessions.npy')[~churned]
    # deltaNextDays_rounded = np.floor(deltaNextDays/interval)
    predDays = np.load('../../results/evaluation/{}/predictions_days.npy'.format(dataset))[~churned]
    days = np.arange(1, 64)
    rmses = np.array(list(map(lambda i: _rmse_by_true_days_mask(deltaNextDays, num_sess==i, predDays), days)))
    rmses_final = _rmse_by_true_days_mask(deltaNextDays, num_sess>=64, predDays)

    fig, ax = newfig(width, height)
    ax.bar(days, rmses)
    ax.bar(64, rmses_final, color='c0')

    # ax.axhline(0, ls='--', color='k', linewidth=1)
    ax.set_xlabel('Number of active days')
    ax.set_ylabel('RMSE (days)')
    ax.set_ylim((0,120))
    ax.set_xticks([0,20,40,64])
    ax.set_xticklabels(['0','20','40','$\geq64$'])
    fig.tight_layout()
    fig.show()

def _rmse_by_true_days_mask(deltaNextDays, mask, predDays):
    y_pred = predDays[mask]
    y_true = deltaNextDays[mask]
    rmse = np.sqrt(((y_pred - y_true)**2).mean())

    return rmse


def plot_mean_err_by_numofsess(dataset, interval=7, width=1, height=None):
    churned = np.load('../../results/evaluation/{}/churned.npy'.format(dataset))
    deltaNextDays = np.load('../../results/evaluation/{}/deltaNextDays.npy'.format(dataset))[~churned]
    num_sess = np.load('../../results/evaluation/num_sessions.npy')[~churned]
    # deltaNextDays_rounded = np.floor(deltaNextDays/interval)
    predDays = np.load('../../results/evaluation/{}/predictions_days.npy'.format(dataset))[~churned]
    days = np.arange(1, 64)
    errs = np.array(list(map(lambda i: _rmse_by_true_days_mask(deltaNextDays, num_sess==i, predDays), days)))
    errs_final = _err_by_true_days_mask(deltaNextDays, num_sess>=64, predDays)

    fig, ax = newfig(width, height)
    ax.bar(days, errs)
    ax.bar(64, errs_final, color='c0')

    ax.axhline(0, ls='--', color='k', linewidth=1)
    ax.set_xlabel('Number of active days')
    ax.set_ylabel('Mean error (days)')
    # ax.set_ylim((0,100))
    ax.set_ylim((-100,100))
    fig.tight_layout()
    fig.show()

def _err_by_true_days_mask(deltaNextDays, mask, predDays):
    y_pred = predDays[mask]
    y_true = deltaNextDays[mask]
    err = (((y_pred - y_true)).mean())

    return err


def plot_errs_by_numofsess(dataset, interval=7, width=1, height=None):
    churned = np.load('../../results/evaluation/{}/churned.npy'.format(dataset))
    deltaNextDays = np.load('../../results/evaluation/{}/deltaNextDays.npy'.format(dataset))[~churned]
    num_sess = np.load('../../results/evaluation/num_sessions.npy')[~churned]
    # deltaNextDays_rounded = np.floor(deltaNextDays/interval)
    predDays = np.load('../../results/evaluation/{}/predictions_days.npy'.format(dataset))[~churned]
    days = np.arange(1, 64)
    errs = np.array(list(map(lambda i: _rmse_by_true_days_mask(deltaNextDays, num_sess==i, predDays), days)))
    errs_final = _err_by_true_days_mask(deltaNextDays, num_sess>=64, predDays)
    rmses = np.array(list(map(lambda i: _rmse_by_true_days_mask(deltaNextDays, num_sess==i, predDays), days)))
    rmses_final = _rmse_by_true_days_mask(deltaNextDays, num_sess>=64, predDays)

    fig, ax = newfig(width, height)
    ax.bar(days, errs, color='c0')
    ax.bar(64, errs_final, color='c0')
    ax.bar(days, rmses, color='c1', alpha=.1)
    ax.bar(64, rmses_final, color='c1', alpha=.1)

    ax.axhline(0, ls='--', color='k', linewidth=1)
    ax.set_xlabel('Number of active days')
    ax.set_ylabel('Mean error (days)')
    # ax.set_ylim((0,100))
    ax.set_ylim((-100,100))
    fig.tight_layout()
    fig.show()


def plot_churnacc_by_numofsess(dataset, interval=7, width=1, height=None):
    churned = np.load('../../results/evaluation/{}/churned.npy'.format(dataset))
    pred_churned = np.load('../../results/evaluation/{}/predictions_churn.npy'.format(dataset))
    pred_days = np.load('../../results/evaluation/{}/predictions_days.npy'.format(dataset))
    recency = np.load('../../results/evaluation/rmtpp_abs/recency.npy')
    num_sess = np.load('../../results/evaluation/num_sessions.npy')
    pred_daysinpred = pred_days - recency
    # deltaNextDays_rounded = np.floor(deltaNextDays/interval)
    predDays = np.load('../../results/evaluation/{}/predictions_days.npy'.format(dataset))
    days = np.arange(1, 64)
    rmses = np.array(list(map(lambda i: _churn_by_true_days_mask(pred_churned, churned, pred_daysinpred, num_sess==i, predDays), days)))
    rmses_final = _churn_by_true_days_mask(pred_churned, churned, pred_daysinpred, num_sess>=64, predDays)

    fig, ax = newfig(width, height)
    ax.bar(days, rmses)
    ax.bar(64, rmses_final, color='c0')

    # ax.axhline(0, ls='--', color='k', linewidth=1)
    ax.set_xlabel('Number of active days')
    # ax.set_ylabel('RMSE (days)')
    # ax.set_ylim((0,120))
    ax.set_xticks([0,20,40,64])
    ax.set_xticklabels(['0','20','40','$\geq64$'])
    fig.tight_layout()
    fig.show()

def _churn_by_true_days_mask(pred_churned, churned, pred_daysinpred, mask, predDays):
    y_pred = pred_churned[mask]
    y_true = churned[mask]
    acc = recall_score(y_true, y_pred)
    # acc = roc_auc_score(y_true, pred_daysinpred[mask])

    return acc


def calcConcordance(dataset):
    churned = np.load('../../results/evaluation/{}/churned.npy'.format(dataset))
    deltaNextDays = np.load('../../results/evaluation/{}/deltaNextDays.npy'.format(dataset))
    # recency = np.load('../../results/evaluation/{}/recency.npy'.format(dataset))
    recency = np.load('../../results/evaluation/rmtpp_abs/recency.npy')
    predDays = np.load('../../results/evaluation/{}/predictions_days.npy'.format(dataset))

    concordance = concordance_index(deltaNextDays-recency, predDays-recency, ~churned)
    return concordance



def calcAUC(dataset):
    churned = np.load('../../results/evaluation/{}/churned.npy'.format(dataset))
    deltaNextDays = np.load('../../results/evaluation/{}/deltaNextDays.npy'.format(dataset))
    # recency = np.load('../../results/evaluation/{}/recency.npy'.format(dataset))
    recency = np.load('../../results/evaluation/rmtpp_abs/recency.npy')
    predDays = np.load('../../results/evaluation/{}/predictions_days.npy'.format(dataset))

    auc = roc_auc_score(churned, predDays-recency)
    return auc

def mcnemar_midp(b, c):
    """
    Compute McNemar's test using the "mid-p" variant suggested by:

    M.W. Fagerland, S. Lydersen, P. Laake. 2013. The McNemar test for
    binary matched-pairs data: Mid-p and asymptotic are better than exact
    conditional. BMC Medical Research Methodology 13: 91.

    `b` is the number of observations correctly labeled by the first---but
    not the second---system; `c` is the number of observations correctly
    labeled by the second---but not the first---system.
    """
    n = b + c
    x = min(b, c)
    dist = binom(n, .5)
    p = 2. * dist.cdf(x)
    midp = p - dist.pmf(x)
    return midp
