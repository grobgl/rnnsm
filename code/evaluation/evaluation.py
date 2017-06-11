import pickle
import pandas as pd
import numpy as np
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from bayes_opt import BayesianOptimization

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

