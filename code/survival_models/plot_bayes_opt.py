import pickle
import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization

import sys
sys.path.insert(0, '../utils')
from plot_format import *
import seaborn as sns
from seaborn import apionly as sns


def posterior(bo, x, steps):
    xmin, xmax = 0, 5000
    bo.gp.fit(bo.X[:steps], bo.Y[:steps])
    mu, sigma = bo.gp.predict(x, return_std=True)
    return mu, sigma

def plot_gp(model, opt='', steps=100, width=1, height=None):
    bo = pickle.load(open(model.RESULT_PATH+'bayes_opt{}.pkl'.format(opt), 'rb'))

    x = np.linspace(0, 5000, 10000).reshape(-1, 1)

    fig, ax = newfig(width, height)

    mu, sigma = posterior(bo, x, steps)
    ax.plot(bo.X[:steps].flatten(), bo.Y[:steps], 'D', markersize=8, label=u'Observations', color='r')
    ax.plot(x, mu, '--', color='k', label='Prediction')

    ax.fill(np.concatenate([x, x[::-1]]),
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.2, fc='black', ec='None', label=r'95\% confidence interval')

    ax.set_xlim((0, 5000))
    ax.set_ylim((None, None))
    ax.set_ylabel(r'Concordance')
    ax.set_xlabel(r'Penalizer')

    ax.legend()
    fig.tight_layout()
    fig.show()


def plot_gp_multiple(model, opt='', steps=[2,8,16], width=1, height=None):
    bo = pickle.load(open(model.RESULT_PATH+'bayes_opt{}.pkl'.format(opt), 'rb'))

    x = np.linspace(0, 5000, 10000).reshape(-1, 1)

    ax = {}
    fig, ax[steps[0]] = newfig(width, height, 131)
    ax[steps[1]] = fig.add_subplot(132)
    ax[steps[2]] = fig.add_subplot(133)

    # p = {i: {'mu': mu, 'sigma': sigma} for mu, sigma in posterior(bo, x, i) for i in steps}
    p = {i: {'mu': mu, 'sigma': sigma} for i, (mu, sigma) in [(i, posterior(bo, x, i)) for i in steps]}
    obs = {i: ax[i].plot(
                            bo.X[:i].flatten(),
                            bo.Y[:i],
                            '.', markersize=8, label=u'Observations', color='r')[0]
            for i in steps}
    pred = {i: ax[i].plot(x, p[i]['mu'], '--', color='k', label='Prediction')[0] for i in steps}

    conf = {i: ax[i].fill(
                    np.concatenate([x, x[::-1]]),
                    np.concatenate(
                        [p[i]['mu'] - 1.9600 * p[i]['sigma'],
                        (p[i]['mu'] + 1.9600 * p[i]['sigma'])[::-1]]),
                    alpha=.2, fc='black', ec='None', label=r'95\% confidence interval')[0]
            for i in steps}

    ax[steps[1]].set_yticklabels([])
    ax[steps[2]].set_yticklabels([])
    ax[steps[0]].set_ylabel('Concordance')
    [ax[i].set_xlabel(r'$\gamma$') for i in steps]
    [ax[i].set_xlim((0, 5000)) for i in steps]
    [ax[i].set_ylim((.73,.84)) for i in steps]

    fig.legend(handles=[obs[steps[0]], pred[steps[0]], conf[steps[0]]], labels=[r'Observations', r'Prediction', r'95\% confidence interval'], loc='upper center', ncol=3, framealpha=1, bbox_to_anchor=(0.55, 0.91))
    fig.tight_layout()
    fig.show()


def plot_gp_measures(model, steps=20, width=1, height=None):
    measures = {'churn_auc':'Churn AUC', 'churn_acc': 'Churn accuracy', 'rmse_days': 'RMSE', 'concordance':'Concordance'}

    bo = {m: pickle.load(open(model.RESULT_PATH+'bayes_opt_{}.pkl'.format(m), 'rb')) for m in measures}
    for b in bo: bo[b].Y = bo[b].Y/bo[b].Y.max()

    x = np.linspace(2000, 3000, 10000).reshape(-1, 1)

    fig, ax = newfig(width, height)

    # p = {i: {'mu': mu, 'sigma': sigma} for mu, sigma in posterior(bo, x, i) for i in steps}
    p = {i: {'mu': mu, 'sigma': sigma} for i, (mu, sigma) in [(i, posterior(bo[i], x, steps)) for i in measures]}
    obs = {i: ax.plot(
                            bo[i].X[:steps].flatten(),
                            bo[i].Y[:steps],
                            '.', markersize=8, label=u'Observations', color='r')[0]
            for i in measures}
    pred = {i: ax.plot(x, p[i]['mu'], '--', label=measures[i])[0] for i in measures}

    # conf = {i: ax[i].fill(
    #                 np.concatenate([x, x[::-1]]),
    #                 np.concatenate(
    #                     [p[i]['mu'] - 1.9600 * p[i]['sigma'],
    #                     (p[i]['mu'] + 1.9600 * p[i]['sigma'])[::-1]]),
    #                 alpha=.2, fc='black', ec='None', label=r'95\% confidence interval')[0]
    #         for i in steps}

    # ax[steps[1]].set_yticklabels([])
    # ax[steps[2]].set_yticklabels([])
    # ax[steps[0]].set_ylabel('Concordance')
    # [ax[i].set_xlabel(r'$\gamma$') for i in steps]
    # [ax[i].set_xlim((0, 5000)) for i in steps]
    # [ax[i].set_ylim((.73,.84)) for i in steps]

    # fig.legend(handles=[obs[steps[0]], pred[steps[0]], conf[steps[0]]], labels=[r'Observations', r'Prediction', r'95\% confidence interval'], loc='upper center', ncol=3, framealpha=1, bbox_to_anchor=(0.55, 0.91))
    ax.legend()
    fig.tight_layout()
    fig.show()
