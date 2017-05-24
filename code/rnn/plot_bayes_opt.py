import pickle
import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization

import sys
sys.path.insert(0, '../utils')
from plot_format import *
import seaborn as sns
from seaborn import apionly as sns

# RNN GP steps: MSE/hidden neur/n_sess
res = [(958.60321, 73.4674, 118.6166),
       (962.46741, 60.2672, 285.7071),
       (873.32880, 1.0000, 10.0000),
       (849.44653, 17.0205, 71.5381),
       (868.69800, 1.0000, 80.4154),
       (1119.38782, 100.0000, 10.0000),
       (836.22321, 13.7568, 63.9338),
       (1076.31970, 1.0000, 52.0294),
       (1081.11499, 1.0000, 203.4100),
       (903.43384, 1.0000, 300.0000)]
res = {'X': [(x[2], x[1]) for x in res], 'Y': [x[0] for x in res]}


def posterior(bo, res, grid):
    # xmin, xmax = 0, 5000
    # bo.gp.fit(bo.X[:steps], bo.Y[:steps])
    bo.gp.fit(res['X'], res['Y'])
    mu, sigma = bo.gp.predict(grid, return_std=True)
    return mu, sigma

def plot_gp(width=1, height=None):
    bounds = {'hidden_neurons': (1, 100), 'n_sessions': (10,300)}
    bo = BayesianOptimization(lambda x: 0, bounds)
    # bo = pickle.load(open(model.RESULT_PATH+'bayes_opt{}.pkl'.format(opt), 'rb'))

    x = np.arange(1, 301)
    y = np.arange(1, 101)
    grid = [(i,j) for j in y for i in x]

    fig, ax = newfig(width, height)

    mu, sigma = posterior(bo, res, grid)

    cs = ax.contourf(x,y,mu.reshape((100,300)), 20)

    samples_x = [x[0] for x in res['X']]
    samples_y = [x[1] for x in res['X']]
    sc = ax.scatter(samples_x, samples_y, label='Samples', color='C3', s=5)
    # ax.plot(bo.X[:steps].flatten(), bo.Y[:steps], 'D', markersize=8, label=u'Observations', color='r')
    # ax.plot(x, mu, '--', color='k', label='Prediction')

    # ax.fill(np.concatenate([x, x[::-1]]),
    #           np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
    #     alpha=.2, fc='black', ec='None', label=r'95\% confidence interval')

    ax.set_xlim((1, 300))
    ax.set_ylim((1, 100))
    # ax.set_ylabel(r'Concordance')
    # ax.set_xlabel(r'Penalizer')

    ax.set_xlabel('Number of active days')
    ax.set_ylabel('Number of LSTM cells')
    ax.legend()
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel('Posterior mean (MSE)')
    sc.set_clip_on(False)
    fig.tight_layout()
    fig.show()

def plot_gp_var(width=1, height=None):
    bounds = {'hidden_neurons': (1, 100), 'n_sessions': (10,300)}
    bo = BayesianOptimization(lambda x: 0, bounds)
    # bo = pickle.load(open(model.RESULT_PATH+'bayes_opt{}.pkl'.format(opt), 'rb'))

    x = np.arange(1, 301)
    y = np.arange(1, 101)
    grid = [(i,j) for j in y for i in x]

    fig, ax = newfig(width, height)

    mu, sigma = posterior(bo, res, grid)

    cs = ax.contourf(x,y,sigma.reshape((100,300)), 20)

    samples_x = [x[0] for x in res['X']]
    samples_y = [x[1] for x in res['X']]
    sc = ax.scatter(samples_x, samples_y, label='Samples', color='C3', s=5)
    # ax.plot(bo.X[:steps].flatten(), bo.Y[:steps], 'D', markersize=8, label=u'Observations', color='r')
    # ax.plot(x, mu, '--', color='k', label='Prediction')

    # ax.fill(np.concatenate([x, x[::-1]]),
    #           np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
    #     alpha=.2, fc='black', ec='None', label=r'95\% confidence interval')

    ax.set_xlim((1, 300))
    ax.set_ylim((1, 100))
    # ax.set_ylabel(r'Concordance')
    # ax.set_xlabel(r'Penalizer')

    ax.set_xlabel('Number of active days')
    ax.set_ylabel('Number of LSTM cells')
    ax.legend()
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel('Posterior variance')
    sc.set_clip_on(False)
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


def plot_grid_search(width=1, height=None):
    # res = {'penalties': space, 'scores': {k: [d[k] for d in scores] for k in scores[0]}}
    res = pickle.load(open(CoxChurnModel.RESULT_PATH+'grid_search_21.pkl', 'rb'))

    scores = {'churn_auc': 'Churn AUC',
              'churn_acc': 'Churn Accuracy',
              'rmse_days': 'RMSE',
              'concordance': 'Concordance'}
    x = res['penalties']
    y = res['scores']

    fig, ax = newfig(width, height)

    for k in scores:
        ax.plot(x, np.array(y[k])/np.max(y[k]), label=scores[k])

    ax.legend()
    fig.tight_layout()
    fig.show()
