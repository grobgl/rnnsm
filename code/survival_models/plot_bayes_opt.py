import pickle
import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization

import sys
sys.path.insert(0, '../utils')
from plot_format import *
import seaborn as sns
from seaborn import apionly as sns


def posterior(bo, x):
    xmin, xmax = 0, 5000
    bo.gp.fit(bo.X, bo.Y)
    mu, sigma = bo.gp.predict(x, return_std=True)
    return mu, sigma

def plot_gp(model, width=1, height=None):
    bo = pickle.load(open(model.RESULT_PATH+'bayes_opt.pkl', 'rb'))

    x = np.linspace(0, 5000, 10000).reshape(-1, 1)

    fig, ax = newfig(width, height)

    mu, sigma = posterior(bo, x)
    ax.plot(bo.X.flatten(), bo.Y, 'D', markersize=8, label=u'Observations', color='r')
    ax.plot(x, mu, '--', color='k', label='Prediction')

    ax.fill(np.concatenate([x, x[::-1]]),
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.2, fc='black', ec='None', label=r'95\% confidence interval')

    ax.set_xlim((0, 5000))
    ax.set_ylim((None, None))
    ax.set_ylabel(r'Concordance')
    ax.set_xlabel(r'Penalizer')


    ax.legend()
    fig.show()
