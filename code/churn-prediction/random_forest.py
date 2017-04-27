from churn_data import ChurnData
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import itertools

import pickle
import os
import sys
sys.path.insert(0, '../utils')
from plot_format import *


_RESULT_PATH = '../../results/churn/random_forest/'

def printTestSetResultsRandFor():
    grid = pickle.load(open(_RESULT_PATH+'grid_search_result.pkl','rb'))
    model = RandomForestClassifier(**grid.best_params_)
    data = ChurnData()

    model.fit(**data.train)

    return data.printScores(model)


def runGridSearch():
    param_grid = {
        # 'n_estimators': [50],
        'n_estimators': [100],
        'max_features': range(10,30),
        # 'max_depth': range(1,10),
        'max_depth': range(1,20),
        'min_samples_leaf': range(5,25)
    }

    data = ChurnData()
    model = RandomForestClassifier()

    # fixed random state for cross validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # default scoring is accuracy
    grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=1, n_jobs=64, cv=cv)
    grid.fit(**data.train)

    with open(_RESULT_PATH+'grid_search_result.pkl', 'wb') as handle:
        pickle.dump(grid, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return grid

def getResultGrid():
    grid = pickle.load(open(_RESULT_PATH+'grid_search_result.pkl', 'rb'))
    return grid

def getBestModel():
    grid = pickle.load(open(_RESULT_PATH+'grid_search_result.pkl', 'rb'))
    return grid.best_estimator_

def bestModelAuc():
    # auc = 0.8026

    params = getResultGrid().best_params_
    data = ChurnData()

    # fixed random state for cross validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    aucs = []

    for i, (trainIndex, testIndex) in enumerate(cv.split(**data.train)):
        print('Split{} out of 10'.format(i+1))
        model = RandomForestClassifier(**params)
        model.fit(data.train['X'][trainIndex], data.train['y'][trainIndex])
        aucs.append(data.getScores(model, X=data.train['X'][testIndex], y=data.train['y'][testIndex])['auc'])

    return np.mean(aucs)


def getFeatureImportances():
    model = getBestModel()
    data = ChurnData()

    importances = pd.DataFrame(list(zip(data.features, model.feature_importances_)))
    importances.columns = ['feature','importance']

    return importances.sort_values(by='importance', ascending=False)


