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
# from plot_format import *


_RESULT_PATH = '../../results/churn/random_forest/'


def runGridSearch():
    param_grid = {
        'n_estimators': [50],
        'max_features': range(10,30),
        'max_depth': range(1,10),
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

