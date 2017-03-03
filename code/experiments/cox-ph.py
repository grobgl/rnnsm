import sys
sys.path.insert(0, '../utils')

from dataPiping import *

import numpy as np
import pandas as pd

from math import sqrt
import datetime

from sklearn.metrics import mean_squared_error

import lifelines as sa
from lifelines.utils import concordance_index, k_fold_cross_validation

X, y = readAggrCoxPhData(include_cens=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)
X_train['returnTime'] = y_train


