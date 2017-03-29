import sys
sys.path.insert(0, '../utils')
from dataPiping import *
# import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from math import exp, fabs, sqrt, log, pi
import datetime

from keras.models import Sequential
from keras.callbacks import ProgbarLogger
from keras.layers import Dense, Dropout
from keras.callbacks import Callback, LambdaCallback, TensorBoard
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error

X, y = readAggrData()
del X['recency']
X_train, X_test, y_train, y_test = splitAndNormaliseAggr(X,y)

def wide_model():
    model = Sequential()
    model.add(Dense(20, input_dim=X.shape[1], init='normal', activation='relu'))
    model.add(Dense(1, init='normal')) # no activation/linear activation
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def deep_model():
    model = Sequential()
    model.add(Dense(X.shape[1], input_dim=X.shape[1], init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, init='normal')) # no activation/linear activation
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def deeper_model():
    model = Sequential()
    model.add(Dense(20, input_dim=X.shape[1], init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(13, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, init='normal')) # no activation/linear activation
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def fit_model(model, batch_size=1000, nb_epoch=10000, log_every=1000):
    hist = model.fit(
	X_train, y_train,
	batch_size=batch_size, nb_epoch=nb_epoch,
	verbose=0
        , callbacks=[TensorBoard(log_dir='../../logs', histogram_freq=100)]
	# , callbacks=[LambdaCallback(on_epoch_end=lambda epoch, logs: print(epoch, logs) if (epoch % log_every== 0) else 0)]
    )
    return hist


