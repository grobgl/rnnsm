import sys
sys.path.insert(0, '../utils')
from dataPiping import *

import os.path

import numpy as np
import pandas as pd
from math import exp, fabs, sqrt, log, pi

from random import random
import datetime

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.callbacks import Callback, LambdaCallback, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from keras import backend as K
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn_pandas import DataFrameMapper

seed = 42
np.random.seed(seed)

class Rmtpp:
    def __init__(self):
        self.setXy()
        self.setTrainTest()
        self.setModel()

    def setXy(self):
        if os.path.exists('../../data/experiments/rmpt1_X.npy'):
            self.X = np.load('../../data/experiments/rmpt1_X.npy')
            self.y = np.load('../../data/experiments/rmpt1_y.npy')
            return (self.X, self.y)

        numSessions = 30
        minNumSessions = 31
        inputCols = ['deltaPrevDays']

        df = pd.read_pickle('../../data/cleaned/stage2_obs_pred.pkl')
        obs = df[df.startUserTime < pd.datetime(2016,2,1)].copy()
        obs['deltaNext'] = pd.TimedeltaIndex(obs.deltaNext)
        obs['deltaPrev'] = pd.TimedeltaIndex(obs.deltaPrev)
        obs['deltaNextDays'] = obs.deltaNext.dt.days
        obs['deltaPrevDays'] = obs.deltaPrev.dt.days
        cust =obs.groupby('customerId').filter(lambda x: \
                len(x) >= minNumSessions and not x.tail(1).deltaNext.isnull()[0]).customerId.unique()
        nonChurnObs = obs[obs.customerId.isin(cust)]

        n = len(cust)
        X = np.zeros((n, numSessions, len(inputCols)))
        y = np.zeros(n)

        for i, (cust, group) in enumerate(nonChurnObs.groupby('customerId')):
            x_vals = group.tail(numSessions)[inputCols].as_matrix().astype('float64')

            X[i][numSessions - len(x_vals):] = x_vals.reshape((-1,len(inputCols)))
            y[i] = group.deltaNextDays.tail(1).values[0]

            sys.stdout.write('\r{:2.2f}%'.format(i/n*100))
            sys.stdout.flush()

        self.X = X
        self.y = y

        return (X,y)


    def setTrainTest(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=.2, random_state=42)
        self.bt =  np.log(1/self.X_train.reshape(-1).mean())
        return self.X_train, self.X_test, self.y_train, self.y_test


    def setModel(self, lr=0.001):
        # self.lr = lr
        # in_neurons = self.X_train.shape[2]
        # out_neurons = 1
        # hidden_neurons = 30

        # model = Sequential()
        # model.add(LSTM(hidden_neurons, return_sequences=False,
        #            input_shape=(None, in_neurons)))
        # model.add(Dense(out_neurons, input_dim=hidden_neurons))
        # model.add(Activation("relu"))
        # model.compile(loss="mean_squared_error", optimizer=Adam(lr=lr))


        self.lr = lr
        in_neurons = self.X_train.shape[2]
        out_neurons = 1
        hidden_neurons = 30

        model = Sequential()
        model.add(LSTM(hidden_neurons, return_sequences=False,
                   # input_shape=(None, in_neurons)))
                   input_shape=(None, in_neurons),activation='relu', recurrent_activation='relu'))
        model.add(Dense(out_neurons, input_dim=hidden_neurons))
        model.compile(loss=self._neg_log_likelihood, optimizer=Adam(lr=lr))
        # model.compile(loss="mean_squared_error", optimizer=Adam(lr=lr))

#         in_neurons = self.X_train.shape[2]
#         out_neurons = 1
#         lstm_neurons = 16

#         model = Sequential()

#         # recurrent layer (weights W_h)
#         model.add(LSTM(
#             lstm_neurons,
#             return_sequences=False,
#             input_shape=(None, in_neurons), activation='relu'))

#         # output layer (weights v_t)
#         model.add(Dense(
#             out_neurons,
#             input_dim=lstm_neurons,
#             activation='linear'))

#         # model.compile(loss=self._neg_log_likelihood, optimizer=Adam(lr=lr))
#         model.compile(loss='mse', optimizer=Adam(lr=lr))

        self.model = model
        return model


    def _neg_log_likelihood(self, deltaT, acc_influence):
        """ Loss function for RMTPP model

        :timings: vector of deltaT: [t_(j+1) - t_j]
        :acc_influence: rnn output = v_t * h_j
        """
        wt = .01
        w = 100
        bt = self.bt

        return -acc_influence - wt*deltaT \
               - bt - 1/w*K.exp(acc_influence + bt) \
               + 1/w*K.exp(acc_influence + wt*(deltaT) + bt)


    def fitModel(self,run):
        log_file = 'rmtpp_lr{}_inp{}_run{}'.format(self.lr,self.X_train.shape[2],run)
        self.model.fit(self.X_train, self.y_train, batch_size=1000, epochs=5000, validation_split=0.2, verbose=0
              , callbacks=[
                TensorBoard(log_dir='../../logs/rmtpp/{}'.format(log_file), histogram_freq=100)
                , EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
                ]
             )


