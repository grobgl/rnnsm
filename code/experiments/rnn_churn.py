import sys
sys.path.insert(0, '../utils')
from dataPiping import *

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
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, confusion_matrix


seed = 42
np.random.seed(seed)

class RnnModel:
    def __init__(self):
        self.df = self.loadData()
        self.X_train, self.X_test, self.y_train, self.y_test = self.formatData(self.df)
        self.model = self.getModel()

    def loadData(self):
        return pd.read_pickle('../../data/cleaned/stage2.pkl')


    def formatData(self, df):
        numEvents = 30
        inputCols = ['returnDays','hourOfDay','dayOfWeek','dayOfMonth','weekOfYear','month','numSessions']

        # customers that have at least 31 sessions  between feb 2015 and feb 2016 and after feb 2016
        cust = df[(df.startUserTime < pd.datetime(year=2016,month=2,day=1)) & \
                (df.startUserTime >= pd.datetime(year=2015,month=2,day=1))].groupby('customerId').customerId.filter(lambda x: len(x) >= 1).unique()
        n = len(cust)

        dfRet = df[df.customerId.isin(cust)]
        dfRet['returnDays'] = pd.TimedeltaIndex(dfRet.returnTime).days

        grouped = dfRet.groupby('customerId')

        X = np.zeros((n, numEvents, len(inputCols)))
        y = np.zeros(n)

        for i, (cust, group) in enumerate(grouped):
            before = group[group.startUserTime < pd.datetime(year=2016,month=2,day=1)]
            x_vals = before.tail(numEvents + 1)
            x_vals = x_vals.head(len(x_vals) - 1)[inputCols].as_matrix().astype('float64')

            X[i][numEvents-len(x_vals):] = x_vals.reshape((-1,len(inputCols)))
            y[i] = int(before.returnDays.tail(1).isnull())
            sys.stdout.write('\r{:2.2f}%'.format(i/n*100))
            sys.stdout.flush()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

        return X_train, X_test, y_train, y_test


    def getModel(self, lr=10):
        in_neurons = self.X_train.shape[2]
        out_neurons = 1
        hidden_neurons = 30

        model = Sequential()
        model.add(LSTM(hidden_neurons, return_sequences=False,
                   input_shape=(None, in_neurons)))
        model.add(Dense(out_neurons, input_dim=hidden_neurons))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr))

        return model


    def setModel(self, model):
        self.model = model


    def fitModel(self,run,lr=10,hidden=30):
        log_dir = 'rnn_lr{}_inp{}_hidden{}_run{}'.format(lr,self.X_train.shape[2],hidden,run)
        self.model.fit(self.X_train, self.y_train, batch_size=10000, epochs=5000, validation_split=0.2, verbose=0
              , callbacks=[
                TensorBoard(log_dir='../../logs/rnn_churn/{}'.format(log_dir), histogram_freq=100)
                , EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
                ]
             )

    def testModel(self):
        predicted = self.model.predict(self.X_test)
        print(confusion_matrix(self.y_test,np.rint(predicted)))
        rmse = sqrt(mean_squared_error(self.y_test, predicted))
        print('RMSE: {}'.format(rmse))
