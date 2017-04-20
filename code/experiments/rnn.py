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
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn_pandas import DataFrameMapper

seed = 42
np.random.seed(seed)

class RnnModel:
    def __init__(self):
        self.df = self.loadData()
        self.X, self.y = self.getXy(self.df)
        self.X_train, self.X_test, self.y_train, self.y_test = self.splitData(self.X,self.y)
        self.model = self.getModel()

    def loadData(self):
        return pd.read_pickle('../../data/cleaned/stage2_pruned.pkl')


    def getXy(self, df):
        numSessions = 30
        minNumSessions = 5
        inputCols = ['deltaPrevDays','hourOfDay','dayOfWeek','dayOfMonth','weekOfYear','month','numSessions','sessionLengthSec']

        # binarizer = ['device']
        # scalars = ['hourOfDay','dayOfWeek','dayOfMonth','weekOfYear','month','numSessions','sessionLengthSec']

        # mapper = DataFrameMapper(
        #     [(binarizer, LabelBinarizer())] +
        #     [(scalar, MinMaxScaler(feature_range=(0, 1))) for scalar in scalars]
        # )

        # customers that have at least 31 sessions  between feb 2015 and feb 2016 and after feb 2016
        beforeCust = df[(df.startUserTime < pd.datetime(year=2016,month=2,day=1)) & \
                (df.startUserTime >= pd.datetime(year=2015,month=2,day=1))].groupby('customerId').customerId.filter(lambda x: len(x) >= minNumSessions).unique()

        afterCust= df[df.startUserTime >= pd.datetime(year=2016,month=2,day=1)].customerId.unique()

        cust = np.intersect1d(beforeCust, afterCust)
        n = len(cust)

        dfRet = df[df.customerId.isin(cust)]
        dfRet['deltaPrevDays'] = pd.TimedeltaIndex(dfRet.deltaPrev).days
        dfRet['deltaNextDays'] = pd.TimedeltaIndex(dfRet.deltaNext).days
        dfRet = dfRet[dfRet.startUserTime < pd.datetime(year=2016,month=2,day=1)]
        dfRet = dfRet.groupby('customerId').tail(numSessions)

        # scale data
        for col in inputCols:
            noNanCol = dfRet.ix[~dfRet[col].isnull(), col].values
            dfRet.ix[~dfRet[col].isnull(), col] = MinMaxScaler(feature_range=(0, 1)).fit_transform(noNanCol)

        # # scale output
		# noNanCol = dfRet.ix[~dfRet[col].isnull(), 'deltaNextDays'].values
        # self.output_scaler = MinMaxScaler(feature_range=(0, 1)).fit(noNanCol)
        # dfRet.ix[~dfRet['deltaNextDays'].isnull(), 'deltaNextDays'] = self.output_scaler.transform(noNanCol)


        # sets deltaPrevDays of first visit to 0
        dfRet.fillna(0, inplace=True)

        grouped = dfRet.groupby('customerId')

        X = np.zeros((n, numSessions, len(inputCols)))
        y = np.zeros(n)

        for i, (cust, group) in enumerate(grouped):
            # x_vals = group.tail(numSessions)[inputCols].as_matrix().astype('float64')
            x_vals = group.tail(numSessions)[inputCols].as_matrix().astype('float64')

            X[i][numSessions - len(x_vals):] = x_vals.reshape((-1,len(inputCols)))
            y[i] = group.deltaNextDays.tail(1).values[0]
            sys.stdout.write('\r{:2.2f}%'.format(i/n*100))
            sys.stdout.flush()

        # fit output scaler
        # self.output_scaler = MinMaxScaler(feature_range=(0, 1)).fit(y)

        return X,y


    def splitData(self,X,y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
        return X_train, X_test, y_train, y_test


    def getModel(self, lr=100):
        in_neurons = self.X_train.shape[2]
        out_neurons = 1
        hidden_neurons = 30

        model = Sequential()
        model.add(LSTM(hidden_neurons, return_sequences=False,
                   input_shape=(None, in_neurons)))
        model.add(Dense(out_neurons, input_dim=hidden_neurons))
        model.add(Activation("relu"))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=lr))

        return model


    def setModel(self, model):
        self.model = model


    def fitModel(self,run,lr=100,hidden=30):
        log_dir = 'rnn_lr{}_inp{}_hidden{}_run{}'.format(lr,self.X_train.shape[2],hidden,run)
        self.model.fit(self.X_train, self.y_train, batch_size=1000, epochs=5000, validation_split=0.2, verbose=0
              , callbacks=[
                TensorBoard(log_dir='../../logs/rnn/{}'.format(log_dir), histogram_freq=100)
                , EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
                ]
             )

    def testModel(self):
        predicted = self.model.predict(self.X_test)
        # predicted = self.output_scaler.inverse_transform(predicted)
        rmse = sqrt(mean_squared_error(self.y_test, predicted))
        print('RMSE: {}'.format(datetime.timedelta(days=rmse)))
