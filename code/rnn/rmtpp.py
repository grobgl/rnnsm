import os.path

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.callbacks import Callback, LambdaCallback, TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K

from sklearn.metrics import mean_squared_error

from rnn_data import *

seed = 42
np.random.seed(seed)

class Rnn:

    def __init__(self, name):
        self.data = RnnData.instance()
        self.set_x_y()
        self.set_model()
        self.model_chk_path = '../../results/rnn/{}.hdf5'.format(name)

    def set_x_y(self, include_churned=False, min_n_sessions=0, n_sessions=90, preset='startUserTimeHours'):
        self.x_train, \
        self.x_test, \
        self.y_train, \
        self.y_test = self.data.get_xy(include_churned, min_n_sessions, n_sessions, preset=preset)

    def load_best_weights(self):
        self.model.load_weights(self.model_chk_path)

    def set_model(self, lr=10):
        self.lr = lr
        in_neurons = self.x_train.shape[2]
        out_neurons = 1
        hidden_neurons = 64

        model = Sequential()
        model.add(Dense(64, input_shape=(None, in_neurons), activation='tanh'))
        model.add(LSTM(hidden_neurons, return_sequences=False))
        model.add(Dense(out_neurons, input_dim=hidden_neurons))
        model.add(Activation("relu"))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=lr))
        # model = Sequential()
        # model.add(LSTM(
        #         hidden_neurons,
        #         return_sequences=False,
        #         input_shape=(None, in_neurons),
        #         activation='relu',
        #         recurrent_activation='relu'))
        # model.add(Dense(out_neurons, input_dim=hidden_neurons))
        # # model.compile(loss=self._neg_log_likelihood, optimizer=Adam(lr=lr))
        # model.compile(loss='mse', optimizer=Adam(lr=lr))

        self.model = model
        return model

    def get_rmse_days(self):
        pred = self.model.predict(self.x_test)
        err = pred - self.y_test
        rmse_days = np.sqrt(mean_squared_error(self.y_test/24, pred/24))

        return rmse_days

    def fit_model(self,run, initial_epoch=0):
        self.best_model_cp = ModelCheckpoint(self.model_chk_path, monitor="val_loss",
                                             save_best_only=True, save_weights_only=False)
        log_file = 'rmtpp_lr{}_inp{}_run{}'.format(self.lr,self.x_train.shape[2],run)
        self.model.fit(self.x_train, self.y_train, batch_size=1000, epochs=5000, validation_split=0.2, verbose=0, initial_epoch=initial_epoch
              , callbacks=[
                TensorBoard(log_dir='../../logs/rnn_new/{}'.format(log_file), histogram_freq=100)
                , EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
                , self.best_model_cp
                ]
             )

def _neg_log_likelihood(deltaT, acc_influence):
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


