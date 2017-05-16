import os.path

import numpy as np
import pandas as pd

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, Masking, concatenate, Embedding
from keras.layers.recurrent import LSTM
from keras.callbacks import Callback, LambdaCallback, TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K

from sklearn.metrics import mean_squared_error

from rmtpp_data import *

seed = 42
np.random.seed(seed)

class Rmtpp:

    def __init__(self, name, run):
        self.predict_sequence = False
        self.data = RmtppData.instance()
        self.set_x_y()
        self.set_model()
        self.name = name
        self.run = run
        self.best_model_cp_file = '../../results/rmtpp/{:02d}_{}.hdf5'.format(run, name)
        self.best_model_cp = ModelCheckpoint(self.best_model_cp_file, monitor="val_loss",
                                             save_best_only=True, save_weights_only=False)

    def set_x_y(self, include_churned=False, min_n_sessions=0, n_sessions=100, preset='deltaNextHours'):
        self.x_train, \
        self.x_test, \
        self.y_train, \
        self.y_test, \
        features = self.data.get_xy(include_churned, min_n_sessions, n_sessions, preset=preset, target_sequences=self.predict_sequence)

        temporal_features = ['dayOfMonth', 'dayOfWeek', 'hourOfDay', 'deltaPrevHours', 'startUserTimeHours']
        device_index = features.index('device')
        temporal_indices = list(map(features.index, temporal_features))
        behav_indices = list(map(features.index, set(features) - set(temporal_features + ['device'])))

        self.x_train_devices = self.x_train[:,:,device_index]
        self.x_train_temporal = self.x_train[:,:,temporal_indices]
        self.x_train_behav = self.x_train[:,:,behav_indices]
        self.x_test_devices = self.x_test[:,:,device_index]
        self.x_test_temporal = self.x_test[:,:,temporal_indices]
        self.x_test_behav = self.x_test[:,:,behav_indices]

        if self.predict_sequence:
            self.y_train = self.y_train.reshape(self.y_train.shape + (1,))
            self.y_test = self.y_test.reshape(self.y_test.shape + (1,))

        # self.y_train = self.y_train.reshape((-1, n_sessions, 1))
        # self.y_test = self.y_test.reshape((-1, n_sessions, 1))

    def load_best_weights(self):
        self.model.load_weights(self.best_model_cp_file)

    def set_model(self, lr=100.):
        self.lr = lr
        len_seq = self.x_train.shape[1]
        n_devices = np.unique(self.x_train_devices).shape[0]
        len_temporal = self.x_train_temporal.shape[2]
        len_behav = self.x_train_behav.shape[2]

        dense_neurons = 64
        lstm_neurons = 64

        # use embedding layer for devices
        device_input = Input(shape=(len_seq,), dtype='int32', name='device_input')
        device_embedding = Embedding(output_dim=4, input_dim=n_devices,
                                     input_length=len_seq, mask_zero=True)(device_input)

        # inputs for temporal and behavioural data
        temporal_input = Input(shape=(len_seq, len_temporal), name='temporal_input')
        behav_input = Input(shape=(len_seq, len_behav), name='behav_input')

        temporal_masking = Masking(mask_value=0.)(temporal_input)
        behav_masking = Masking(mask_value=0.)(behav_input)

        merge_inputs = concatenate([device_embedding, temporal_masking, behav_masking])
        lstm_output = LSTM(lstm_neurons, return_sequences=self.predict_sequence)(merge_inputs)

        predictions = Dense(1, activation='relu', name='predictions')(lstm_output)

        model = Model(inputs=[device_input, temporal_input, behav_input], outputs=predictions)
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=lr))

        self.model = model
        return model

        # maskingLayer = Masking(mask_value=0., input_shape=input_shape)(inputs)

    def get_rmse_days(self, last_only=False):
        y_pred = self.model.predict([self.x_test_devices, self.x_test_temporal, self.x_test_behav])
        y_true = self.y_test
        if last_only:
            y_pred = y_pred[:,-1,0]
            y_true = y_true[:,-1,0]

        rmse_days = np.sqrt(mean_squared_error(y_true.ravel()/24, y_pred.ravel()/24))

        return rmse_days

    def fit_model(self, initial_epoch=0):
        log_file = '{:02d}_{}_lr{}_inp{}'.format(self.run, self.name, self.lr,self.x_train.shape[2])
        self.model.fit([self.x_train_devices, self.x_train_temporal, self.x_train_behav], self.y_train, batch_size=1000, epochs=5000, validation_split=0.2, verbose=0, initial_epoch=initial_epoch
              , callbacks=[
                TensorBoard(log_dir='../../logs/rmtpp/{}'.format(log_file), histogram_freq=100)
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


