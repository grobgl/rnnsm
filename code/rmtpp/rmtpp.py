import os.path

import numpy as np
import pandas as pd
from operator import mul
from functools import reduce, partial

from scipy.integrate import trapz

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, Masking, concatenate, Embedding, RepeatVector, Reshape
from keras.layers.recurrent import LSTM
from keras.callbacks import Callback, LambdaCallback, TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, RMSprop
from keras.initializers import Constant, Zeros
from keras.constraints import non_neg
from keras import backend as K

from sklearn.metrics import mean_squared_error

from rmtpp_data import *

seed = 42
np.random.seed(seed)

class Rmtpp:

    w_scale = 0.1
    time_scale = 0.1

    def __init__(self, name, run):
        self.predict_sequence = True
        self.data = RmtppData.instance()
        self.set_x_y()
        self.set_model()
        self.name = name
        self.run = run
        self.best_model_cp_file = '../../results/rmtpp/{:02d}_{}.hdf5'.format(run, name)
        self.best_model_cp = ModelCheckpoint(self.best_model_cp_file, monitor="val_loss",
                                             save_best_only=True, save_weights_only=False)

    def set_x_y(self, include_churned=False, min_n_sessions=0, n_sessions=50, preset='deltaNextDays'):
        self.x_train, \
        self.x_test, \
        self.x_train_unscaled, \
        self.x_test_unscaled, \
        self.y_train, \
        self.y_test, \
        features, \
        target_features = self.data.get_xy(include_churned, min_n_sessions, n_sessions, preset=preset, target_sequences=self.predict_sequence)

        self.features = features
        temporal_features = ['dayOfMonth', 'dayOfWeek', 'hourOfDay', 'deltaPrevDays', 'startUserTimeDays']
        device_index = features.index('device')
        temporal_indices = list(map(features.index, temporal_features))
        behav_indices = list(map(features.index, set(features) - set(temporal_features + ['device'])))
        startTimeDaysIndex = features.index('startUserTimeDays')

        self.x_train_devices = self.x_train[:,:,device_index]
        self.x_train_temporal = self.x_train[:,:,temporal_indices]
        self.x_train_behav = self.x_train[:,:,behav_indices]
        self.x_train_bias = np.ones(self.x_train.shape[:1] + (1,))
        self.x_train_mask = ~(self.x_train == 0).all(2)
        self.train_starttimes = self.x_train_unscaled[:,:,startTimeDaysIndex] * self.time_scale

        self.x_test_devices = self.x_test[:,:,device_index]
        self.x_test_temporal = self.x_test[:,:,temporal_indices]
        self.x_test_behav = self.x_test[:,:,behav_indices]
        self.x_test_bias = np.ones(self.x_test.shape[:1] + (1,))
        self.x_test_mask = ~(self.x_test == 0).all(2)
        self.test_starttimes = self.x_test_unscaled[:,:,startTimeDaysIndex] * self.time_scale

        self.y_train = self.y_train * self.time_scale
        self.y_test = self.y_test * self.time_scale
        self.train_nextstarttime = self.y_train.T[0].T
        self.y_train = self.y_train.T[1].T
        self.test_nextstarttime = self.y_test.T[0].T
        self.y_test = self.y_test.T[1].T

        if self.predict_sequence:
            self.y_train = self.y_train.reshape(self.y_train.shape + (1,))
            self.y_test = self.y_test.reshape(self.y_test.shape + (1,))
            self.y_train = np.append(self.y_train, self.x_train_mask.astype('float').reshape(self.x_train_mask.shape + (1,)), 2)
            self.y_test = np.append(self.y_test, self.x_test_mask.astype('float').reshape(self.x_test_mask.shape + (1,)), 2)

        self.y_train = self.y_train.astype('float32')
        self.y_test = self.y_test.astype('float32')


    def load_best_weights(self):
        self.model.load_weights(self.best_model_cp_file)

    def set_model(self, lr=.001):
        self.lr = lr
        len_seq = self.x_train.shape[1]
        n_devices = np.unique(self.x_train_devices).shape[0]
        len_temporal = self.x_train_temporal.shape[2]
        len_behav = self.x_train_behav.shape[2]

        dense_neurons = 32
        lstm_neurons = 32

        # use embedding layer for devices
        device_input = Input(shape=(len_seq,), dtype='int32', name='device_input')
        device_embedding = Embedding(output_dim=2, input_dim=n_devices,
                                     input_length=len_seq, mask_zero=True)(device_input)

        # inputs for temporal and behavioural data
        temporal_input = Input(shape=(len_seq, len_temporal), name='temporal_input')
        behav_input = Input(shape=(len_seq, len_behav), name='behav_input')

        temporal_masking = Masking(mask_value=0.)(temporal_input)
        behav_masking = Masking(mask_value=0.)(behav_input)

        merge_inputs = concatenate([device_embedding, temporal_masking, behav_masking])
        # merge_inputs = concatenate([temporal_masking, behav_masking])

        lstm_output = LSTM(lstm_neurons, return_sequences=self.predict_sequence, recurrent_activation='relu')(merge_inputs)

        predictions = Dense(1, activation='linear')(lstm_output)

        bias_input = Input(shape=(1,), name='bias_input')
        # bias_w_t = Dense(1, activation = 'linear', name='bias_w_t', kernel_initializer=Zeros(), bias_initializer=Constant(.001), kernel_constraint=non_neg(), bias_constraint=non_neg())(bias_input)
        # bias_input = Masking(mask_value=0.)(bias_input)
        bias_w = Dense(1, activation = 'linear', kernel_constraint=non_neg(), bias_constraint=non_neg())(bias_input)
        bias_w = RepeatVector(len_seq)(bias_w)

        output = concatenate([predictions, bias_w])

        # model = Model(inputs=temporal_input, outputs=predictions)
        # model = Model(inputs=[device_input, temporal_input, behav_input, bias_input], outputs=output)
        # model = Model(inputs=[temporal_input, behav_input, bias_input], outputs=[predictions, bias_w])
        model = Model(inputs=[device_input, temporal_input, behav_input, bias_input], outputs=output)

        # model.compile(loss=neg_log_likelihood, optimizer=RMSprop(lr=lr))
        model.compile(loss=self.neg_log_likelihood, optimizer=RMSprop(lr=lr))

        self.model = model
        return model

        # maskingLayer = Masking(mask_value=0., input_shape=input_shape)(inputs)

    def fit_model(self, initial_epoch=0):
        log_file = '{:02d}_{}_lr{}_inp{}'.format(self.run, self.name, self.lr,self.x_train.shape[2])
        # self.model.fit([self.x_train_temporal, self.x_train_behav, self.x_train_bias], self.y_train, batch_size=1000, epochs=5000, validation_split=0.2, verbose=0, initial_epoch=initial_epoch
        # self.model.fit([self.x_train_temporal, self.x_train_behav, self.x_train_bias], [self.y_train, np.ones((self.y_train.shape[0],1))], batch_size=1000, epochs=5000, validation_split=0.2, verbose=0, initial_epoch=initial_epoch
        self.model.fit([self.x_train_devices, self.x_train_temporal, self.x_train_behav, self.x_train_bias], self.y_train, batch_size=1000, epochs=5000, validation_split=0.2, verbose=0, initial_epoch=initial_epoch
        # self.model.fit(self.x_train_temporal, self.y_train, batch_size=1000, epochs=5000, validation_split=0.2, verbose=0, initial_epoch=initial_epoch
              , callbacks=[
                TensorBoard(log_dir='../../logs/rmtpp/{}'.format(log_file), histogram_freq=100)
                , EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
                , self.best_model_cp
                ]
             )

    def neg_log_likelihood(self, targets, output):
        """ Loss function for RMTPP model

        :targets: vector of: [t_(j+1) - t_j, mask]
        :output: rnn output = v_t * h_j + b_t
        """

        # w_t = self.w_scale
        mask = K.batch_flatten(targets[:,:,1])
        w = K.batch_flatten(output[:,:,1])
        w_t = w
        cur_state = K.batch_flatten(output[:,:,0])
        delta_t = K.batch_flatten(targets[:,:,0])

        res = -cur_state - w_t*delta_t \
               - (1/w)*K.exp(cur_state) \
               + (1/w)*K.exp(cur_state + w_t*(delta_t))

        # return res
        return res*mask


    def get_rmse_days(self, last_only=False, dataset='test'):
        pred_next_starttime_vec = np.vectorize(self.pred_next_starttime)

        if dataset=='test':
            pred = self.model.predict([self.x_test_devices, self.x_test_temporal, self.x_test_behav, self.x_test_bias])
            cur_states = pred[:,-1,0]
            ws = pred[:,-1,1]
            t_js = self.test_starttimes[:,-1]
            t_true = self.test_nextstarttime[:,-1]
        else:
            pred = self.model.predict([self.x_train_devices, self.x_train_temporal, self.x_train_behav, self.x_train_bias])
            cur_states = pred[:,-1,0]
            ws = pred[:,-1,1]
            t_js = self.train_starttimes[:,-1]
            t_true = self.train_nextstarttime[:,-1]

        t_pred = pred_next_starttime_vec(cur_states, ws, t_js)

        return np.sqrt(mean_squared_error(t_true/self.time_scale, t_pred/self.time_scale))


    def pred_next_starttime(self, cur_state, w, t_j):
        ts = np.arange(t_j, 800*self.time_scale, self.time_scale)
        delta_ts = ts - t_j
        samples = ts * self._pred_next_starttime(delta_ts, cur_state, w)

        return trapz(samples, ts)


    def _pred_next_starttime(self, delta_t, cur_state, w):
        # w_t = self.w_scale
        w_t = w
        # w_t = 1
        return np.exp(-(-cur_state - w_t*delta_t \
               - (1/w)*np.exp(cur_state) \
               + (1/w)*np.exp(cur_state + w_t*(delta_t))))


