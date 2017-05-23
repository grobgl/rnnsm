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

import sys
sys.path.insert(0, '../utils')
from plot_format import *
from seaborn import apionly as sns

seed = 42
np.random.seed(seed)

class Rmtpp:

    w_scale = 1.
    time_scale = 0.1

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
        # merge_inputs = concatenate([temporal_masking, behav_masking])

        lstm_output = LSTM(lstm_neurons, return_sequences=self.predict_sequence, recurrent_activation='relu')(merge_inputs)

        predictions = Dense(1, activation='linear')(lstm_output)

        # bias_input = Input(shape=(1,), name='bias_input')
        # bias_w_t = Dense(1, activation = 'linear', name='bias_w_t', kernel_initializer=Zeros(), bias_initializer=Constant(.001), kernel_constraint=non_neg(), bias_constraint=non_neg())(bias_input)
        # bias_input = Masking(mask_value=0.)(bias_input)
        # bias_w = Dense(1, activation = 'linear', kernel_constraint=non_neg(), bias_constraint=non_neg())(bias_input)
        # bias_w = RepeatVector(len_seq)(bias_w)

        # output = concatenate([predictions, bias_w])

        # model = Model(inputs=[device_input, temporal_input, behav_input, bias_input], outputs=output)
        model = Model(inputs=[device_input, temporal_input, behav_input], outputs=predictions)

        loss = self.neg_log_likelihood_seq if self.predict_sequence else self.neg_log_likelihood
        model.compile(loss=loss, optimizer=RMSprop(lr=lr))

        self.model = model
        return model

        # maskingLayer = Masking(mask_value=0., input_shape=input_shape)(inputs)

    def fit_model(self, initial_epoch=0):
        log_file = '{:02d}_{}_lr{}_inp{}'.format(self.run, self.name, self.lr,self.x_train.shape[2])
        # self.model.fit([self.x_train_devices, self.x_train_temporal, self.x_train_behav, self.x_train_bias], self.y_train, batch_size=1000, epochs=5000, validation_split=0.2, verbose=0, initial_epoch=initial_epoch
        self.model.fit([self.x_train_devices, self.x_train_temporal, self.x_train_behav], self.y_train, batch_size=1000, epochs=5000, validation_split=0.2, verbose=0, initial_epoch=initial_epoch
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

        w = self.w_scale
        w_t = w
        cur_state = K.batch_flatten(output)
        delta_t = K.batch_flatten(targets)

        res = -cur_state - w_t*delta_t \
               - (1/w)*K.exp(cur_state) \
               + (1/w)*K.exp(cur_state + w_t*(delta_t))

        # return res
        return res

    def neg_log_likelihood_seq(self, targets, output):
        """ Loss function for RMTPP model

        :targets: vector of: [t_(j+1) - t_j, mask]
        :output: rnn output = v_t * h_j + b_t
        """

        # w_t = self.w_scale
        mask = K.batch_flatten(targets[:,:,1])
        # w = K.batch_flatten(output[:,:,1])
        w = self.w_scale
        w_t = w
        # cur_state = K.batch_flatten(output[:,:,0])
        cur_state = K.batch_flatten(output)
        delta_t = K.batch_flatten(targets[:,:,0])

        res = -cur_state - w_t*delta_t \
               - (1/w)*K.exp(cur_state) \
               + (1/w)*K.exp(cur_state + w_t*(delta_t))

        # return res
        return res*mask

    def get_predictions(self, dataset='test', include_recency=False):
        if include_recency:
            pred_next_starttime_vec = np.vectorize(self.pred_next_starttime_rec)
        else:
            pred_next_starttime_vec = np.vectorize(self.pred_next_starttime)

        if dataset=='test':
            pred = self.model.predict([self.x_test_devices, self.x_test_temporal, self.x_test_behav])
            cur_states = pred[:,-1].ravel()
            t_js = self.test_starttimes[:,-1]
            if self.predict_sequence:
                t_true = self.y_test[:,-1,0]
            else:
                t_true = self.y_test
        else:
            pred = self.model.predict([self.x_train_devices, self.x_train_temporal, self.x_train_behav])
            cur_states = pred[:,-1].ravel()
            t_js = self.train_starttimes[:,-1]
            if self.predict_sequence:
                t_true = self.y_train[:,-1,0]
            else:
                t_true = self.y_train

        t_pred = pred_next_starttime_vec(cur_states, t_js)

        return t_pred/self.time_scale, t_true/self.time_scale


    def get_rmse_days(self, dataset='test', include_recency=False):
        t_pred, t_true = self.get_predictions(dataset, include_recency)

        return np.sqrt(mean_squared_error(t_true, t_pred))


    def pred_next_starttime(self, cur_state, t_j):
        ts = np.arange(t_j, 1000*self.time_scale, self.time_scale)
        delta_ts = ts - t_j
        samples = self._pt(delta_ts, cur_state)
        # samples = delta_ts * self._pt(delta_ts, cur_state)

        return trapz(samples, ts)


    def pred_next_starttime_rec(self, cur_state, t_j):
        absence_time = 365*self.time_scale - t_j
        s_ts = self._pt(absence_time, cur_state)

        ts = np.arange(t_j, 1000*self.time_scale, self.time_scale)
        delta_ts = ts - t_j
        samples = self._pt(delta_ts, cur_state)
        # return samples
        # samples = delta_ts * self._pt(delta_ts, cur_state)

        return (1/s_ts) * trapz(samples[ts>(365*self.time_scale)], ts[ts>(365*self.time_scale)]) + trapz(samples[ts<=(365*self.time_scale)], ts[ts<=(365*self.time_scale)])


    def _pt(self, delta_t, cur_state):
        w_t = self.w_scale
        w = self.w_scale
        # w_t = w
        # w_t = 1
        return delta_t * np.exp(-(-cur_state - w_t*delta_t \
                                   - (1/w)*np.exp(cur_state) \
                                   + (1/w)*np.exp(cur_state + w_t*(delta_t))))



predPeriod = {
    'start': pd.Timestamp('2016-02-01'),
    'end': pd.Timestamp('2016-06-01')
}
obsPeriod = {
    'start': pd.Timestamp('2015-02-01'),
    'end': pd.Timestamp('2016-02-01')
}
predPeriodHours = (predPeriod['end'] - predPeriod['start']) / np.timedelta64(1, 'h')
hours_year = np.timedelta64(pd.datetime(2017,2,1) - pd.datetime(2016,2,1)) / np.timedelta64(1,'h')

def showResidPlot_short_date(y_true, y_pred, true_ret_time_days, width=1, height=None):
    df = pd.DataFrame()
    df['predicted (days)'] = y_pred
    df['actual (days)'] = y_true
    df['daysInObs'] = true_ret_time_days
    df['date'] = df['daysInObs'] * np.timedelta64(24,'h') + obsPeriod['start']
    df['residual (days)'] = df['predicted (days)'] - df['actual (days)']

    grid = sns.JointGrid('daysInObs', 'residual (days)', data=df, size=figsize(.5,.5)[0], xlim=(0,3000), ylim=(-110,110))
    grid = grid.plot_marginals(sns.distplot, kde=False, color='k')#, shade=True)
    grid = grid.plot_joint(plt.scatter, alpha=.1, s=6, lw=0)
    grid.ax_joint.clear()

    retUnc = grid.ax_joint.scatter(df['daysInObs'], df['residual (days)'], alpha=.1, s=6, lw=0, color='C0', label='Ret. user (uncens.)')

    xDates = [pd.datetime(2016,i,1) for i in [2,4,6]]
    xDatesHours = [(d - obsPeriod['start']).to_timedelta64()/np.timedelta64(24,'h') for d in xDates]
    xDatesStr = [d.strftime('%Y-%m') for d in xDates]
    grid.ax_joint.set_xticks(xDatesHours)
    grid.ax_joint.set_xticklabels(xDatesStr)
    grid.ax_joint.set_xlabel('actual return date')
    grid.ax_joint.set_ylabel('residual (days)')

    grid.ax_joint.set_ylim((-110,110))
    plt.show()


def showResidPlot_short_days(y_true, y_pred, true_ret_time_days, width=1, height=None):
    df = pd.DataFrame()
    df['predicted (days)'] = y_pred
    df['actual (days)'] = y_true
    df['daysInObs'] = true_ret_time_days
    df['date'] = df['daysInObs'] * np.timedelta64(24,'h') + obsPeriod['start']
    df['residual (days)'] = df['predicted (days)'] - df['actual (days)']

    grid = sns.JointGrid('actual (days)', 'residual (days)', data=df, size=figsize(.5,.5)[0], xlim=(0,3000), ylim=(-110,110))
    grid = grid.plot_marginals(sns.distplot, kde=False, color='k')#, shade=True)
    grid = grid.plot_joint(plt.scatter, alpha=.1, s=6, lw=0)
    grid.ax_joint.clear()

    retUnc = grid.ax_joint.scatter(df['actual (days)'], df['residual (days)'], alpha=.1, s=6, lw=0, color='C0', label='Ret. user (uncens.)')

    grid.ax_joint.set_xlabel('actual return time (days)')
    grid.ax_joint.set_ylabel('residual (days)')
    grid.ax_joint.set_ylim((-110,110))

    plt.show()


