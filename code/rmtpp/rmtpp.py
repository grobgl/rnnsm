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
from keras import regularizers
from keras import backend as K

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, roc_auc_score
from lifelines.utils import concordance_index
from sklearn.gaussian_process.kernels import Matern
from bayes_opt import BayesianOptimization

from rmtpp_data import *

import sys
sys.path.insert(0, '../utils')
from plot_format import *
from seaborn import apionly as sns

seed = 42
np.random.seed(seed)

class Rmtpp:

    w_scale = .5
    time_scale = .1

    def __init__(self, name, run, hidden_neurons=32, n_sessions=100):
        self.predict_sequence = False
        self.hidden_neurons = hidden_neurons
        self.n_sessions = n_sessions
        self.data = RmtppData.instance()
        self.set_x_y()
        self.set_model()
        self.name = name
        self.run = run
        self.best_model_cp_file = '../../results/rmtpp/{:02d}_{}.hdf5'.format(run, name)
        self.best_model_cp = ModelCheckpoint(self.best_model_cp_file, monitor="val_loss",
                                             save_best_only=True, save_weights_only=False)

    def set_x_y(self, include_churned=False, preset='deltaNextDays'):
        self.x_train, \
        self.x_test, \
        self.x_train_unscaled, \
        self.x_test_unscaled, \
        self.y_train, \
        self.y_test, \
        self.features, \
        self.targets = self.data.get_xy(min_n_sessions=0, n_sessions=self.n_sessions, preset=preset, target_sequences=self.predict_sequence, encode_devices=True)

        if self.predict_sequence:
            self.y_train_churned = self.y_train[:,-1,self.targets.index('churned')].astype('bool')
            self.y_test_churned = self.y_test[:,-1,self.targets.index('churned')].astype('bool')
        else:
            self.y_train_churned = self.y_train[:,self.targets.index('churned')].astype('bool')
            self.y_test_churned = self.y_test[:,self.targets.index('churned')].astype('bool')

        train_train_i, train_val_i = self.train_i, self.test_i = next(StratifiedShuffleSplit(test_size=.2, random_state=42).split(self.x_train, self.y_train_churned))

        temporal_features = ['dayOfMonth', 'dayOfWeek', 'hourOfDay', 'deltaPrevDays', 'startUserTimeDays', 'sessionLengthSec']
        self.device_index = self.features.index('device')
        self.temporal_indices = list(map(self.features.index, temporal_features))
        self.behav_indices = list(map(self.features.index, set(self.features) - set(temporal_features + ['device'])))
        self.startTimeDaysIndex = self.features.index('startUserTimeDays')

        # self.x_train = self.x_train[:,:,used_feature_indices]
        # self.x_test = self.x_test[:,:,used_feature_indices]
        self.x_train_train = self.x_train[train_train_i]
        self.x_train_val = self.x_train[train_val_i]
        self.x_train_train_unscaled = self.x_train_unscaled[train_train_i]
        self.x_train_val_unscaled = self.x_train_unscaled[train_val_i]

        self.y_train_train = self.y_train.T[[1,2]].T.astype('float32')[train_train_i]
        self.y_train_val = self.y_train.T[[1,2]].T.astype('float32')[train_val_i]
        self.y_train_train[:,0] *= self.time_scale
        self.y_train_val[:,0] *= self.time_scale

        self.y_train_train_churned = self.y_train_churned[train_train_i]
        self.y_train_val_churned = self.y_train_churned[train_val_i]

        self.x_train_train_ret = self.x_train_train[~self.y_train_train_churned]
        self.x_train_val_ret = self.x_train_val[~self.y_train_val_churned]
        self.y_train_train_ret = self.y_train_train[~self.y_train_train_churned]
        self.y_train_val_ret = self.y_train_val[~self.y_train_val_churned]

        self.y_train = self.y_train.T[[1,2]].T.astype('float32')
        self.y_test = self.y_test.T[[1,2]].T.astype('float32')
        self.y_train[:,0] *= self.time_scale
        self.y_test[:,0] *= self.time_scale

        if self.predict_sequence:
            self.y_train_train = self.y_train_train.reshape(self.y_train_train.shape+(1,))
            self.y_train_val = self.y_train_val.reshape(self.y_train_val.shape+(1,))
            self.y_train_train_ret = self.y_train_train_ret.reshape(self.y_train_train_ret.shape+(1,))
            self.y_train_val_ret = self.y_train_val_ret.reshape(self.y_train_val_ret.shape+(1,))
            self.y_train = self.y_train.reshape(self.y_train.shape+(1,))
            self.y_test = self.y_test.reshape(self.y_test.shape+(1,))


    def load_best_weights(self):
        self.model.load_weights(self.best_model_cp_file)


    def set_model(self, lr=.01):
        self.lr = lr
        len_seq = self.x_train.shape[1]
        n_devices = np.unique(self.x_train[:,:,self.device_index]).shape[0]
        len_temporal = len(self.temporal_indices)
        len_behav = len(self.behav_indices)

        lstm_neurons = self.hidden_neurons

        # use embedding layer for devices
        device_input = Input(shape=(len_seq,), dtype='int32', name='device_input')
        device_embedding = Embedding(output_dim=4, input_dim=n_devices,
                                     input_length=len_seq, mask_zero=True)(device_input)

        # inputs for temporal and behavioural data
        temporal_input = Input(shape=(len_seq, len_temporal), name='temporal_input')
        # behav_input = Input(shape=(len_seq, len_behav), name='behav_input')

        temporal_masking = Masking(mask_value=0.)(temporal_input)
        # behav_masking = Masking(mask_value=0.)(behav_input)

        # merge_inputs = concatenate([device_embedding, temporal_masking, behav_masking])
        merge_inputs = concatenate([device_embedding, temporal_masking])

        # lstm_output = LSTM(lstm_neurons, return_sequences=self.predict_sequence, recurrent_activation='relu')(merge_inputs)
        lstm_output = LSTM(lstm_neurons, activation='relu', return_sequences=self.predict_sequence, kernel_regularizer=regularizers.l2(0.02), activity_regularizer=regularizers.l2(0.02))(merge_inputs)

        predictions = Dense(1, activation='linear')(lstm_output)

        # bias_input = Input(shape=(1,), name='bias_input')
        # bias_w_t = Dense(1, activation = 'linear', name='bias_w_t', kernel_initializer=Zeros(), bias_initializer=Constant(.001), kernel_constraint=non_neg(), bias_constraint=non_neg())(bias_input)
        # bias_input = Masking(mask_value=0.)(bias_input)
        # bias_w = Dense(1, activation = 'linear', kernel_constraint=non_neg(), bias_constraint=non_neg())(bias_input)
        # bias_w = RepeatVector(len_seq)(bias_w)

        # output = concatenate([predictions, bias_w])

        # model = Model(inputs=[device_input, temporal_input, behav_input, bias_input], outputs=output)
        # model = Model(inputs=[device_input, temporal_input, behav_input], outputs=predictions)
        model = Model(inputs=[device_input, temporal_input], outputs=predictions)

        # loss = self.neg_log_likelihood_seq if self.predict_sequence else self.neg_log_likelihood
        loss = self.neg_log_likelihood_cens
        model.compile(loss=loss, optimizer=RMSprop(lr=lr))

        self.model = model
        return model

        # maskingLayer = Masking(mask_value=0., input_shape=input_shape)(inputs)

    def fit_model(self, initial_epoch=0):
        log_file = '{:02d}_{}_lr{}_inp{}'.format(self.run, self.name, self.lr,self.x_train.shape[2])

        # self.model.fit([self.x_train_train_ret[:,:,self.device_index], self.x_train_train_ret[:,:,self.temporal_indices], self.x_train_train_ret[:,:,self.behav_indices]], self.y_train_train_ret, batch_size=1000, epochs=5000, validation_split=0.2, verbose=0, initial_epoch=initial_epoch
        self.model.fit([self.x_train_train[:,:,self.device_index], self.x_train_train[:,:,self.temporal_indices]], self.y_train_train, batch_size=1000, epochs=5000, validation_split=0.2, verbose=0, initial_epoch=initial_epoch
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


    def neg_log_likelihood_cens(self, targets, output):
        """ Loss function for RMTPP model

        :targets: vector of: [t_(j+1) - t_j, mask]
        :output: rnn output = v_t * h_j + b_t
        """
        ret_mask = K.cast(K.equal(targets[:,1], 0), 'float32')
        delta_t = targets[:,0]
        w = self.w_scale
        w_t = w

        cur_state = K.batch_flatten(output)

        ret_term = -cur_state - w_t*delta_t
        ret_term = ret_mask * ret_term
        common_term = -(1/w)*K.exp(cur_state) + (1/w)*K.exp(cur_state + w_t*(delta_t))

        # res = -cur_state - w_t*delta_t \
        #        - (1/w)*K.exp(cur_state) \
        #        + (1/w)*K.exp(cur_state + w_t*(delta_t))

        return ret_term + common_term
        # return res


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

    def get_predictions(self, dataset='val', include_recency=False):
        if include_recency:
            pred_next_starttime_vec = np.vectorize(self.pred_next_starttime_rec)
        else:
            pred_next_starttime_vec = np.vectorize(self.pred_next_starttime)

        if dataset=='test':
            x = [self.x_test[:,:,self.device_index],
                 self.x_test[:,:,self.temporal_indices]]
                 # self.x_test[:,:,self.behav_indices]]
            t_js = self.x_test_unscaled[:,-1,self.startTimeDaysIndex] * self.time_scale
            y = self.y_test
        else:
            x = [self.x_train_val[:,:,self.device_index],
                 self.x_train_val[:,:,self.temporal_indices]]
                 # self.x_train_val[:,:,self.behav_indices]]
            t_js = self.x_train_val_unscaled[:,-1,self.startTimeDaysIndex] * self.time_scale
            y = self.y_train_val

        pred = self.model.predict(x)
        cur_states = pred[:,-1].ravel()

        if self.predict_sequence:
            t_true = y[:,-1,0]
        else:
            t_true = y

        t_pred = pred_next_starttime_vec(cur_states, t_js)

        return t_pred/self.time_scale, t_true[:,0]/self.time_scale


    # def get_rmse_days(self, dataset='val', include_recency=False):
    #     t_pred, t_true = self.get_predictions(dataset, include_recency)

    #     return np.sqrt(mean_squared_error(t_true, t_pred))


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

    def get_scores(self, dataset='val', include_recency=False):
        if dataset=='val':
            churned = self.y_train_val_churned
        else:
            churned = self.y_test_churned

        pred_0, y_0 = self.get_predictions(dataset, include_recency)

        if self.predict_sequence:
            mask = y_0 != 0
            churned_mask = mask[~churned]
            pred_last = pred_0[:,-1].ravel()
            y_last = y_0[:,-1].ravel()
        else:
            pred_last = pred_0.ravel()
            y_last = y_0.ravel()

        # return pred_0, mask
        rmse_days = np.sqrt(mean_squared_error(pred_last[~churned], y_last[~churned]))

        if self.predict_sequence:
            rmse_days_all = np.sqrt(mean_squared_error(pred_0[~churned][churned_mask].ravel(), y_0[~churned][churned_mask].ravel()))
        else:
            rmse_days_all = 0

        rtd_ind = self.features.index('startUserTimeDays')
        ret_time_days_pred = self.x_train_val_unscaled[:,-1,rtd_ind] + pred_last
        ret_time_days_true = self.x_train_val_unscaled[:,-1,rtd_ind] + y_last

        churned_pred = ret_time_days_pred >= churn_days
        churned_true = ret_time_days_true >= churn_days

        churn_acc = accuracy_score(churned_true, churned_pred)
        churn_recall = recall_score(churned_true, churned_pred)
        churn_auc = roc_auc_score(churned_true, pred_last)

        concordance = concordance_index(y_last, pred_last, ~churned)

        return {'rmse_days': rmse_days,
                'rmse_days_all': rmse_days_all,
                'churn_acc': churn_acc,
                'churn_auc': churn_auc,
                'churn_recall': churn_recall,
                'concordance': concordance}




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
churn_days = (predPeriod['end'] - obsPeriod['start']) / np.timedelta64(24, 'h')


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


