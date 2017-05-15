import numpy as np
import pandas as pd
import patsy as pt
import os.path
import pickle
from functools import partial
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

obsPeriod = {
    'start': pd.Timestamp('2015-02-01'),
    'end': pd.Timestamp('2016-02-01')
}

actPeriod = {
    'start': pd.Timestamp('2015-10-01'),
    'end': pd.Timestamp('2016-02-01')
}

predPeriod = {
    'start': pd.Timestamp('2016-02-01'),
    'end': pd.Timestamp('2016-06-01')
}


class RnnData:
    PATH = '../../data/rnn/first/'

    def instance():
        if os.path.isfile(RnnData.PATH+'rnn_data.pkl'):
            return pickle.load(open(RnnData.PATH+'rnn_data.pkl', 'rb'))
        else:
            d = RnnData()
            d._initialise()
            return d

    def get_xy(self, include_churned=True, min_n_sessions=10, n_sessions=10, encode_devices=True,  preset='deltaPrevHours'):
        x_train, x_test, y_train, y_test = self.x_train, self.x_test, self.y_train_unscaled, self.y_test_unscaled
        feature_indices = self.presets[preset]['feature_indices']
        features = self.presets[preset]['features']
        target_index = self.presets[preset]['target_index']

        if encode_devices:
            feature_indices = [self.deviceEncIndex] + feature_indices
            features = ['device'] + features
        else:
            feature_indices = self.deviceIndices + feature_indices
            features = self.devices + features

        x_train = x_train.apply(lambda x: x.T[feature_indices].T)
        y_train = y_train[self.presets[preset]['target']]
        x_test = x_test.apply(lambda x: x.T[feature_indices].T)
        y_test = y_test[self.presets[preset]['target']]

        if not include_churned:
            x_train = x_train[~x_train.index.isin(self.churned_cust)]
            y_train = y_train[~y_train.index.isin(self.churned_cust)]
            x_test = x_test[~x_test.index.isin(self.churned_cust)]
            y_test = y_test[~y_test.index.isin(self.churned_cust)]

        if min_n_sessions > 1:
            cust = self.num_sessions[self.num_sessions > min_n_sessions].index
            x_train = x_train[x_train.index.isin(cust)]
            y_train = y_train[y_train.index.isin(cust)]
            x_test = x_test[x_test.index.isin(cust)]
            y_test = y_test[y_test.index.isin(cust)]

        if n_sessions == -1:
            n_sessions = self.num_sessions.max()

        x_train = _pad_x(x_train, n_sessions)
        x_test = _pad_x(x_test, n_sessions)

        return x_train, x_test, y_train.values, y_test.values, features

    def _initialise(self):
        df_0 = self.df_0 = pd.read_pickle(self.PATH+'rnn_df.pkl')
        churned = self.churned = df_0.groupby('customerId').last().churned
        self.churned_cust = churned.index[churned].values
        self.num_sessions = self.df_0.groupby('customerId').customerId.count()

        # encode devices
        self.deviceEncoder = LabelEncoder()
        df_0['device_enc'] = self.deviceEncoder.fit_transform(df_0.device)

        # train/test split, stratify by churn
        train_i, test_i = self.train_i, self.test_i = next(StratifiedShuffleSplit(test_size=.2, random_state=42).split(churned, churned.values))
        train_df_unscaled = self.train_df_unscaled = df_0[df_0.customerId.isin(churned.index[train_i])]
        test_df_unscaled = self.test_df_unscaled = df_0[df_0.customerId.isin(churned.index[test_i])]

        # scaling
        features_numeric = self.features_numeric = sorted(list(set(df_0.columns) - set(['customerId','startUserTime', 'device_enc', 'device'])))
        train_df_scaled = self.train_df_scaled = train_df_unscaled.copy()
        test_df_scaled = test_df_unscaled.copy()
        scaler = self.scaler = StandardScaler()
        train_df_scaled[features_numeric] = scaler.fit_transform(train_df_unscaled[features_numeric])
        test_df_scaled[features_numeric] = scaler.transform(test_df_unscaled[features_numeric])
        self.train_df_scaled = train_df_scaled
        self.test_df_scaled = test_df_scaled

        train_features = self.train_features = sorted(list(set(df_0.columns) - set(['customerId','startUserTime'])))

        # storing feature/target combinations as features for quick access
        # format: predict/mainFeature
        self.deviceEncIndex = train_features.index('device_enc')
        self.devices = [x for x in train_features if x.startswith('device[')]
        self.deviceIndices = list(map(train_features.index, self.devices))
        self.presets = {
            'churn_deltaNextHours': {
                'features': sorted(list(set(self.train_features) - \
                                        set(['deltaNextHours', 'churned', 'device_enc', 'device'] + self.devices))),
                'target': 'churned' },
            'deltaPrevHours': {
                'features': sorted(list(set(self.train_features) - \
                                        set(['deltaNextHours', 'churned', 'device_enc', 'device'] + self.devices))),
                'target': 'deltaPrevHours' },
            'startUserTimeHours': {
                'features': sorted(list(set(self.train_features) - \
                                        set(['deltaNextHours', 'churned', 'device_enc', 'device'] + self.devices))),
                'target': 'startUserTimeHours' }}

        for preset in self.presets:
            self.presets[preset]['feature_indices'] = list(map(self.train_features.index, self.presets[preset]['features']))
            self.presets[preset]['target_index'] = self.train_features.index(self.presets[preset]['target'])

        # convert to array
        self.x_train, self.y_train = _df_to_xy_array(train_df_scaled, train_features)
        self.x_train_unscaled, self.y_train_unscaled = _df_to_xy_array(train_df_unscaled, train_features)
        self.x_test, self.y_test = _df_to_xy_array(test_df_scaled, train_features)
        self.x_test_unscaled, self.y_test_unscaled = _df_to_xy_array(test_df_unscaled, train_features)

        # store customer ids
        train_cust = self.train_cust = self.y_train.index.values
        test_cust = self.test_cust = self.y_test.index.values

        with open(self.PATH+'rnn_data.pkl', 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _pad_x(x, n):
    return np.array(list(map(lambda _x: np.pad(_x[-n:], ((max(n-len(_x),0),0),(0,0)), 'constant'), x.values)))

def _df_to_xy_array(df, features):
    grouped = df.groupby('customerId')
    x = grouped.apply(lambda g: g.head(-1)[features].as_matrix())
    y = grouped.last()

    return x, y

