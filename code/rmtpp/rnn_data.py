import numpy as np
import pandas as pd
import patsy as pt
import os.path
import pickle
from functools import partial
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
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

    def get_xy(self, include_churned=True, min_n_sessions=10, n_sessions=10):
        x_train, x_test, y_train, y_test = self.x_train, self.x_test, self.y_train, self.y_test
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

        x_train = _pad_x(x_train, n_sessions)
        x_test = _pad_x(x_test, n_sessions)

        return x_train, x_test, y_train.values, y_test.values

    def _initialise(self):
        df_0 = self.df_0 = pd.read_pickle(self.PATH+'rnn_df.pkl')
        churned = self.churned = df_0.groupby('customerId').last().churned
        self.churned_cust = churned.index[churned].values
        self.num_sessions = self.df_0.groupby('customerId').customerId.count()

        # train/test split, stratify by churn
        train_i, test_i = self.train_i, self.test_i = next(StratifiedShuffleSplit(test_size=.2, random_state=42).split(churned, churned.values))
        train_df_unscaled = self.train_df_unscaled = df_0[df_0.customerId.isin(churned.index[train_i])]
        test_df_unscaled = self.test_df_unscaled = df_0[df_0.customerId.isin(churned.index[test_i])]

        # scaling
        features_0 = self.features_0 = list(set(df_0.columns) - set(['customerId','startUserTime','deltaNextHours', 'churned']))
        train_df_scaled = self.train_df_scaled = train_df_unscaled.copy()
        test_df_scaled = test_df_unscaled.copy()
        scaler = self.scaler = StandardScaler()
        train_df_scaled[features_0] = scaler.fit_transform(train_df_unscaled[features_0])
        test_df_scaled[features_0] = scaler.transform(test_df_unscaled[features_0])
        self.train_df_scaled = train_df_scaled
        self.test_df_scaled = test_df_scaled

        # convert to array
        x_train, y_train = self.x_train, self.y_train = _df_to_xy_array(train_df_scaled, features_0)
        x_test, y_test = self.x_test, self.y_test = _df_to_xy_array(test_df_scaled, features_0)

        # store customer idsk
        train_cust = self.train_cust = y_train.index.values
        test_cust = self.test_cust = y_test.index.values

        with open(self.PATH+'rnn_data.pkl', 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

def _pad_x(x, n):
    return np.array(list(map(lambda _x: np.pad(_x[-n:], ((max(n-len(_x),0),0),(0,0)), 'constant'), x.values)))

def _df_to_xy_array(df, features):
    grouped = df.groupby('customerId')
    x = grouped.apply(lambda g: g[features].as_matrix())
    y = grouped.deltaNextHours.last()

    return x, y

