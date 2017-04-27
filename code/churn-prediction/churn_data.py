import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

class ChurnData:
    def __init__(self, features=None, predict='churned'):
        """
        Provides scaled and split aggregated customer session data

        :features: list of features to include
        :predict: value to predict: 'churned' or 'deltaNextHours'
        """

        self.df = pd.read_pickle('../../data/churn/churn.pkl')
        self.pred_col = predict

        if features is None:
            features = list(set(self.df.columns) - set(['customerId','churned','deltaNextHours']))

        self.setFeatures(features)


    def setFeatures(self, features):
        """
        Select subset of features from original dataset. Re-creates all X/y sets accordingly
        """
        self.features = np.array(features)

        # set deltaNextHours as first column in X
        self.y, self.X = dmatrices(
                'churned~' + 'deltaNextHours+' + '+'.join(self.features) + '-1',
                self.df)

        # workaround for missing pickling support in patsy
        self.X = np.array(self.X.tolist())
        self.y = np.array(self.y.tolist())

        # use stratified split
        self.train_ind, self.test_ind  = next(StratifiedShuffleSplit(test_size=.2, random_state=42).split(self.X, self.y))
        self.X_train0, self.X_test0, self.y_train, self.y_test = self.X[self.train_ind], self.X[self.test_ind], self.y[self.train_ind], self.y[self.test_ind]

        # scale values
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train0)
        self.X_test = self.scaler.transform(self.X_test0)

        # un-scale deltaNextHours column
        self.X_train.T[0] = self.X_train0.T[0]
        self.X_test.T[0] = self.X_test0.T[0]

        # further split training set into train and validation sets
        # use stratified split
        self.split_train_ind, self.split_val_ind = next(StratifiedShuffleSplit(test_size=.2, random_state=42).split(self.X_train, self.y_train))
        self.X_split_train, self.X_split_val, self.y_split_train, self.y_split_val = self.X_train[self.split_train_ind], self.X_train[self.split_val_ind], self.y_train[self.split_train_ind], self.y_train[self.split_val_ind]

        # set features for churned / deltaNextHours prediction
        if self.pred_col=='churned':
            # remove deltaNextHours and observed columns
            self.X = self.X.T[1:].T
            self.X_train = self.X_train.T[1:].T
            self.X_test = self.X_test.T[1:].T
            self.X_split_train = self.X_split_train.T[1:].T
            self.X_split_val = self.X_split_val.T[1:].T
        elif self.pred_col=='deltaNextHours':
            # set y as observed column in X
            self.X = np.append(self.X.T, self.y).reshape(-1, len(self.y)).T
            self.X_train = np.append(self.X_train.T, self.y_train).reshape(-1, len(self.y_train)).T
            self.X_test = np.append(self.X_test.T, self.y_test).reshape(-1, len(self.y_test)).T
            self.X_split_train = np.append(self.X_split_train.T, self.y_split_train).reshape(-1, len(self.y_split_train)).T
            self.X_split_val = np.append(self.X_split_val.T, self.y_split_val).reshape(-1, len(self.y_split_val)).T
            # set deltaNextHours as y
            self.y = self.X.T[0].reshape(-1)
            self.y_train = self.X_train.T[0].reshape(-1)
            self.y_test = self.X_test.T[0].reshape(-1)
            self.y_split_train = self.X_split_train.T[0].reshape(-1)
            self.y_split_val = self.X_split_val.T[0].reshape(-1)
            # remove deltaNextHours from X
            self.X = self.X.T[1:].T
            self.X_train = self.X_train.T[1:].T
            self.X_test = self.X_test.T[1:].T
            self.X_split_train = self.X_split_train.T[1:].T
            self.X_split_val = self.X_split_val.T[1:].T

        self.y_train = self.y_train.reshape(-1)
        self.y_test = self.y_test.reshape(-1)
        self.y_split_train = self.y_split_train.reshape(-1)
        self.y_split_val = self.y_split_val.reshape(-1)

        self.train = {'X': self.X_train, 'y': self.y_train}
        self.train_df = self._asDf(**self.train)
        self.train_unscaled_df = self._asDf(X=self.X[self.train_ind], y=self.y[self.train_ind])
        self.test = {'X': self.X_test, 'y': self.y_test}
        self.test_df = self._asDf(**self.test)
        self.test_unscaled_df = self._asDf(X=self.X[self.test_ind], y=self.y[self.test_ind])
        self.split_train = {'X': self.X_split_train, 'y': self.y_split_train}
        self.split_train_df = self._asDf(**self.split_train)
        self.split_train_unscaled_df = self.train_unscaled_df.iloc[self.split_train_ind]
        self.split_val = {'X': self.X_split_val, 'y': self.y_split_val}
        self.split_val_df = self._asDf(**self.split_val)
        self.split_val_unscaled_df = self.train_unscaled_df.iloc[self.split_val_ind]


    def _asDf(self,X,y):
        df = pd.DataFrame(X)
        if self.pred_col=='deltaNextHours':
            df.columns = self.features.tolist() + ['observed']
        else:
            df.columns = self.features

        df[self.pred_col] = y
        return df


    def getScores(self, model, dataset='test', X=None, y=None):
        if dataset=='test':
            X = self.X_test if X is None else X
            y = self.y_test if y is None else y
        elif dataset=='train':
            X = self.X_train if X is None else X
            y = self.y_train if y is None else y
        elif dataset=='split_train':
            X = self.X_split_train if X is None else X
            y = self.y_split_train if y is None else y
        elif dataset=='split_val':
            X = self.X_split_val if X is None else X
            y = self.y_split_val if y is None else y

        predicted = model.predict(X)
        probs = model.predict_proba(X)

        accuracy = metrics.accuracy_score(y, predicted)
        auc = metrics.roc_auc_score(y, probs[:, 1])
        f1 = metrics.f1_score(y, predicted)
        report = metrics.classification_report(y, predicted)
        confusion_matrix =  metrics.confusion_matrix(y, predicted)

        return {'accuracy': accuracy,
                'auc': auc,
                'f1': f1,
                'classification_report': report,
                'confusion_matrix': confusion_matrix}


    def printScores(self, model, dataset='test', X=None, y=None):
        scores = self.getScores(model, dataset, X, y)
        print('Accuracy: {}\n'.format(scores['accuracy']))
        print('AUC: {}\n'.format(scores['auc']))
        print(scores['classification_report'])
        print('Confusion matrix:\n', scores['confusion_matrix'])


