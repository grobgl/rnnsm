import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class ChurnData:
    def __init__(self, features=None, dataset='churn'):
        self.df = pd.read_pickle('../../data/churn/churn.pkl')
        self.dataset = dataset
        self.pred_col = 'churned'

        if dataset=='cox':
            self.df = pd.read_pickle('../../data/churn/cox_churn.pkl')
            self.pred_col = 'deltaNextHours'

        if features is None:
            features = list(set(self.df.columns) - set(['customerId','churned','deltaNextHours','observed']))
        self.setFeatures(features)


    def setFeatures(self, features, intercept=False):
        self.features = np.array(features)

        if self.dataset == 'churn':
            self.y, self.X = dmatrices(
                    'churned~' + '+'.join(self.features) + ('' if intercept else '-1'),
                    self.df)
        elif self.dataset=='cox':
            self.y, self.X = dmatrices('deltaNextHours~' + '+'.join(self.features) + ('+observed-1'), self.df)

        # workaround for missing pickling support in patsy
        self.X = np.array(self.X.tolist())
        self.y = np.array(self.y.tolist())

        if intercept:
            np.insert(self.features, 0, 'intercept')

        self.X_train0, self.X_test0, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42)

        # scale values
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train0)
        self.X_test = self.scaler.transform(self.X_test0)

        if self.dataset=='cox':
            # un-scale observed column
            self.X_train.T[-1] = self.X_train0.T[-1].astype('bool')
            self.X_test.T[-1] = self.X_test0.T[-1].astype('bool')

        # further split training set into train and validation sets
        self.X_split_train, self.X_split_val, self.y_split_train, self.y_split_val = train_test_split(
                self.X_train, self.y_train, test_size=0.2, random_state=42)

        if intercept:
            self.X_train.T[0] = 1
            self.X_test.T[0] = 1

        self.y_train = self.y_train.reshape(-1)
        self.y_test = self.y_test.reshape(-1)
        self.y_split_train = self.y_split_train.reshape(-1)
        self.y_split_val = self.y_split_val.reshape(-1)

        self.train = {'X': self.X_train, 'y': self.y_train}
        self.train_df = self._asDf(**self.train)
        self.test = {'X': self.X_test, 'y': self.y_test}
        self.test_df = self._asDf(**self.test)
        self.split_train = {'X': self.X_split_train, 'y': self.y_split_train}
        self.split_train_df = self._asDf(**self.split_train)
        self.split_val = {'X': self.X_split_val, 'y': self.y_split_val}
        self.split_val_df = self._asDf(**self.split_val)


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


