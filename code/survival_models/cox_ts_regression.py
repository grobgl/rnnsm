import pickle
from churn_data import ChurnData, getChurnScores
from survival_model import SurvivalModel
import pandas as pd
import numpy as np
from scipy.integrate import trapz
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index, _get_index
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold

from multiprocessing import Pool
from functools import partial



class CoxTsChurnModel(SurvivalModel):
    RESULT_PATH = '../../results/churn/cox_ts_regression/'

    def __init__(self, penalizer=0):
        super().__init__()
        self.cf = CoxPHFitter(penalizer=penalizer)

    def predict_expectation(self, indices=None, dataset='train'):
        df = self.data.train_df
        df_unscaled = self.data.train_unscaled_df

        if dataset=='test':
            df = self.data.test_df
            df_unscaled = self.data.test_unscaled_df

        if indices is None:
            indices = self.data.split_val_ind

        x_df = df.iloc[indices]
        x_df_unscaled = df_unscaled.iloc[indices]
        recency = x_df_unscaled.recency#.values.reshape(-1)

        index = _get_index(x_df)
        v = self.cf.predict_survival_function(x_df)[index]

        # set all values in predicted survival function at position lower than recency to 0
        for i in v.columns:
            v[i][v.index < recency[i]] = 0

        targets = pd.DataFrame(recency + 1/recency * trapz(v.values.T, v.index), index=index)
        # targets = pd.DataFrame(recency + trapz(v.values.T, v.index), index=index)
        # targets = pd.DataFrame(1/recency * trapz(v.values.T, v.index), index=index)

        pred = self.reverseTransformTargets(targets)

        return pred.values.reshape(-1)
