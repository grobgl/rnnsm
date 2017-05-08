from survival_model import *
from cox_regression import *
from cox_regression_sqrt import *


class CoxChurnModel_short(CoxChurnModel):
    RESULT_PATH = '../../results/churn/cox_regression_short/'

    def __init__(self, penalizer=2100, include_recency=False):
        self.data = ChurnData(predict='deltaNextHours', dataset='../../data/churn/churn_short.pkl')
        self.include_recency = include_recency
        self.cf = CoxPHFitter(penalizer=penalizer)

    def fit(self, dataset, indices=None):
        dataset = dataset.copy()
        del dataset['churnedFull']
        del dataset['deltaNextHoursFull']

        return super().fit(dataset, indices)

    def getScores(self, indices=None, dataset='train'):
        df = self.data.train_df
        df_unscaled = self.data.train_unscaled_df

        if dataset=='test':
            df = self.data.test_df
            df_unscaled = self.data.test_unscaled_df

        if indices is None:
            indices = self.data.split_val_ind

        df = df.iloc[indices]
        df_unscaled = df_unscaled.iloc[indices]

        pred_durations = self.predict_expectation(df, df_unscaled)
        pred_churn = self.predict_churn(pred_durations, df_unscaled)

        pred_durations[pred_durations==np.inf] = hours_year
        pred_durations[pred_durations>hours_year] = hours_year
        pred_durations[pred_durations==np.nan] = hours_year

        churn_err = getChurnScores(~df.observed, pred_churn, pred_durations)

        cens = ~df.observed.astype('bool') & ~df.churnedFull.astype('bool')
        uncens = df.observed.astype('bool')
        rmse_days_uncens = np.sqrt(mean_squared_error(df.deltaNextHours[uncens], pred_durations[uncens])) / 24
        rmse_days_cens = np.sqrt(mean_squared_error(df.deltaNextHoursFull[cens], pred_durations[cens])) / 24

        return {'churn_acc': churn_err['accuracy'],
                'churn_auc': churn_err['auc'],
                'churn_prec': churn_err['precision'][1],
                'churn_recall': churn_err['recall'][1],
                'churn_f1': churn_err['f1'][1],
                'rmse_days_uncens': rmse_days_uncens,
                'rmse_days_cens': rmse_days_cens,
                'concordance': concordance_index(df.deltaNextHours, pred_durations, df.observed)}


class CoxSqrtChurnModel_short(CoxChurnModel_short):
    RESULT_PATH = '../../results/churn/cox_regression_sqrt_short/'

    def transformTargets(self, targets):
        return np.sqrt(targets)

    def reverseTransformTargets(self, targets):
        return targets**2

