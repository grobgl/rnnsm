from cox_regression import *
from cox_regression_log import *
from cox_regression_sqrt import *
from aalen_additive import *
from aalen_additive_log import *
from aalen_additive_sqrt import *
from survival_model import *


def run():
    runParameterSearch(CoxChurnModel, include_recency=True, error='concordance', maximise=True)

    # runParameterSearch(CoxChurnModel, include_recency=True, error='rmse_days', maximise=False)
    # runParameterSearch(CoxChurnModel, include_recency=False, error='rmse_days', maximise=False)
    # runParameterSearch(CoxSqrtChurnModel, include_recency=True, error='rmse_days', maximise=False)
    # runParameterSearch(CoxSqrtChurnModel, include_recency=False, error='rmse_days', maximise=False)
    # runParameterSearch(CoxLogChurnModel, include_recency=False, error='rmse_days', maximise=False)
    # runParameterSearch(CoxLogChurnModel, include_recency=True, error='rmse_days', maximise=False)
    # runParameterSearch(CoxChurnModel, include_recency=False, error='churn_auc', maximise=True)
    # runParameterSearch(CoxChurnModel, include_recency=True, error='churn_auc', maximise=True)
    # runParameterSearch(CoxSqrtChurnModel, include_recency=True, error='churn_auc', maximise=True)
    # runParameterSearch(CoxSqrtChurnModel, include_recency=False, error='churn_auc', maximise=True)
    # runParameterSearch(CoxLogChurnModel, include_recency=True, error='churn_auc', maximise=True)

    # runParameterSearch(CoxChurnModel, include_recency=True, error='rmse_days', maximise=False)
    # runParameterSearch(CoxSqrtChurnModel, include_recency=True, error='rmse_days', maximise=False)
    # runParameterSearch(CoxLogChurnModel, include_recency=True, error='rmse_days', maximise=False)

    # runParameterSearch(CoxChurnModel, include_recency=True, error='churn_acc', maximise=True)
    # runParameterSearch(CoxSqrtChurnModel, include_recency=True, error='churn_acc', maximise=True)
    # runParameterSearch(CoxLogChurnModel, include_recency=True, error='churn_acc', maximise=True)

    # runParameterSearch(CoxChurnModel, include_recency=True, error='concordance', maximise=True)
    # runParameterSearch(CoxSqrtChurnModel, include_recency=True, error='concordance', maximise=True)
    runParameterSearch(CoxLogChurnModel, include_recency=True, error='concordance', maximise=True)

    # runParameterSearch(CoxChurnModel, include_recency=False, error='churn_acc', maximise=True)
    # runParameterSearch(CoxLogChurnModel, include_recency=True)
    # runParameterSearch(CoxSqrtChurnModel, include_recency=True)
    # runParameterSearch(AalenChurnModel, include_recency=True)
    # runParameterSearch(AalenLogChurnModel)
    # runParameterSearch(AalenSqrtChurnModel)

    return
