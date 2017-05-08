from cox_regression import *
from cox_regression_log import *
from cox_regression_sqrt import *
from cox_ph_short import *
from aalen_additive import *
from aalen_additive_log import *
from aalen_additive_sqrt import *
from survival_model import *


def run():
    # runGridSearch(CoxChurnModel, include_recency=False)
    # runParameterSearch(CoxSqrtChurnModel, include_recency=True, error='rmse_days', maximise=False)

    # runBayesOpt(CoxChurnModel_short, include_recency=False, error='rmse_days_uncens', maximise=False)

    # runBayesOpt(CoxChurnModel_short, include_recency=True, error='concordance', maximise=True)
    # runBayesOpt(CoxChurnModel_short, include_recency=True, error='churn_f1', maximise=True)
    # runBayesOpt(CoxChurnModel_short, include_recency=True, error='rmse_days_uncens', maximise=False)

    # runBayesOpt(CoxSqrtChurnModel_short, include_recency=True, error='concordance', maximise=True)
    # runBayesOpt(CoxSqrtChurnModel_short, include_recency=True, error='churn_f1', maximise=True)
    runBayesOpt(CoxSqrtChurnModel_short, include_recency=True, error='rmse_days_uncens', maximise=False)

    return
