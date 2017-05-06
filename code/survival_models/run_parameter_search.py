from cox_regression import *
from cox_regression_log import *
from cox_regression_sqrt import *
from aalen_additive import *
from aalen_additive_log import *
from aalen_additive_sqrt import *
from survival_model import *


def run():
    runGridSearch(CoxChurnModel, include_recency=False)

    # runParameterSearch(CoxSqrtChurnModel, include_recency=True, error='rmse_days', maximise=False)

    return
