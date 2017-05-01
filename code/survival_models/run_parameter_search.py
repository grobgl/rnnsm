from cox_regression import *
from cox_regression_log import *
from cox_regression_sqrt import *
from aalen_additive import *
from aalen_additive_log import *
from aalen_additive_sqrt import *
from survival_model import *


def run():
    runParameterSearch(CoxChurnModel, include_recency=True)
    runParameterSearch(CoxLogChurnModel, include_recency=True)
    runParameterSearch(CoxSqrtChurnModel, include_recency=True)
    # runParameterSearch(AalenChurnModel)
    # runParameterSearch(AalenLogChurnModel)
    # runParameterSearch(AalenSqrtChurnModel)

    return
