from cox_regression import *
from cox_regression_log import *
from cox_regression_sqrt import *
from aalen_additive import *
from aalen_additive_log import *
from aalen_additive_sqrt import *
from survival_model import *


def run():
    runParameterSearch(CoxChurnModel)
    # runParameterSearch(CoxLogChurnModel)
    # runParameterSearch(CoxSqrtChurnModel)
    # runParameterSearch(AalenChurnModel)
    # runParameterSearch(AalenLogChurnModel)
    # runParameterSearch(AalenSqrtChurnModel)

    return
