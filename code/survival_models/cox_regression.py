import pickle
from churn_data import ChurnData, getChurnScores
from survival_model import SurvivalModel
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold

from multiprocessing import Pool
from functools import partial
import sys
sys.path.insert(0, '../utils')
from plot_format import *
# import seaborn as sns
from seaborn import apionly as sns



class CoxChurnModel(SurvivalModel):
    RESULT_PATH = '../../results/churn/cox_regression/'

    def __init__(self, penalizer=0):
        super().__init__()
        self.cf = CoxPHFitter(penalizer=penalizer)

