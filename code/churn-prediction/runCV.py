import pickle
import os
import sys
sys.path.insert(0, '../utils')
# from plot_format import *

# from logistic_regression import *
from random_forest import *

def run():
    # runL1GridSearch()
    # runL2GridSearch()
    # runFeatureElimination(includeFeat='all')
    # runFeatureElimination(includeFeat='avg')
    # runFeatureElimination(includeFeat='wght')
    runGridSearch()

