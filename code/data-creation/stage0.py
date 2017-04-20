import sys
sys.path.insert(0, '../utils')
from dataPiping import makeunixtime

import pandas as pd
import numpy as np
from multiprocessing import Pool

def createStage0():
    df = loadData()
    df = parallelizeDataframe(df, convertTimeCols)
    return df

# read raw data
def loadData():
    """Loads data into df
    """
    df = pd.read_pickle('../../data/sessionDF.pkl')

    return df

def convertTimeCols(df):
    # convert session start time to pandas time index
    df.startTime = pd.DatetimeIndex(df.startTime.apply(makeunixtime) * 1000000000)
    df.startUserTime = pd.DatetimeIndex(df.startUserTime.apply(makeunixtime) * 1000000000)
    df.sessionLength = pd.TimedeltaIndex(df.sessionLength * 1000000000)

    return df

def parallelizeDataframe(df, func, num_cores=8):
    df['partition'] = ((df.customerId // 100) % 16) // 2

    pt = df.groupby('partition')
    df_split = [pt.get_group(x) for x in pt.groups]

    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))

    pool.close()
    pool.join()
    return df

