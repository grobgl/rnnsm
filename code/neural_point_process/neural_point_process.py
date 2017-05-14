import numpy as np
import pandas as pd
import patsy as pt
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

obsPeriod = {
    'start': pd.Timestamp('2015-02-01'),
    'end': pd.Timestamp('2016-02-01')
}

actPeriod = {
    'start': pd.Timestamp('2015-10-01'),
    'end': pd.Timestamp('2016-02-01')
}

predPeriod = {
    'start': pd.Timestamp('2016-02-01'),
    'end': pd.Timestamp('2016-06-01')
}

# PATH = '../../data/neural_point_process/'
DATA_PATH = '../../../NeuralPointProcess/data/real/asos/'
TRAIN_PATH = '{}timings-train.txt'.format(DATA_PATH)
TEST_PATH = '{}timings-test.txt'.format(DATA_PATH)
RESULT_PATH = '../../../NeuralPointProcess/results/time-asos-hidden-128-embed-128-bptt-3-bsize-64/test_pred_iter_400000.txt'
# RESULT_PATH = '../../../NeuralPointProcess/results/time-asos-hidden-128-embed-128-bptt-3-bsize-64/test_pred_iter_0.txt'

def get_predictions():
    pred = np.loadtxt(RESULT_PATH)
    pass


def get_train_test_timings():
    train_timings = pickle.load(open(DATA_PATH+'timings-train.pkl', 'rb'))
    test_timings = pickle.load(open(DATA_PATH+'timings-test.pkl', 'rb'))

    def to_df(timings):
        df = pd.DataFrame()
        df['timings'] = timings
        df['deltas'] = df.timings.apply(np.diff)
        df['len_deltas'] = df.deltas.apply(len)
        df['delta_last'] = df.deltas.apply(lambda x: x[-1])
        return df

    train_df, test_df = to_df(train_timings), to_df(test_timings)

    # load predictions for test set
    pred = np.loadtxt(RESULT_PATH)
    # pred *= 10 # adjust for scaling

    # segment predictions
    len_delt = test_df.len_deltas.values
    ind = np.insert(np.cumsum(len_delt), 0, 0)
    pred = [np.array(pred[i:j]) for i, j in zip(ind, np.roll(ind,-1))][:-1]
    test_df['predictions'] = pred

    # prediction errors
    test_df['errors'] = test_df.predictions/10 - test_df.deltas

    return train_df, test_df

def get_rmse_days(test_df, last_only=False):
    errors = np.concatenate(test_df.errors.values)

    if last_only:
        errors = test_df.errors.apply(lambda x: x[-1]).values

    rmse = np.sqrt((errors**2).mean())
    return rmse
    return rmse/24


def store_neural_point_process_ds():
    """
    Creates dataset using same time frames as churn DS, without summarizing sessions. Used for RNN models.
    """
    df = pd.read_pickle('../../data/cleaned/stage1_obs_pred.pkl')

    # only look at sessions in obs period
    df = df[df.startUserTime < obsPeriod['end']]

    # customers with session in act period
    actCust = df[(df.startUserTime >= actPeriod['start']) &
                 (df.startUserTime < actPeriod['end'])].customerId.unique()
    df = df[df.customerId.isin(actCust)]

    # only returning users
    churned = df.groupby('customerId').apply(lambda x: x.tail(1).deltaNext.isnull())
    ret_cust = churned.index[~churned].values
    df = df[df.customerId.isin(ret_cust)]

    # only with min no of sessions
    min_no_sessions = 20
    no_sessions = df.groupby('customerId').customerId.count()
    min_sess_cust = no_sessions.index[no_sessions > min_no_sessions]
    df = df[df.customerId.isin(min_sess_cust)]

    df['delta_next_hours'] = df.deltaNext / np.timedelta64(1,'h')
    df['hours_since_start'] = (df.startUserTime - obsPeriod['start']) / np.timedelta64(1,'h')
    # mapping = ['android', 'desktop', 'ios', 'mobile', 'unknown']
    df['device_enc'] = LabelEncoder().fit_transform(df.device)

    grouped = df.groupby('customerId')
    timings = grouped.apply(lambda g: g[['hours_since_start']].as_matrix().ravel())
    # last_delta_next  = grouped.apply(lambda g: g.delta_next_hours.tail(1))
    # last_timing = timings.apply(lambda x: x[-1]) + last_delta_next
    # timings = np.array(list(map(lambda t: np.append(*t), zip(timings, last_timing))))

    devices = grouped.apply(lambda g: g[['device_enc']].values.ravel()).as_matrix()

    timings_train, timings_test, devices_train, devices_test = train_test_split(timings, devices, test_size=0.2, random_state=42)

    to_txt(timings_train, 'timings-train')
    to_txt(timings_test, 'timings-test')
    to_txt(devices_train, 'devices-train')
    to_txt(devices_test, 'devices-test')


def to_txt(rows, filename):
    with open(DATA_PATH+filename+'.txt', 'w') as outfile:
        for row in rows:
            outfile.write(' '.join(map(str, row))+'\n')
    with open(DATA_PATH+filename+'.pkl', 'wb') as outfile:
        pickle.dump(rows, outfile, protocol=pickle.HIGHEST_PROTOCOL)

