import pickle
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, '../utils')
from churn_data import *
from plot_format import *
from seaborn import apionly as sns

predPeriod = {
    'start': pd.Timestamp('2016-02-01'),
    'end': pd.Timestamp('2016-06-01')
}


def plotCensorship(width=1, height=None):
    data = ChurnData(predict='deltaNextHours')
    users = data.train_unscaled_df.head(17).tail(10).copy()
    users['start'] = -users['recency']
    users['end'] = users['start'] + users['deltaNextHours']

    fig, ax = newfig(width, height)

    pred = ax.axvspan(0, users['end'].max(), facecolor='C2', alpha=0.2, label='Prediction window')

    lines = {}

    for i in range(len(users)):
        color = 'C3'
        label = 'Churned user (censored)'
        observed = users.iloc[i].observed
        if observed:
            color = 'C0'
            label = 'Returning user'

        lines['Session'] = ax.scatter([users.iloc[i].start], [i], marker='D', color='black', alpha=.3, label='Session')
        lines['Session'].remove()
        lines[label] = ax.plot([users.iloc[i].start, users.iloc[i].end], [i,i], 'D--', color=color, label=label)[0]

        if not observed:
            lines[label].remove()
            ax.plot([users.iloc[i].start, users.iloc[i].end], [i,i], '--', color=color, label=label)
            ax.scatter([users.iloc[i].start], [i], marker='D', color=color, label=label)


    xDates = [pd.datetime(2015,i,1) for i in range(10,13,2)] + [pd.datetime(2016,i,1) for i in range(2,7,2)]
    xDatesHours = [(d - predPeriod['start']).to_timedelta64()/np.timedelta64(1,'h') for d in xDates]
    xDatesStr = [d.strftime('%Y-%m') for d in xDates]
    ax.set_xticks(xDatesHours)
    ax.set_xticklabels(xDatesStr)

    ax.set_yticks([])
    ax.legend(handles=[pred] + list(lines.values()))

    fig.tight_layout()
    fig.show()

