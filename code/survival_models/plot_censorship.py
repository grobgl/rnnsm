import pickle
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, '../utils')
from churn_data import *
from plot_format import *
from seaborn import apionly as sns
import matplotlib.gridspec as gridspec

predPeriod = {
    'start': pd.Timestamp('2016-02-01'),
    'mid': pd.Timestamp('2016-04-01'),
    'end': pd.Timestamp('2016-06-01')
}


def plotCensorship(width=1, height=None):
    data = ChurnData(predict='deltaNextHours')
    users = data.train_unscaled_df.head(17).tail(10).copy()
    users['start'] = -users['recency']
    users['end'] = users['start'] + users['deltaNextHours']

    midHours2 = (pd.Timestamp('2015-10-01') - predPeriod['start']) / np.timedelta64(1,'h')
    endHours = (predPeriod['end'] - predPeriod['start']) / np.timedelta64(1,'h')

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
    ax.set_xlim((midHours2,endHours))

    fig.tight_layout()
    fig.show()

def plotCensorship3(width=1, height=None):
    gap = .1

    data = ChurnData(predict='deltaNextHours')
    users = data.train_unscaled_df.head(17).tail(10).copy()
    users['start'] = -users['recency']
    users['end'] = users['start'] + users['deltaNextHours']

    startHours = (pd.Timestamp('2015-02-01') - predPeriod['start']) / np.timedelta64(1,'h')
    midHours1 = (pd.Timestamp('2015-04-01') - predPeriod['start']) / np.timedelta64(1,'h')
    midHours2 = (pd.Timestamp('2015-09-01') - predPeriod['start']) / np.timedelta64(1,'h')
    midHours = (predPeriod['mid'] - predPeriod['start']) / np.timedelta64(1,'h')
    endHours = (predPeriod['end'] - predPeriod['start']) / np.timedelta64(1,'h')

    gs1 = gridspec.GridSpec(1, 8)
    gs1.update(wspace=gap, hspace=0.2) # set the spacing between axes.

    fig, ax = newfig(width, height, gs1[1:])
    ax.set_xlim(midHours2, endHours)
    ax.yaxis.tick_right()
    ax.spines['left'].set_visible(False)
    ax.yaxis.tick_right()
    ax.tick_params(labelleft='off')

    ax2 = fig.add_subplot(gs1[0], sharey=ax)
    ax2.set_xlim(startHours, midHours1)
    ax2.tick_params(labelright='off')
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.tick_left()

    d = .015 # how big to make the diagonal lines in axes coordinates

    kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
    ax.plot((1-d+gap,1+d+gap), (-d,+d), **kwargs)
    ax.plot((1-d+gap,1+d+gap),(1-d,1+d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((1-d,1+d), (-d,+d), **kwargs)
    ax2.plot((1-d,1+d),(1-d,1+d), **kwargs)


    predNew = ax.axvspan(0, endHours, facecolor='C2', alpha=0.2, label='Prediction window')

    lines = {}

    for i in range(len(users)):
        color = 'C3'
        label = 'Churned user (censored)'
        observed = users.iloc[i].observed
        if observed:
            color = 'C0'
            label = 'Returning user (uncensored)'

        lines['Session'] = ax.scatter([users.iloc[i].start], [i], marker='D', color='black', alpha=.3, label='Session')
        lines['Session'].remove()
        lines[label] = ax.plot([users.iloc[i].start, users.iloc[i].end], [i,i], 'D-', color=color, label=label)[0]
        ax.plot([startHours, users.iloc[i].start], [i,i], ':', color=color, label=label, alpha=.1)[0]
        ax2.plot([startHours, users.iloc[i].start], [i,i], ':', color=color, label=label, alpha=.1)[0]

        if not observed:
            lines[label].remove()
            ax.plot([users.iloc[i].start, users.iloc[i].end], [i,i], '-', color=color, label=label)
            ax.scatter([users.iloc[i].start], [i], marker='D', color=color, label=label)


    xDates2 = [pd.datetime(2015,i,1) for i in range(2,4,2)]
    xDates = [pd.datetime(2015,i,1) for i in range(10,13,2)] + [pd.datetime(2016,i,1) for i in range(2,7,2)]
    xDates2Hours = [(d - predPeriod['start']).to_timedelta64()/np.timedelta64(1,'h') for d in xDates2]
    xDatesHours = [(d - predPeriod['start']).to_timedelta64()/np.timedelta64(1,'h') for d in xDates]
    xDates2Str = [d.strftime('%Y-%m') for d in xDates2]
    xDatesStr = [d.strftime('%Y-%m') for d in xDates]
    ax2.set_xticks(xDates2Hours)
    ax2.set_xticklabels(xDates2Str)
    ax.set_xticks(xDatesHours)
    ax.set_xticklabels(xDatesStr)

    ax.set_ylim([-1,10])
    ax.set_yticks([])
    ax2.set_yticks([])
    handles=[predNew] + list(lines.values())
    leg = ax2.legend(handles=handles, labels=[h.get_label() for h in handles], loc=4, bbox_to_anchor=(3.43, .022))
    leg.get_frame().set_alpha(1)

    plt.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.12)
    fig.show()


def plotCensorship2(width=1, height=None):
    gap = .1

    data = ChurnData(predict='deltaNextHours')
    users = data.train_unscaled_df.head(17).tail(10).copy()
    users['start'] = -users['recency']
    users['end'] = users['start'] + users['deltaNextHours']

    startHours = (pd.Timestamp('2015-02-01') - predPeriod['start']) / np.timedelta64(1,'h')
    midHours1 = (pd.Timestamp('2015-04-01') - predPeriod['start']) / np.timedelta64(1,'h')
    midHours2 = (pd.Timestamp('2015-09-01') - predPeriod['start']) / np.timedelta64(1,'h')
    midHours = (predPeriod['mid'] - predPeriod['start']) / np.timedelta64(1,'h')
    endHours = (predPeriod['end'] - predPeriod['start']) / np.timedelta64(1,'h')

    gs1 = gridspec.GridSpec(1, 8)
    gs1.update(wspace=gap, hspace=0.2) # set the spacing between axes.

    fig, ax = newfig(width, height, gs1[1:])
    ax.set_xlim(midHours2, endHours)
    ax.yaxis.tick_right()
    ax.spines['left'].set_visible(False)
    ax.yaxis.tick_right()
    ax.tick_params(labelleft='off')

    ax2 = fig.add_subplot(gs1[0], sharey=ax)
    ax2.set_xlim(startHours, midHours1)
    ax2.tick_params(labelright='off')
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.tick_left()

    d = .015 # how big to make the diagonal lines in axes coordinates

    kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
    ax.plot((1-d+gap,1+d+gap), (-d,+d), **kwargs)
    ax.plot((1-d+gap,1+d+gap),(1-d,1+d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((1-d,1+d), (-d,+d), **kwargs)
    ax2.plot((1-d,1+d),(1-d,1+d), **kwargs)

    predNew = ax.axvspan(0, midHours, facecolor='C2', alpha=0.2, label='Prediction window')

    lines = {}

    for i in range(len(users)):
        color = 'C3'
        label = 'Churned user (censored)'
        observed = users.iloc[i].observed
        if observed:
            if pd.Timedelta((users.iloc[i].deltaNextHours - users.iloc[i].recency)*np.timedelta64(1,'h')) + pd.Timestamp('2016-02-01') > predPeriod['mid']:
                color = 'C4'
                label = 'Returning user (censored)'
            else:
                color = 'C0'
                label = 'Returning user (uncensored)'

        lines['Session'] = ax.scatter([users.iloc[i].start], [i], marker='D', color='black', alpha=.3, label='Session')
        lines['Session'].remove()
        lines[label] = ax.plot([users.iloc[i].start, users.iloc[i].end], [i,i], 'D-', color=color, label=label)[0]
        ax.plot([startHours, users.iloc[i].start], [i,i], ':', color=color, label=label, alpha=.1)[0]
        ax2.plot([startHours, users.iloc[i].start], [i,i], ':', color=color, label=label, alpha=.1)[0]

        if not observed:
            lines[label].remove()
            ax.plot([users.iloc[i].start, users.iloc[i].end], [i,i], '-', color=color, label=label)
            ax.scatter([users.iloc[i].start], [i], marker='D', color=color, label=label)


    xDates2 = [pd.datetime(2015,i,1) for i in range(2,4,2)]
    xDates = [pd.datetime(2015,i,1) for i in range(10,13,2)] + [pd.datetime(2016,i,1) for i in range(2,7,2)]
    xDates2Hours = [(d - predPeriod['start']).to_timedelta64()/np.timedelta64(1,'h') for d in xDates2]
    xDatesHours = [(d - predPeriod['start']).to_timedelta64()/np.timedelta64(1,'h') for d in xDates]
    xDates2Str = [d.strftime('%Y-%m') for d in xDates2]
    xDatesStr = [d.strftime('%Y-%m') for d in xDates]
    ax2.set_xticks(xDates2Hours)
    ax2.set_xticklabels(xDates2Str)
    ax.set_xticks(xDatesHours)
    ax.set_xticklabels(xDatesStr)

    ax.set_ylim([-1,10])
    ax.set_yticks([])
    ax2.set_yticks([])
    handles=[predNew] + list(lines.values())
    leg = ax2.legend(handles=handles, labels=[h.get_label() for h in handles], loc=4, bbox_to_anchor=(3.43, .022))
    leg.get_frame().set_alpha(1)

    plt.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.12)
    # fig.tight_layout()
    fig.show()

