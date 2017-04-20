import sys
import os
sys.path.insert(0, '../utils')

from dataPiping import makeunixtime
from plot_format import *
import seaborn.apionly as sns
import matplotlib.dates as mdates

import datetime
import math
import numpy as np
import pandas as pd

from matplotlib.collections import LineCollection

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
obs_timefr = [pd.datetime(2015,2,1), pd.datetime(2016,2,1)]
pred_timefr = [pd.datetime(2016,2,1), pd.datetime(2016,6,1)]

# read raw data
def loadData(stage=2, pruned=True):
    """Loads data into df
    """
    if stage == 0:
        return pd.read_pickle('../../data/cleaned/stage0.pkl')

    if stage == 1:
        return pd.read_pickle('../../data/cleaned/stage1{}.pkl'.format('_pruned' if pruned else '_startUserTimes'))

    return pd.read_pickle('../../data/cleaned/stage{}{}.pkl'.format(stage, '_pruned' if pruned else ''))


def loadStage2():
    df = pd.read_pickle('../../data/cleaned/stage2_pruned.pkl')
    return df

# check for null values
def checkNullValues(df):
    print(df.isnull().sum())

# check for number of values in interaction markers
def checkMarkerCounts(df):
    print(df[['numberdivisions','viewonly','changeThumbnail','imageZoom','watchVideo','view360','sizeGuide']].sum())


def plotTimeOfDay(df, width=1, height=None):
    """Plots visits vs time of day
    """
    dayCount = len(df.startUserTime.dropna().dt.date.unique())
    startTimes = df.startUserTime.dropna()
    startTimes += np.timedelta64(-4*60*60,'s')
    minutes = np.array(list(map(lambda t: t.hour*60 + t.minute, startTimes)))

    fig, ax = newfig(width, height)

    n,bins,patches  = ax.hist(minutes, bins=24*12)
    maxN = max(n)
    maxY = maxN / dayCount
    yticks = np.linspace(0, int(maxY), int(maxY+1))

    ax.set_yticks(yticks * dayCount)
    ax.set_yticklabels(yticks)
    ax.set_xticks(np.linspace(0,24*60,25))
    ax.set_xticklabels(np.roll(range(0,25),-4))
    ax.set_xlabel(r'Time of day \texttt{[h]}')
    ax.set_ylabel('Mean number of sessions')

    ax.margins(0,.1)
    fig.show()


def plotDayOfWeek(df, width=1, height=None):
    """Plots visits vs time of day
    """
    dayCount = np.bincount([d.weekday() for d in df.startUserTime.dropna().dt.date.unique()])
    countPerDay = np.bincount(pd.DatetimeIndex(df.startUserTime.dropna()).dayofweek) / dayCount

    fig, ax = newfig(width, height)

    ax.bar(range(7), countPerDay)
    ax.set_xticks(range(7))
    ax.set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
    ax.set_xlabel('Day of week')
    ax.set_ylabel('Mean number of sessions')

    fig.show()

def plotDayOfMonth(df, width=1, height=None):
    """Plots visits vs day of month
    """
    countPerDay = np.bincount(pd.DatetimeIndex(df.startUserTime.dropna()).day)[1:]

    fig, ax = newfig(width, height)

    ax.bar(range(31), countPerDay)
    ax.set_xticks(range(31))
    ax.set_xticklabels(range(1,32))
    ax.set_xlabel('Day of month')
    ax.set_ylabel('Number of sessions')

    fig.show()


def plotWeekOfYear(df, width=1, height=None):
    """Plots visits vs week of year (jan 2015 to jan 2016)
    """
    countPerWeek = np.bincount(pd.DatetimeIndex(df[(df.startTime >= pd.datetime(year=2015,month=1,day=1)) & (df.startTime < pd.datetime(year=2016,month=1,day=1))].startTime.dropna()).week)[1:]

    fig, ax = newfig(width, height)

    ax.bar(range(len(countPerWeek)), countPerWeek)
    ax.set_xticks(range(len(countPerWeek)))
    ax.set_xticklabels(range(1,len(countPerWeek)+1))
    ax.set_xlabel('Week of year')
    ax.set_ylabel('Number of sessions')

    fig.show()


def plotVisitsByDay(df, width=1, height=None):
    """Plots visits by day in obs. timeframe
    """
    fig, ax = newfig(width, height)

    df.groupby(df.startTime.dt.date).customerId.count().plot(ax=ax)
    obs = ax.axvspan(pd.datetime(2015,2,1), pd.datetime(2016,2,1), facecolor=colors[2], alpha=0.2, label='Observation time frame')
    pred = ax.axvspan(pd.datetime(2016,2,1), pd.datetime(2016,6,1), facecolor=colors[3], alpha=0.2, label='Prediction time frame')

    ax.set_xlabel('Date')
    ax.set_ylabel('Number of sessions')
    xTicks = [pd.datetime(2015,i,1) for i in range(2,13,2)] + [pd.datetime(2016,i,1) for i in range(2,7,2)]
    ax.set_xticks(xTicks)
    ax.legend(handles=[obs,pred])

    fig.tight_layout()
    fig.show()

def plotReturnTime(df, width=1, height=None):
    fig, ax = newfig(width, height)

    ax.hist(df.deltaNext.dropna().dt.days, bins=540)
    ax.set_yscale('log', nonposy='clip')
    ax.set_xlabel(r'Return time \texttt{[days]}')
    ax.set_ylabel('Number of sessions')
    ax.margins(0,.02)

    fig.tight_layout()
    fig.show()

def plotSessionLengthHist(df, width=1, height=None):
    fig, ax = newfig(width, height)
    sessionLenSec = pd.TimedeltaIndex(df.sessionLength).get_values()/np.timedelta64(1,'s')
    ax.hist(sessionLenSec, bins=np.logspace(0,np.log10(sessionLenSec.max()),500))
    ax.set_xscale('log', nonposy='clip')
    ax.set_yscale('log', nonposy='clip')
    ax.set_xlabel('Session length (s)')
    ax.set_ylabel('Number of sessions')
    ax.scatter(10**5, 1, alpha=0) # just to grow plot window to the right

    fig.tight_layout()
    ax.margins(0, .05)
    fig.show()

def plotSessionLengthVsDate(df, width=1, height=None):
    fig, ax = newfig(width, height)

    df.groupby(df.startTime.dt.date).apply(lambda x: (x.sessionLength/np.timedelta64(1,'s')).mean()).plot(ax=ax, label=r'Mean session length \texttt{[s]}')
    df.groupby(df.startTime.dt.date).apply(lambda x: (x.sessionLength/np.timedelta64(1,'s')).median()).plot(ax=ax, label=r'Median session length \texttt{[s]}')

    ax.legend()
    ax.set_yscale('log', nonposy='clip')
    # ax.set_ylabel(r'Mean session length \texttt{[s]}')
    ax.set_xlabel('Date')

    fig.tight_layout()
    fig.show()

def plotSessionLengthVsDevice(df, width=1, height=None):
    # sessionLenSec = pd.TimedeltaIndex(df.sessionLength).get_values()/np.timedelta64(1,'s')
    fig, ax = newfig(width, height)

    xMax = 0
    for device in df.device.unique():
        sessionLenLogNS = np.log10(df[df.device==device].sessionLength/np.timedelta64(1,'s') + 1)
        y,binEdges=np.histogram(sessionLenLogNS,bins=500)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        xMax = max(xMax, bincenters.max())
        ax.plot(bincenters,y, label=device)

    # ax.xscale('log', nonposy='clip')
    # ax.yscale('log', nonposy='clip')
    xTicks = np.array(list(range(0, np.ceil(xMax).astype('int64')+1)))
    ax.set_xticks(xTicks)
    ax.set_xticklabels(map(_secToStr,10**xTicks))
    ax.set_ylabel(r'Session length \texttt{[s]}')
    ax.set_xlabel('Device')
    ax.legend()

    fig.show()

def plotSessionLengthVs(df, vs='dayOfMonth'):
    fig, ax = newfig(width, height)

    dfLen = df.copy()
    dfLen['sessionLenSec'] = dfLen.sessionLength/np.timedelta64(1,'s')

    dfLen.groupby(vs).sessionLenSec.mean().plot(ax=ax)

    # plt.yscale('log', nonposy='clip')
    # plt.ylabel('Mean session length (s)')

    fig.show()


def _secToStr(secs):
    return '{:02d}:{:02d}:{:02d}'.format(secs//3600, secs%3600//60, secs%60)

def plotDeltaNextDiscVs(df, vs, width=1, height=None):
    fig, ax = newfig(width, height)

    dfNoNan = df[~df.deltaNext.isnull()]
    retDays = pd.TimedeltaIndex(dfNoNan.deltaNext.values).days
    # sns.regplot(x=vs, y='deltaNext', data=dfNoNan)
    sns.boxplot(x=dfNoNan[vs].values, y=np.log10(retDays), ax=ax)

    yTicks = np.array(range(0, int(np.floor(np.log10(retDays.max()))) + 1))
    ax.set_yticks(yTicks)
    ax.set_yticklabels(10**yTicks)
    # ax.set_ylabel('Return time')
    fig.show()

def plotDeltaNextContVs(df, vs, width=1, height=None):
    """
    Plots return time of sessions in 2015
    """
    fig, ax = newfig(width, height)
    df[df.startUserTime < pd.datetime(2016,1,1)].groupby(vs).deltaNextDays.mean().plot()
    fig.show()


def plotInteractionVsDevice(df, width=1, height=None):
    fig, ax = newfig(width, height)

    interactions = sum(df[col] for col in ['changeThumbnail', 'imageZoom', 'watchVideo', 'view360'])
    sns.boxplot(x=df.device.values, y=np.log10(interactions.values + 1))

    yTicks = np.array(range(0, int(np.floor(np.log10(interactions.max()))) + 1))
    yTicks[0] = 1
    ax.set_yticks(yTicks)
    ax.set_yticklabels(10**yTicks)
    ax.set_ylabel('number of interactions')

    fig.show()

def plotDevicesByTime(df, width=1, height=None):
    dfC = df.copy()
    dfC['startDate'] = dfC.startTime.dt.date
    devices = dfC.device.unique()
    fig, ax = newfig(width, height)

    msFactor = 1/dfC.groupby(['device','startDate']).startDate.count().max()

    for i, device in enumerate(devices):
        print(i, device)
        sessions = dfC[dfC.device == device]
        for date, count in sessions.groupby('startDate').startDate.count().iteritems():
            ax.plot_date(date, i, ms=count*20*msFactor,color=colors[i],marker='|')

    ax.margins(0, .25)
    ax.set_yticks(list(range(len(devices))))
    ax.set_yticklabels(r'\texttt{{{}}}'.format(d) for d in devices)
    ax.set_ylabel('Device')
    ax.set_xlabel('Date')

    fig.tight_layout()
    fig.show()


def plotActionsByDeviceAndTime(df, width=1, height=None):
    dfC = df.copy()
    dfC['startDate'] = dfC.startTime.dt.date
    dfC['interactions'] = dfC.changeThumbnail + dfC.imageZoom + dfC.watchVideo + dfC.view360
    devices = ['unknown','desktop','mobile']
    devPosition = [-1,1,2]
    fig, ax = newfig(width, height)

    msFactor = 1/dfC.groupby(['device','startDate']).startDate.count().max()

    for j, (i, device) in enumerate(zip(devPosition, devices)):
        print(i, device)
        sessions = dfC[dfC.device == device]
        msFactor = 1/(sessions.groupby(['startDate']).interactions.sum().max()+1)
        for date, count in sessions.groupby('startDate').interactions.sum().iteritems():
            ax.plot_date(date, i, ms=count*20*msFactor,color=colors[j],marker='|')

    ax.plot_date(dfC.startDate.min(), 2.5, ms=0., marker='|')

    sessionOffsetWeb = .5
    numSessionsWeb = dfC[dfC.device.isin(['mobile','desktop'])].groupby('startDate').startDate.count()
    numSessionsWeb = numSessionsWeb / numSessionsWeb.max() - sessionOffsetWeb
    ax.fill_between(numSessionsWeb.keys().values, -sessionOffsetWeb, numSessionsWeb.values, facecolor='grey', alpha=.5)

    sessionOffsetUnk = 2.5
    numSessionsUnk = dfC[dfC.device=='unknown'].groupby('startDate').startDate.count()
    numSessionsUnk = numSessionsUnk / numSessionsUnk.max() - sessionOffsetUnk
    ax.fill_between(numSessionsUnk.keys().values, -sessionOffsetUnk, numSessionsUnk.values, facecolor='grey', alpha=.5)

    ax.axhline(y=-sessionOffsetWeb, c="black", linewidth=.75)
    ax.margins(0, 0)
    ax.set_yticks([-sessionOffsetWeb + .5, -sessionOffsetUnk + .5] + devPosition)
    ax.set_yticklabels([r'\textit{Total web}' + '\n' + r'\textit{sessions}', r'\textit{Total}' + '\n' + r'\textit{unknown}' + '\n' + r'\textit{sessions}'] + [r'\texttt{{{}}}'.format(d) for d in devices])
    ax.set_ylabel('Interactions by device')
    ax.set_xlabel('Date')

    fig.tight_layout()
    fig.show()


def plotSessionMarkers(df, width=1, height=None):
    markers = ['viewonly', 'changeThumbnail', 'imageZoom', 'watchVideo', 'view360']

    dfC = df.copy()
    dfC['startDate'] = pd.DatetimeIndex(dfC.startTime.dt.date)
    fig, ax = newfig(width, height)

    for i, marker in enumerate(markers):
        msFactor = 1/(dfC.groupby(['startDate'])[marker].sum().max()+1)
        print(i, marker, 1/msFactor)
        for date, count in dfC.groupby('startDate')[marker].sum().iteritems():
            ax.plot_date(date, i, ms=count*20*msFactor,color=colors[i],marker='|')

    ax.plot_date(dfC.startDate.min(), 4.5, ms=0., marker='|')

    sessionOffset = 1.5
    numSessions = dfC.groupby('startDate').startDate.count()
    numSessions = numSessions / numSessions.max() - sessionOffset
    ax.fill_between(numSessions.keys().values, -sessionOffset, numSessions.values, facecolor='grey', alpha=.5)

    # ax.margins(0, .25)
    ax.margins(0, 0)
    ax.set_yticks([-1] + list(range(len(markers))))
    ax.set_yticklabels([r'\textit{Num. of sessions}'] + [r'\texttt{{{}}}'.format(d) for d in markers])
    ax.set_ylabel('Action')
    ax.set_xlabel('Date')

    fig.tight_layout()
    fig.show()


def plotSessionUnknownDeviceSessions(df, width=1, height=None):
    """Plots all sessions by device of users who have any session with unknown device.

    :df: dataframe
    :width: plot width
    :height: plot height
    :returns: None

    """

    dfCC = df.copy()
    dfCC['startDate'] = pd.DatetimeIndex(dfCC.startTime.dt.date)
    unknownCust = df[df.device=='unknown'].customerId.unique()
    dfC = dfCC[dfCC.customerId.isin(unknownCust)]
    devices = dfC.device.unique()
    fig, ax = newfig(width, height)

    msFactor = 1/(dfC.groupby(['startDate','device'])).startDate.count().max()

    for i, device in enumerate(devices):
        print(i, device)
        sessions = dfC[dfC.device == device]
        # msFactor = 1/(sessions.groupby(['startDate']).startDate.count().max()+1)
        for date, count in sessions.groupby('startDate').startDate.count().iteritems():
            ax.plot_date(date, i, ms=count*20*msFactor,color=colors[i],marker='|')

    ax.plot_date(dfC.startDate.min(), 4.5, ms=0., marker='|')

    sessionOffset = 1.5
    numSessions = dfC.groupby('startDate').startDate.count()
    numSessions = numSessions / numSessions.max() - sessionOffset
    ax.fill_between(numSessions.keys().values, -sessionOffset, numSessions.values, facecolor='grey', alpha=.5)

    # ax.margins(0, .25)
    ax.margins(0, 0)
    ax.set_yticks([-1] + list(range(len(devices))))
    ax.set_yticklabels([r'\textit{Num. of sessions}'] + [r'\texttt{{{}}}'.format(d) for d in devices])
    ax.set_ylabel('Device')
    ax.set_xlabel('Date')

    fig.tight_layout()
    fig.show()



def plotMissingLocalTimeSessions(df, width=1, height=None):
    """Plots all sessions with unknown local time by device.

    :df: dataframe
    :width: plot width
    :height: plot height
    :returns: None

    """

    dfCC = df.copy()
    dfCC['startDate'] = pd.DatetimeIndex(dfCC.startTime.dt.date)
    dfC = dfCC[dfCC.startUserTime.isnull()]
    devices = dfC.device.unique()
    fig, ax = newfig(width, height)

    msFactor = 1/(dfC.groupby(['startDate','device'])).startDate.count().max()

    for i, device in enumerate(devices):
        print(i, device)
        sessions = dfC[dfC.device == device]
        # msFactor = 1/(sessions.groupby(['startDate']).startDate.count().max()+1)
        for date, count in sessions.groupby('startDate').startDate.count().iteritems():
            ax.plot_date(date, i, ms=count*20*msFactor,color=colors[i],marker='|')

    ax.plot_date(dfC.startDate.min(), len(devices) - .5, ms=0., marker='|')

    sessionOffset = 1.5
    numSessions = dfCC.groupby('startDate').startDate.count()
    numSessions = numSessions / numSessions.max() - sessionOffset
    ax.fill_between(numSessions.keys().values, -sessionOffset, numSessions.values, facecolor='grey', alpha=.5)

    # ax.margins(0, .25)
    ax.margins(0, 0)
    ax.set_yticks([-1] + list(range(len(devices))))
    ax.set_yticklabels([r'\textit{Num. of sessions}'] + [r'\texttt{{{}}}'.format(d) for d in devices])
    ax.set_ylabel('Device')
    ax.set_xlabel('Date')

    fig.tight_layout()
    fig.show()

def plotChurnCust(df, width=1, height=None):
    fig, ax = newfig(width, height)

    obsDf = df[(df.startUserTime >= pd.datetime(2015,2,1)) & (df.startUserTime < pd.datetime(2016,2,1))].copy()

    churn = obsDf.groupby('customerId').tail(1)
    churnCust = churn[churn.deltaNext.isnull()].customerId.unique()
    print(len(churnCust))
    nonChurnCust = churn[~churn.deltaNext.isnull()].customerId.unique()
    print(len(nonChurnCust))

    obsDf[obsDf.customerId.isin(churnCust)].startUserTime.hist(ax=ax, bins=365, alpha=.5)
    obsDf[obsDf.customerId.isin(nonChurnCust)].startUserTime.hist(ax=ax, bins=365, alpha=.5)

    fig.show()


def plotChurnWindows(width=1, height=None):
    """Plots visits by day in obs. timeframe
    """
    fig, ax = newfig(width, height)
    start = pd.datetime(2015,1,1)

    obs = ax.axvspan(pd.datetime(2015,2,1), pd.datetime(2016,2,1), facecolor=colors[2], alpha=0.2, label='Observation window')
    act = ax.axvspan(pd.datetime(2015,10,1), pd.datetime(2016,2,1), ymin=.075, ymax=.925, facecolor='#87de87', label='Activity window')
    pred = ax.axvspan(pd.datetime(2016,2,1), pd.datetime(2016,6,1), facecolor=colors[3], alpha=0.2, label='Prediction window')

    noChurnAllWin = [pd.datetime(2015,4,10), pd.datetime(2015,6,23), pd.datetime(2015,7,28), pd.datetime(2015,8,4),
                     pd.datetime(2015,10,23), pd.datetime(2015,11,27), pd.datetime(2015,11,29),
                     pd.datetime(2015,12,18), pd.datetime(2016,3,9)]
    churnObsAct = [pd.datetime(2015,3,16), pd.datetime(2015,5,27), pd.datetime(2015,8,10), pd.datetime(2016,1,4)]

    for i,dates in enumerate([noChurnAllWin, churnObsAct]):
        sess = ax.plot_date(dates, [i+3]*len(dates), color=colors[0], linestyle='dashed', label='Customer session')

    noChurnOnlyPred = [pd.datetime(2016,2,29), pd.datetime(2016,4,16), pd.datetime(2016,5,19)]
    churnNoAct = [pd.datetime(2015,2,28), pd.datetime(2015,3,16), pd.datetime(2015,8,19)]
    noChurnNoAct = [pd.datetime(2015,3,21), pd.datetime(2015,8,27), pd.datetime(2015,9,3), pd.datetime(2016,3,29),
                    pd.datetime(2016,4,14), pd.datetime(2016,4,25), pd.datetime(2016,5,8)]
    for i,dates in enumerate([noChurnOnlyPred, churnNoAct, noChurnNoAct]):
        ax.plot_date(dates, [i]*len(dates), color=colors[0], linestyle='dashed')

    ax.scatter(pd.datetime(2016,2,29), -1, alpha=0) # just to grow plot window to the right

    ylab = [r'\texttt{not churned}'+'\n'+r'\textit{not included}',
            r'\texttt{churned}'+'\n'+r'\textit{not included}',
            r'\texttt{not churned}'+'\n'+r'\textit{not included}',
            r'\texttt{not churned}'+'\n'+r'\textit{included}',
            r'\texttt{churned}'+'\n'+r'\textit{included}']
    ax.set_yticks(range(len(ylab)))
    ax.set_yticklabels(ylab)

    xTicks = [pd.datetime(2015,i,1) for i in range(2,13,2)] + [pd.datetime(2016,i,1) for i in range(2,7,2)]
    ax.set_xticks(xTicks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.legend(handles=[obs,act,pred,sess[0]], ncol=2)
    # ax.legend()

    ax.margins(0,.2)
    fig.tight_layout()
    fig.show()
