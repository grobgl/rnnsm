import sys
import os
sys.path.insert(0, '../utils')

# from dataPiping import makeunixtime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')


class PlotsRaw:
    def __init__(self):
        self.loadData()

    def loadData(self):
        """Loads data into df
        """
        if os.path.exists('../../data/cleaned/stage1.pkl'):
            self.df = pd.read_pickle('../../data/cleaned/stage1.pkl')

        # self.df = pd.read_pickle('../../data/sessionDF.pkl')

        # # convert session start time to pandas time index
        # self.df.startTime = pd.DatetimeIndex(self.df.startTime.apply(makeunixtime) * 1000000000)
        # self.df.startUserTime = pd.DatetimeIndex(self.df.startUserTime.apply(makeunixtime) * 1000000000)
        # self.df.sessionLength = pd.TimedeltaIndex(self.df.sessionLength * 1000000000)

        # self.df['endTime'] = self.df.startTime + self.df.sessionLength
        # self.df['endUserTime'] = self.df.startUserTime + self.df.sessionLength

    def pruneLongCust(self):
        self.df['sessionLenSec'] = self.df.sessionLength/np.timedelta64(1,'s')
        longCust = self.df[self.df.sessionLenSec > 10e2].customerId.unique()
        self.df = self.df[~self.df.customerId.isin(longCust)]


    # check for null values
    def checkNullValues(self):
        print(self.df.isnull().sum())

    # check for number of values in interaction markers
    def checkMarkerCounts(self):
        print(self.df[['numberdivisions','viewonly','changeThumbnail','imageZoom','watchVideo','view360','sizeGuide']].sum())

    def plotTimeOfDay(self):
        """Plots visits vs time of day
        """
        startTimes = self.df.startUserTime.dropna()
        startTimes += np.timedelta64(-4*60*60,'s')
        minutes = np.array(list(map(lambda t: t.hour*60 + t.minute, startTimes)))

        plt.hist(minutes, bins=24*60)
        plt.xticks(np.linspace(0,24*60,25), np.roll(range(0,25),-4))
        plt.xlabel('time of day (h)')
        plt.ylabel('visits')

        plt.show()


    def plotDayOfWeek(self):
        """Plots visits vs time of day
        """
        countPerDay = np.bincount(pd.DatetimeIndex(self.df.startUserTime.dropna()).dayofweek)

        plt.bar(range(7), countPerDay)
        plt.xticks(range(7), ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
        plt.xlabel('day of week')
        plt.ylabel('visits')

        plt.show()

    def plotDayOfMonth(self):
        """Plots visits vs day of month
        """
        countPerDay = np.bincount(pd.DatetimeIndex(self.df.startUserTime.dropna()).day)[1:]

        plt.bar(range(31), countPerDay)
        plt.xticks(range(31), range(1,32))
        plt.xlabel('day of month')
        plt.ylabel('visits')

        plt.show()


    def plotWeekOfYear(self):
        """Plots visits vs week of year
        """
        countPerWeek = np.bincount(pd.DatetimeIndex(self.df.startTime.dropna()).week)[1:]

        plt.bar(range(len(countPerWeek)), countPerWeek)
        plt.xticks(range(len(countPerWeek)), range(1,len(countPerWeek)+1))
        plt.xlabel('week of year')
        plt.ylabel('visits')

        plt.show()


    def plotVisitsByDay(self):
        """Plots visits by day in obs. timeframe
        """
        days = pd.TimedeltaIndex(self.df.startTimeDelta).days
        days_count = np.bincount(days)
        plt.plot(range(len(days_count)), days_count)
        plt.xlabel('day')
        plt.ylabel('number of observations')

        plt.show()

    def plotReturnTime(self):
        plt.hist(pd.TimedeltaIndex(self.df.returnTime.dropna()).days, bins=540)
        plt.yscale('log', nonposy='clip')
        plt.xlabel('Return time (days)')
        plt.ylabel('Number of sessions')
        plt.show()

    def plotSessionLength(self):
        sessionLenSec = pd.TimedeltaIndex(self.df.sessionLength).get_values()/np.timedelta64(1,'s')
        plt.hist(sessionLenSec, bins=np.logspace(0,np.log10(sessionLenSec.max()),500))
        plt.xscale('log', nonposy='clip')
        plt.yscale('log', nonposy='clip')
        plt.xlabel('Session length (s)')
        plt.show()

    def plotSessionLengthVsDevice(self):
        # sessionLenSec = pd.TimedeltaIndex(self.df.sessionLength).get_values()/np.timedelta64(1,'s')
        xMax = 0
        for device in self.df.device.unique():
            sessionLenLogNS = np.log10(self.df[self.df.device==device].sessionLength/np.timedelta64(1,'s') + 1)
            y,binEdges=np.histogram(sessionLenLogNS,bins=500)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            xMax = max(xMax, bincenters.max())
            plt.plot(bincenters,y, label=device)

        # plt.xscale('log', nonposy='clip')
        plt.yscale('log', nonposy='clip')
        # xTicks = np.array(list(range(0, np.ceil(xMax).astype('int64')+1)))
        # plt.xticks(xTicks,map(_secToStr,10**xTicks))
        plt.xlabel('Session length (s)')
        plt.ylabel('Number of sessions')
        plt.legend()

        plt.show()

    def plotSessionLengthVs(self, vs='dayOfMonth'):
        dfLen = self.df.copy()
        dfLen['sessionLenSec'] = dfLen.sessionLength/np.timedelta64(1,'s')

        dfLen.groupby(vs).sessionLenSec.mean().plot()

        plt.yscale('log', nonposy='clip')
        plt.ylabel('Mean session length (s)')

        plt.show()


    def _secToStr(self, secs):
        return '{:02d}:{:02d}:{:02d}'.format(secs//3600, secs%3600//60, secs%60)

    def plotReturnTimeVs(self, vs):
        dfNoNan = self.df[~self.df.returnTime.isnull()]
        retDays = pd.TimedeltaIndex(dfNoNan.returnTime.values).days
        # sns.regplot(x=vs, y='returnTime', data=dfNoNan)
        sns.boxplot(x=dfNoNan[vs].values, y=np.log10(retDays))

        yTicks = np.array(range(0, int(np.floor(np.log10(retDays.max()))) + 1))
        plt.yticks(yTicks,10**yTicks)
        plt.show()


    def plotInteractionVsDevice(self):
        interactions = sum(self.df[col] for col in ['changeThumbnail', 'imageZoom', 'watchVideo', 'view360'])
        sns.boxplot(x=self.df.device.values, y=np.log10(interactions.values + 1))

        yTicks = np.array(range(0, int(np.floor(np.log10(interactions.max()))) + 1))
        yTicks[0] = 1
        plt.yticks(yTicks,10**yTicks)
        plt.ylabel('number of interactions')

        plt.show()
