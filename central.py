from pandas_datareader import data as pdr
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

import collections
import csv
import datetime
from datetime import date, timedelta
import numpy as np
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

import random

import os

import fix_yahoo_finance as yf, numpy as np

# Yahoo! Finance no longer works, so the following line is a patch
yf.pdr_override()

pd.options.mode.chained_assignment = None

US_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())

# Stock to analyze
ticker = "FB"

# List of discretely labeled price movements
deltaList = ["Very Much Up", "Up", "Neutral", "Down", "Very Much Down"]


def ema(stock, period, code):
    """Calculates the expnential moving average of a stock's price given a period of days
    E.g. Period = 12 or 26 for the 12-Day EMA and 26-Day EMA respectively
    The code allows the system to calculate the EMA of the stock price and the EMA of the MACD.
    It is also possible to detect a change in trend when the MACD crosses the MACD Signal."""

    if code == "EMA":
        # Calculates EMA if the period is shorter than data; otherwise, calculates the simple moving average (SMA)
        if period < len(stock.index):
            for i in range(period, len(stock.index)):

                # List of last (period) closing prices at i
                closingPrices = [stock['Adj Close'][i - j] for j in range(period, 0, -1)]

                # SMA of those closing prices
                sma = sum(closingPrices)/period

                # EMA multiplier
                multiplier = 2.0/(period + 1)

                stock['EMA' + str(period)][i] = sma

                if i != period:
                    stock['EMA' + str(period)][i] = (stock['Adj Close'][i] - stock['EMA' + str(period)][i - 1]) * multiplier + stock['EMA' + str(period)][i - 1]

            return stock

        # Calculates just the simple moving average in the case of period > len(stock.index)
        else:
            for i in range(len(stock.index)):
                closingPrices = [stock['Adj Close'][j] for j in range(len(stock.index))]
                sma = sum(closingPrices)/len(stock.index)
                stock['EMA' + str(period)][i] = sma
            return stock
    # Calculates the EMA of the MACD (e.g. 9-Day MACD EMA) to produce the MACD signal
    # (which is the 9-day MACD EMA and a function of the difference between the 12-Day Price EMA and the 26-Day Price EMA)
    else:
        if period < len(stock.index):
            for i in range(period, len(stock.index)):
                macdList = [stock['MACD'][i - j] for j in range(period, 0, -1)]
                sma = sum(macdList)/period
                multiplier = 2.0/(period + 1)
                stock['MACD Signal'][i] = sma
                if i != period:
                    if stock['EMA12'][i] != 0 and stock['EMA26'][i] != 0:
                        stock['MACD Signal'][i] = ((stock['EMA12'][i] - stock['EMA26'][i]) - stock['MACD Signal'][i - 1]) * multiplier + stock['MACD Signal'][i - 1]
                    else:
                        stock['MACD Signal'][i] = 0

                # Trend change when the MACD crosses the MACD Signal
                if stock['MACD'][i] < stock['MACD Signal'][i] and stock['MACD'][i - 1] < stock['MACD Signal'][i - 1]:
                    stock['MACD Trend'][i] = 1.0
                if stock['MACD'][i] < stock['MACD Signal'][i] and stock['MACD'][i - 1] > stock['MACD Signal'][i - 1]:
                    stock['MACD Trend'][i] = 2.0
                if stock['MACD'][i] > stock['MACD Signal'][i] and stock['MACD'][i - 1] < stock['MACD Signal'][i - 1]:
                    stock['MACD Trend'][i] = 3.0
                if stock['MACD'][i] > stock['MACD Signal'][i] and stock['MACD'][i - 1] > stock['MACD Signal'][i - 1]:
                    stock['MACD Trend'][i] = 4.0
        # If period is large, use the SMA
        else:
            for i in range(len(stock.index)):
                macdList = [stock['MACD'][i - j] for j in range(period, 0, -1)]
                sma = sum(macdList)/period
                stock['MACD Signal'][i] = sma

        return stock

def macd(stock):
    """The moving average convergence divergence (MACD) is the difference between the 12-Day EMA and the 26-Day EMA
    and is a momentum indicator that indicates whether rcent price movement is high or low relative to past price movement.
    This indicator is useful when used in conjunction with its 9-Day EMA (the MACD Signal indicator)."""

    for i in range(len(stock.index)):
        if stock['EMA12'][i] != 0 and stock['EMA26'][i] != 0:
            stock['MACD'][i] = stock['EMA12'][i] - stock['EMA26'][i]
        else:
            stock['MACD'][i] = 0
    return stock

def convergence(stock, period):
    """The EMA Convergence assigns the price movement discretized values (from 1 to 6)
    based on whether the price is approaching or moving away from its EMA (and based on which
    side it is on).  We decided having six values instead of two (moving toward vs. moving away)
    would allow for more feature specificity in our classification.  You can see the commented
    lines where this would have been different in the two-value case."""

    for i in range(period, len(stock.index)):
        if stock['Adj Close'][i] - stock['EMA' + str(period)][i] > 0:
            if stock['Adj Close'][i] - stock['EMA' + str(period)][i] < stock['Adj Close'][i - 1] - stock['EMA' + str(period)][i - 1]:
                stock['EMA' + str(period) + ' Convergence'][i] = 1.0
            else:
                stock['EMA' + str(period) + ' Convergence'][i] = 2.0
        if stock['Adj Close'][i] - stock['EMA' + str(period)][i] < 0:
            if stock['EMA' + str(period)][i] - stock['Adj Close'][i] < stock['EMA' + str(period)][i - 1] - stock['Adj Close'][i - 1]:
                # stock['EMA' + str(period) + ' Convergence'][i] = 2.0
                stock['EMA' + str(period) + ' Convergence'][i] = 4.0
            else:
                # stock['EMA' + str(period) + ' Convergence'][i] = 1.0
                stock['EMA' + str(period) + ' Convergence'][i] = 3.0
        if stock['Adj Close'][i] - stock['EMA' + str(period)][i] >= 0 and stock['Adj Close'][i - 1] - stock['EMA' + str(period)][i - 1] < 0:
            # stock['EMA' + str(period) + ' Convergence'][i] = 2.0
            stock['EMA' + str(period) + ' Convergence'][i] = 5.0
        if stock['Adj Close'][i] - stock['EMA' + str(period)][i] <= 0 and stock['Adj Close'][i - 1] - stock['EMA' + str(period)][i - 1] > 0:
            # stock['EMA' + str(period) + ' Convergence'][i] = 1.0
            stock['EMA' + str(period) + ' Convergence'][i] = 6.0
    return stock

def rsi(stock, period):
    """The relative strength index (RSI) of the stock price is an indicator that illustrates "overbuying"/"overselling"
    of an asset by comparing its average recent gains and losses on a continual basis.  Crossing 50 in either direction
    can be used as a signal of trend change."""
    stock['RSI'][0] = 50
    gainsList = []
    lossesList = []
    for i in range(1, period):
        avgGain = sum(stock['pctChange'][i - j + 1] for j in range(i, 0, -1) if stock['pctChange'][i - j + 1] > 0)/i
        avgLoss = -1 * sum(stock['pctChange'][i - j + 1] for j in range(i, 0, -1) if stock['pctChange'][i - j + 1] < 0)/i
        if avgGain == 0:
            avgGain = 1
        if avgLoss == 0:
            avgLoss = 1
        gainsList.append(avgGain)
        lossesList.append(avgLoss)
        value = 100 - (100/(1 + avgGain/avgLoss))
        if value > 100:
            value = 100
        stock['RSI'][i] = value


    for i in range(period, len(stock.index)):
        avgGain = sum(stock['pctChange'][i - j] for j in range(period, 0, -1) if stock['pctChange'][i - j] > 0)/period
        gainsList.append(avgGain)
        avgLoss = -1 * sum(stock['pctChange'][i - j] for j in range(period, 0, -1) if stock['pctChange'][i - j] < 0)/period
        lossesList.append(avgLoss)

        value = 100 - (100/(1 + (gainsList[i - 1] * 13 + stock['pctChange'][i])/(lossesList[i - 1] * 13 + stock['pctChange'][i])))

        if value > 100:
            value = 100
        stock['RSI'][i] = value

        # Trend change when 50 is crossed
        if stock['RSI'][i] < 50.0 and stock['RSI'][i - 1] < 50.0:
            stock['RSI Trend'][i] = 1.0
        if stock['RSI'][i] < 50.0 and stock['RSI'][i - 1] >= 50.0:
            stock['RSI Trend'][i] = 2.0
        if stock['RSI'][i] >= 50.0 and stock['RSI'][i - 1] < 50.0:
            stock['RSI Trend'][i] = 3.0
        if stock['RSI'][i] >= 50.0 and stock['RSI'][i - 1] >= 50.0:
            stock['RSI Trend'][i] = 4.0


    return stock

def stochastic(stock, period):
    """The Stochastic Oscillator compares the previous closing price to the high and low
    of the past (period) days.  This indicator is another depiction of "overbuying"/"overselling"
    relative to the recent trading range of the price."""
    lows = []
    highs = []
    high = 0.0
    low = 0.0
    k = 0.0
    kList = [50.0]

    for i in range(1, period):
        lows = [stock['Low'][i - j] for j in range(i, 0, -1)]
        highs = [stock['High'][i - j] for j in range(i, 0, -1)]
        low = min(lows)
        high = max(highs)
        k = 100 * (stock['Adj Close'][i - 1] - low)/(high - low)
        if k > 100:
            k = 100
        kList.append(k)
        stock['Stochastic'][i] = k

    for i in range(period, len(stock.index)):
        lows = [stock['Low'][i - j] for j in range(period, 0, -1)]
        highs = [stock['High'][i - j] for j in range(period, 0, -1)]
        low = min(lows)
        high = max(highs)

        k = 100 * (stock['Adj Close'][i - 1] - low)/(high - low)
        if k > 100:
            k = 100
        kList.append(k)
        stock['Stochastic'][i] = k

    for i in range(period, len(stock.index)):
        stock['Stochastic SMA'][i] = sum(kList[i - j] for j in range(3))/3

    return stock

def volumeIndicator(stock):
    """A high-volume trading day might indicate the health of a given trend.  Therefore, this
    indicator assigns discrete values to various levels of high volume to add another feature to the
    specificity of the classification algorithms."""
    averageVolume = 0.0
    totalTrades = 0.0
    for i in range(1, len(stock.index)):
        totalTrades += stock['Volume'][i]
        averageVolume = totalTrades/(i + 1)
        if stock['Volume'][i] - 10000000 > averageVolume:
            stock['Volume Indicator'][i] = 3.0
        elif stock['Volume'][i] - 5000000 > averageVolume:
            stock['Volume Indicator'][i] = 2.0
        elif stock['Volume'][i] > averageVolume:
            stock['Volume Indicator'][i] = 1.0
    return stock

def consecutive(stock):
    """Another indicator is the number of consecutive trading days a stock has moved in a single direction.
    There are two potential outcomes of this kind of repetition: (1) a solidified trend that keeps going;
    or (2) a change in direction to converge back to a "real price."  We originally built a transitions probability matrix
    based on consecutive days (e.g. if you witness Down, Down, Down, what are the probabilities of the price going Up vs Down
    the next day), but that technique does not fit perfectly into the classification system (it spits out its own recommendation
    instead), so we decided simply to include the number of consecutive days as a feature with the understanding that
    the course of training should interpret historical instances of consecutive movement into probabilities for the future."""
    delta = "Neutral"
    for i in range(29, len(stock.index) - 1):
        delta = stock['delta'][i]
        if delta == "Very Much Up":
            delta = "Up"
        if delta == "Very Much Down":
            delta = "Down"
        for j in range(1, 29):
            if stock['delta'][i - j] in delta:
                stock['Consecutive'][i] += 1
            else:
                break
    return stock

def jointAnalysis(stock, daysAhead):
    """This function calculates the joint distributions we need to calculate the probability of a given move based on our recorded
    indicators.  This is a frequentist application of a Bayes Net.  An alternative and equally viable approach would have been to
    calculate the marginal probability of each indicator (which are parent nodes to the future price change according to our problem
    structure), and divide the full joint distribution by the product of those potentially independent probabilities, but the use of
    a frequentist joint istribution that includes each result and all the indicators for the numerator makes it very easy simply to
    divide by a joint distribution of the indicators.  Therefore, we are producing P(A|B, C) = P(A, B, C)/P(B, C) here (but with nine
    given indicators) for each possible move."""
    joints = []
    partialJoints = []
    for i in range(26, len(stock.index) - daysAhead):
        fullDistribution = (stock['delta'][i + daysAhead],
                convertRSI(stock['RSI'][i]), convertRSI(stock['Stochastic'][i]), convertMACDSignal(stock['MACD'][i] - stock['MACD Signal'][i]), stock['Consecutive'][i], stock['Volume Indicator'][i],
                stock['RSI Trend'][i], stock['MACD Trend'][i], stock['EMA12 Convergence'][i], stock['EMA26 Convergence'][i])
        partialDistribution = (convertRSI(stock['RSI'][i]), convertRSI(stock['Stochastic'][i]), convertMACDSignal(stock['MACD'][i] - stock['MACD Signal'][i]), stock['Consecutive'][i], stock['Volume Indicator'][i],
                stock['RSI Trend'][i], stock['MACD Trend'][i], stock['EMA12 Convergence'][i], stock['EMA26 Convergence'][i])
        joints.append(fullDistribution)
        partialJoints.append(partialDistribution)

    counter = collections.Counter(joints)
    partialCounter = collections.Counter(partialJoints)

    probabilities = {}
    partialProbabilities = {}

    # Calculates the probability of each joint distribution
    for key, value in counter.items():
        value = (value * 1.0)/len(joints)
        probabilities[key] = value

    # Calculates the probability of each partial joint distribution
    for key, value in partialCounter.items():
        value = (value * 1.0)/len(partialJoints)
        partialProbabilities[key] = value

    pd.DataFrame(probabilities, index = [0]).to_csv("DataFiles/" + ticker + 'fullJointDistribution.csv', index=False)
    pd.DataFrame(partialProbabilities, index = [0]).to_csv("DataFiles/" + ticker + 'partialJointDistribution.csv', index=False)

    # The function in the return statement returns the actual conditional probabilities (and so does the actual math past the joint distribution work)
    return conditionalProbabilities(probabilities, partialProbabilities)

def conditionalProbabilities(jointDistribution, partialJointDistribution):
    """Creates a dictionary with keys P(A|B, C) for each price delta in the joint distribution and every combination of indicators.
    Each key is assigned a value equal to the probability of the sum joint distribution (P(A, B, C)) divided by the probability
    of the partial joint distribution (P(B, C)).  This is our inference on the frequentist-built Bayes Net and forms the core of
    our classification technique."""
    conditionals = {}
    for key, value in jointDistribution.items():
        rKey = "R = " + str(key[1])
        sKey = "S = " + str(key[2])
        mKey = "M = " + str(key[3])
        cKey = "C = " + str(key[4])
        vKey = "V = " + str(key[5])
        rtKey = "RT = " + str(key[6])
        mtKey = "MT = " + str(key[7])
        twelveKey = "12EMA = " + str(key[8])
        twsixKey = "26EMA = " + str(key[9])
        conditionals["P(D = " + str(key[0]) + " | " + rKey + ", " + sKey + ", " + mKey + ", " + cKey + ", " + vKey + ", " + rtKey + ", " + mtKey + ", " + twelveKey + ", " + twsixKey + ")"] = value/(partialJointDistribution[(key[1], key[2], key[3], key[4], key[5], key[6], key[7], key[8], key[9])])
    pd.DataFrame(conditionals, index=[0]).to_csv("DataFiles/" + ticker + 'conditionalProbabilities.csv', index=False)

    # conditionals is a complete probability distribution, but it is not very orderly for the human eye, so we sort it first
    return dictSort(conditionals)
    # return conditionals

def dictSort(oldDict):
    """This function orders the conditional distribution by header value, so it is sequential visually and can be affirmed."""
    sortedDict = {}
    for twsixEMA in range(3):
        for twelveEMA in range(3):
            for macdTrend in range(5):
                for rsiTrend in range(5):
                    for volumeIndication in range(2):
                        for consecutiveDays in range(1, 4):
                            for macd in range(2):
                                for stochastic in range(5):
                                    for rsi in range(5):
                                        for delta in deltaList:
                                            key = "P(D = " + delta + " | R = " + str(rsi) + ", S = " + str(stochastic) + ", M = " + str(macd) + ", C = " + str(consecutiveDays) + ", V = " + str(volumeIndication) + ", RT = " + str(rsiTrend) + ", MT = " + str(macdTrend) + ", 12EMA = " + str(twelveEMA) + ", 26EMA = " + str(twsixEMA) + ")"
                                            if key in oldDict:
                                                sortedDict[key] = oldDict[key]
    pd.DataFrame(sortedDict, index=[0]).to_csv("DataFiles/" + ticker + 'SortedConditionalProbabilities.csv', index=False)
    return sortedDict

def convertRSI(percentage):
    """Makes the 0-100 RSI discrete with 5 values: 0, 1, 2, 3, 4."""
    value = int(percentage * 0.05)
    if value == 5:
        value = 4
    return value

def convertMACDSignal(difference):
    """An indicator variable of whether or not the MACD is greater than its signal."""
    if difference >= 0:
        return 1
    else:
        return 0
    # table['P(R = )']

def convert(percentage, probabilities):
    """Converts continuous price data to discrete labels and builds out emission-to-emission
    probability matrix"""
    if isinstance(percentage, float):
        if percentage > 0.025:
            probabilities[0] += 1
            return ["Very Much Up", probabilities, 0]
        if percentage <= 0.025 and percentage > 0.0010:
            probabilities[1] += 1
            return ["Up", probabilities, 1]
        if percentage >= -0.025 and percentage < -0.0010:
            probabilities[3] += 1
            return ["Down", probabilities, 3]
        if percentage < -0.025:
            probabilities[4] += 1
            return ["Very Much Down", probabilities, 4]
        else:
            probabilities[2] += 1
            return ["Neutral", probabilities, 2]

def prev_weekday(adate, period):
    """Calculates the previous weekday (US business day).  Keep in mind we must work
    within the parameters of functional trading days."""
    adate -= US_BUSINESS_DAY * period
    while not is_business_day(adate): # Mon-Fri are 0-4

        adate -= US_BUSINESS_DAY
    return adate

def next_weekday(adate, period):
    """Calculates the next weekday (US business day).  Keep in mind we must work
    within the parameters of functional trading days."""
    adate += timedelta(days=period)
    while not is_business_day(adate): # Mon-Fri are 0-4
        adate += timedelta(days=1)
    return adate

def is_business_day(date):
    """Checks whether a date is a US business day.  Technique found online."""
    return bool(len(pd.bdate_range(date, date)))

def stockHistory(symbol, startDate, endDate, code):
    """Compiles and processes the actual stock data.  Constructs transitions matrix and sets indicator values."""
    stock = pdr.get_data_yahoo(symbol, start=startDate, end=endDate)

    # Assign `Adj Close` to `daily_close`
    daily_close = stock[['Adj Close']]

    # Daily returns
    daily_pct_change = daily_close.pct_change()

    # Replace NA values with 0
    daily_pct_change.fillna(0, inplace=True)

    stock['pctChange'] = stock[['Adj Close']].pct_change()

    # stock['logDelta'] = np.log(daily_close.pct_change()+1)

    stock.fillna(0, inplace=True)


    # Default values necessary for later modification
    stock['delta'] = "Test"
    stock['EMA12'] = 0.0
    stock['EMA26'] = 0.0
    stock['EMA12 Convergence'] = 0
    stock['EMA26 Convergence'] = 0
    stock['MACD'] = 0.0
    stock['MACD Signal'] = 0.0
    stock['MACD Trend'] = 0
    stock['RSI'] = 0.0
    stock['RSI Trend'] = 0
    stock['Stochastic'] = 0.0
    stock['Stochastic SMA'] = 0.0
    stock['Consecutive'] = 1
    stock['Volume Indicator'] = 0


    # Transitions Matrix for emissions-based prediction
    probabilities = [0, 0, 0, 0, 0]
    transitions = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    transitionsDict = {}
    for key in [('Very Much Up', 'Very Much Up'), ('Very Much Up', 'Up'), ('Very Much Up', 'Neutral'), ('Very Much Up', 'Down'), ('Very Much Up', 'Very Much Down')]:
        transitionsDict[key] = 0
    for key in [('Up', 'Very Much Up'), ('Up', 'Up'), ('Up', 'Neutral'), ('Up', 'Down'), ('Up', 'Very Much Down')]:
        transitionsDict[key] = 0
    for key in [('Neutral', 'Very Much Up'), ('Neutral', 'Up'), ('Neutral', 'Neutral'), ('Neutral', 'Down'), ('Neutral', 'Very Much Down')]:
        transitionsDict[key] = 0
    for key in [('Down', 'Very Much Up'), ('Down', 'Up'), ('Down', 'Neutral'), ('Down', 'Down'), ('Down', 'Very Much Down')]:
        transitionsDict[key] = 0
    for key in [('Very Much Down', 'Very Much Up'), ('Very Much Down', 'Up'), ('Very Much Down', 'Neutral'), ('Very Much Down', 'Down'), ('Very Much Down', 'Very Much Down')]:
        transitionsDict[key] = 0

    sums = 0
    classes = []

    for i in range(0, len(stock.index)):
        result = convert(stock.iloc[i]['pctChange'], probabilities)
        stock['delta'][i] = result[0]
        classes.append(result[2])
        # Adds transitions to transitions matrix before producing probabilities
        if i > 0:
            transitionsDict[(result[0], stock['delta'][i - 1])] += 1
            if stock['delta'][i - 1] is "Very Much Up":
                transitions[result[2]][0] += 1
                sums += 1
            elif stock['delta'][i - 1] is "Up":
                transitions[result[2]][1] += 1
                sums += 1
            elif stock['delta'][i - 1] is "Neutral":
                transitions[result[2]][2] += 1
                sums += 1
            elif stock['delta'][i - 1] is "Down":
                transitions[result[2]][3] += 1
            elif stock['delta'][i - 1] is "Very Much Down":
                transitions[result[2]][4] += 1
                sums += 1

        probabilities = result[1]


    probabilitiesPct = []
    summed = probabilities[0] + probabilities[1] + probabilities[2] + probabilities[3] + probabilities[4]
    probabilitiesPct = [probabilities[0]/summed, probabilities[1]/summed, probabilities[2]/summed, probabilities[3]/summed, probabilities[4]/summed]

    # Updates stock data with indicators
    stock=ema(stock, 12, 'EMA')
    stock=ema(stock, 26, 'EMA')
    stock=macd(stock)
    stock=ema(stock, 9, 'MACD')
    stock=rsi(stock, 14)
    stock=stochastic(stock, 14)
    stock=consecutive(stock)
    stock=volumeIndicator(stock)
    stock=convergence(stock, 12)
    stock=convergence(stock, 26)

    stock.to_csv("DataFiles/" + symbol + code + ".csv")

    return (stock, classes, transitions)

def predictor(stock, conditionalProbabilityDistribution, period, date, transitions, indicator):
    """This function predicts the stock price movement on a given date.  It is called in testing() and
    is given the necessary information to make the prediction for the conditional probability classification
    technique, the emissions model technique, and the naive benchmark techniques.  These techniques are
    executed with an indicator input fed in from testing()."""
    prediction = "Neutral"
    system = "Default"
    dateDate = datetime.datetime.strptime(date, '%Y-%m-%d').date()

    checkValue = 0.0
    downSum = 0.0
    neutralSum = 0.0
    upSum = 0.0
    totalSum = 0.0

    while not is_business_day(dateDate):
        dateDate += 1
    date = dateDate
    # We want to predict a given date's stock price movement given the previous day's indicator values
    yesterday = prev_weekday(date, period).strftime('%Y-%m-%d')
    if yesterday in stock['RSI']:
        pastChange = stock['delta'][yesterday]
        listIndex = deltaList.index(pastChange)
        rKey = convertRSI(stock['RSI'][yesterday])
        sKey = convertRSI(stock['Stochastic'][yesterday])
        mKey = convertMACDSignal(stock['MACD'][yesterday] - stock['MACD Signal'][yesterday])
        cKey = stock['Consecutive'][yesterday]
        vKey = stock['Volume Indicator'][yesterday]
        rtKey = stock['RSI Trend'][yesterday]
        mtKey = stock['MACD Trend'][yesterday]
        twelveKey = stock['EMA12 Convergence'][yesterday]
        twsixKey = stock['EMA26 Convergence'][yesterday]

        choices = []

        for delta in deltaList:
            key = "P(D = " + delta + " | R = " + str(rKey) + ", S = " + str(sKey) + ", M = " + str(mKey) + ", C = " + str(cKey) + ", V = " + str(vKey) + ", RT = " + str(rtKey) + ", MT = " + str(mtKey) + ", 12EMA = " + str(twelveKey) + ", 26EMA = " + str(twsixKey) + ")"
            if key in conditionalProbabilityDistribution:
                choices.append((delta, conditionalProbabilityDistribution[key]))
            else:
                matrix = transitions[listIndex]
                choice = max(enumerate(matrix), key = lambda x: x[1])[0]

                choices.append((deltaList[choice], 1.0))

        # Conditional Probabilities Classification Prediction
        if indicator == 0:
            prediction = max(choices, key = lambda x: x[1])[0]
            system = "Conditional Probabilities"

        # Emissions Model Prediction
        elif indicator == 1:
            matrix = transitions[listIndex]
            prediction = deltaList[max(enumerate(matrix), key = lambda x: x[1])[0]]
            system = "Emissions Model"

        # First Naive Benchmark: Predict Last Change
        elif indicator == 2:
            prediction = pastChange
            system = "Naive (Last Change)"

        # Second Naive Benchmark: Predict Opposite of Last Change
        elif indicator == 3:
            if pastChange == "Very Much Up" or pastChange == "Up":
                prediction = "Down"
            elif pastChange == "Very Much Down" or pastChange == "Down":
                prediction = "Up"
            else:
                prediction = pastChange
            system = "Opposite"

        # Third Naive Benchmark: Choose random value and predict using distribution of
        # each direction historically
        elif indicator == 4:
            checkValue = random.random()
            upSum = sum(transitions[0]) + sum(transitions[1])
            neutralSum = sum(transitions[2])
            downSum = sum(transitions[3]) + sum(transitions[4])
            totalSum = upSum + neutralSum + downSum
            system = "Random"
            if checkValue <= upSum/totalSum:
                prediction = "Up"
            elif checkValue < (upSum + neutralSum)/totalSum:
                prediction = "Neural"
            else:
                prediction = "Down"

        # Fourth Naive Benchmark: Predict up always
        elif indicator == 5:
            prediction = "Up"
            system = "Only Up"

        # Fifth naive benchmark: Predict down always
        elif indicator == 6:
            prediction = "Down"
            system = "Only Down"

        print(system + " Prediction: The price will move " + prediction + " on " + date.strftime('%Y-%m-%d'))

        if date.strftime('%Y-%m-%d') in stock['RSI']:
            realAction = stock['delta'][date.strftime('%Y-%m-%d')]
            print("True action: " + realAction)

            # If correct, return +1 correct, +1 total;
            # else if incorrect, return +0 correct, +1 total;
            # else (for whatever reason there is nothing to predict),
            # return +0 correct, +0 total
            if 'Up' in prediction and 'Up' in realAction:
                print("Accurate")
                return (1.0, 1.0)

            elif 'Down' in prediction and 'Down' in realAction:
                print("Accurate")
                return (1.0, 1.0)
            elif prediction in realAction:
                print("Accurate")
                return (1.0, 1.0)
            else:
                print("Inaccurate")
                return (0.0, 1.0)
        else:
            return (0.0, 0.0)
    else:
        print("No prediction")
        return (0.0, 0.0)

def testing():
    """This function tests the conditional probability classification technique, the
    emissions model technique, and the naive benchmark techniques."""

    # Create folder for data files if not already existent
    if not os.path.exists("DataFiles/"):
        os.makedirs("DataFiles/")

    # Prediction start data
    startDate = '2018-01-01'
    predictionDateString = startDate
    predictionDate = datetime.datetime.strptime(predictionDateString, '%Y-%m-%d').date()
    # Prediction end date (as close to today as possible)
    finalPredictionDate = '2018-12-19'

    # Some functions were built to allow for predictions n days into the future,
    # but 1 should be the most accurate and what the trader looks to use
    period = 1

    # Current data set (used for indicator values for current predictions)
    stockHistories = stockHistory(ticker, "2014-12-29", "2018-12-31", "Full")[0]

    # Historical data set (used for training the probability matrices)
    items = stockHistory(ticker, "1990-01-01", "2017-12-31", "Historical")
    # The actual financial data set
    pastHistory = items[0]
    # A transitions matrix for the emissions model
    transitions = items[2]
    conditionalProbabilityDistribution = jointAnalysis(pastHistory, period)

    # Predictions and accuracy
    conditionalCorrect = 0.0
    conditionalTotal = 0.0
    transitionsCorrect = 0.0
    transitionsTotal = 0.0
    naiveCorrect = 0.0
    naiveTotal = 0.0
    oppositeNaiveCorrect = 0
    oppositeNaiveTotal = 0
    randomCorrect = 0.0
    randomTotal = 0.0
    upCorrect = 0.0
    upTotal = 0.0
    downCorrect = 0.0
    downTotal = 0.0

    while predictionDate < datetime.datetime.strptime(finalPredictionDate, '%Y-%m-%d').date() + timedelta(days=1):
        if is_business_day(predictionDate):
            conditionalPredictionTuple = predictor(stockHistories, conditionalProbabilityDistribution, period, predictionDate.strftime('%Y-%m-%d'), transitions, 0)
            transitionsPredictionTuple = predictor(stockHistories, conditionalProbabilityDistribution, period, predictionDate.strftime('%Y-%m-%d'), transitions, 1)
            naiveTuple = predictor(stockHistories, conditionalProbabilityDistribution, period, predictionDate.strftime('%Y-%m-%d'), transitions, 2)
            oppositeNaiveTuple = predictor(stockHistories, conditionalProbabilityDistribution, period, predictionDate.strftime('%Y-%m-%d'), transitions, 3)
            randomTuple = predictor(stockHistories, conditionalProbabilityDistribution, period, predictionDate.strftime('%Y-%m-%d'), transitions, 4)
            upTuple = predictor(stockHistories, conditionalProbabilityDistribution, period, predictionDate.strftime('%Y-%m-%d'), transitions, 5)
            downTuple = predictor(stockHistories, conditionalProbabilityDistribution, period, predictionDate.strftime('%Y-%m-%d'), transitions, 6)
            conditionalCorrect += conditionalPredictionTuple[0]
            conditionalTotal += conditionalPredictionTuple[1]
            transitionsCorrect += transitionsPredictionTuple[0]
            transitionsTotal += transitionsPredictionTuple[1]
            naiveCorrect += naiveTuple[0]
            naiveTotal += naiveTuple[1]
            oppositeNaiveCorrect += oppositeNaiveTuple[0]
            oppositeNaiveTotal += oppositeNaiveTuple[1]
            randomCorrect += randomTuple[0]
            randomTotal += randomTuple[1]
            upCorrect += upTuple[0]
            upTotal += upTuple[1]
            downCorrect += downTuple[0]
            downTotal += downTuple[1]
        predictionDate = next_weekday(predictionDate, period)
    accuracyDict = {}

    print("Conditional Probability Prediction Accuracy: ")
    print("Correct: " + str(conditionalCorrect))
    print("Total: " + str(conditionalTotal))
    print("Accuracy: " + str(conditionalCorrect/conditionalTotal))
    accuracyDict["Conditional Probability"] = ["Correct: " + str(conditionalCorrect), "Total: " + str(conditionalTotal), "Accuracy: " + str(conditionalCorrect/conditionalTotal)]
    print("Emissions Prediction Accuracy: ")
    print("Correct: " + str(transitionsCorrect))
    print("Total: " + str(transitionsTotal))
    print("Accuracy: " + str(transitionsCorrect/transitionsTotal))
    accuracyDict["Emissions"] = ["Correct: " + str(transitionsCorrect), "Total: " + str(transitionsTotal), "Accuracy: " + str(transitionsCorrect/transitionsTotal)]
    print("Naive Prediction Accuracy: ")
    print("Correct: " + str(naiveCorrect))
    print("Total: " + str(naiveTotal))
    print("Accuracy: " + str(naiveCorrect/naiveTotal))
    accuracyDict["Naive"] = ["Correct: " + str(naiveCorrect), "Total: " + str(naiveTotal), "Accuracy: " + str(naiveCorrect/naiveTotal)]
    print("Opposite Naive Prediction Accuracy: ")
    print("Correct: " + str(oppositeNaiveCorrect))
    print("Total: " + str(oppositeNaiveTotal))
    print("Accuracy: " + str(naiveCorrect/naiveTotal))
    accuracyDict["Opposite"] = ["Correct: " + str(oppositeNaiveCorrect), "Total: " + str(oppositeNaiveTotal), "Accuracy: " + str(oppositeNaiveCorrect/oppositeNaiveTotal)]
    print("Random Prediction Accuracy: ")
    print("Correct: " + str(randomCorrect))
    print("Total: " + str(randomTotal))
    print("Accuracy: " + str(randomCorrect/randomTotal))
    accuracyDict["Random"] = ["Correct: " + str(randomCorrect), "Total: " + str(randomTotal), "Accuracy: " + str(randomCorrect/randomTotal)]
    print("Up Prediction Accuracy: ")
    print("Correct: " + str(upCorrect))
    print("Total: " + str(upTotal))
    print("Accuracy: " + str(upCorrect/upTotal))
    accuracyDict["Up"] = ["Correct: " + str(upCorrect), "Total: " + str(upTotal), "Accuracy: " + str(upCorrect/upTotal)]
    print("Down Prediction Accuracy: ")
    print("Correct: " + str(downCorrect))
    print("Total: " + str(downTotal))
    print("Accuracy: " + str(downCorrect/downTotal))
    accuracyDict["Down"] = ["Correct: " + str(downCorrect), "Total: " + str(downTotal), "Accuracy: " + str(downCorrect/downTotal)]
    pd.DataFrame([accuracyDict], index = [0]).to_csv("DataFiles/" + ticker + ' Accuracy Log With R, S, M, C, V, RT, MT, 12-26EMA.csv', index=False)

testing()
