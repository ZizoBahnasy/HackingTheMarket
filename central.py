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


import fix_yahoo_finance as yf, numpy as np
yf.pdr_override() # <== that's all it takes :-)

pd.options.mode.chained_assignment = None
US_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())

ticker = "BABA"

def ema(stock, period, code):
    if code == "EMA":
        if period < len(stock.index):
            for i in range(period, len(stock.index)):

                closingPrices = [stock['Adj Close'][i - j] for j in range(period, 0, -1)]

                sma = sum(closingPrices)/period

                multiplier = 2.0/(period + 1)
                stock['EMA' + str(period)][i] = sma

                if i != period:
                    stock['EMA' + str(period)][i] = (stock['Adj Close'][i] - stock['EMA' + str(period)][i - 1]) * multiplier + stock['EMA' + str(period)][i - 1]
            return stock
        else:
            for i in range(len(stock.index)):
                closingPrices = [stock['Adj Close'][j] for j in range(len(stock.index))]
                sma = sum(closingPrices)/len(stock.index)
                stock['EMA' + str(period)][i] = sma
            return stock
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
            return stock
        else:
            for i in range(len(stock.index)):
                macdList = [stock['MACD'][i - j] for j in range(period, 0, -1)]
                sma = sum(macdList)/period
                stock['MACD Signal'][i] = sma
            return stock



def macd(stock):
    for i in range(len(stock.index)):
        if stock['EMA12'][i] != 0 and stock['EMA26'][i] != 0:
            stock['MACD'][i] = stock['EMA12'][i] - stock['EMA26'][i]
        else:
            stock['MACD'][i] = 0
    return stock

def rsi(stock, period):
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

    return stock

def stochastic(stock, period):
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

def analyze(stock):

    rSum = 0
    sSum = 0
    mSum = 0
    dataDict = {}
    for i in range(5):
        dataDict['P(R = ' + str(i) + ')'] = [0.0]
    for i in range(5):
        dataDict['P(S = ' + str(i) + ')'] = [0.0]
    for i in range(2):
        dataDict['P(M = ' + str(i) + ')'] = [0.0]

    for i in range(26, len(stock.index)):
        if stock["RSI"][i] < 20.0:
            dataDict['P(R = 0)'][0] += 1.0
            rSum += 1
        elif stock["RSI"][i] < 40.0:
            dataDict['P(R = 1)'][0] += 1.0
            rSum += 1
        elif stock["RSI"][i] < 60.0:
            dataDict['P(R = 2)'][0] += 1.0
            rSum += 1
        elif stock["RSI"][i] < 80.0:
            dataDict['P(R = 3)'][0] += 1.0
            rSum += 1
        else:
            dataDict['P(R = 4)'][0] += 1.0
            rSum += 1
        if stock["Stochastic"][i] < 20.0:
            dataDict['P(S = 0)'][0] += 1.0
            sSum += 1
        elif stock["Stochastic"][i] < 40.0:
            dataDict['P(S = 1)'][0] += 1.0
            sSum += 1
        elif stock["Stochastic"][i] < 60.0:
            dataDict['P(S = 2)'][0] += 1.0
            sSum += 1
        elif stock["Stochastic"][i] < 80.0:
            dataDict['P(S = 3)'][0] += 1.0
            sSum += 1
        else:
            dataDict['P(S = 4)'][0] += 1.0
            sSum += 1
        if stock['MACD'][i] >= stock['MACD Signal'][i]:
            dataDict['P(M = 1)'][0] += 1.0
            mSum += 1
        else:
            dataDict['P(M = 0)'][0] += 1.0
            mSum += 1

    for key, value in dataDict.items():
        if 'R' in key:
            value[0] = value[0] / rSum
        elif 'S' in key:
            value[0] = value[0] / sSum
        else:
            value[0] = value[0] / mSum

    df = pd.DataFrame(data=dataDict)
    df.to_csv(ticker + "marginalProbabilities.csv",index=False)
    return dataDict

def jointAnalysis(stock, daysAhead):
    joints = []
    partialJoints = []
    for i in range(26, len(stock.index) - daysAhead):
        joints.append((stock['delta'][i + daysAhead], convertRSI(stock['RSI'][i]), convertRSI(stock['Stochastic'][i]), convertMACDSignal(stock['MACD'][i] - stock['MACD Signal'][i])))
        partialJoints.append((convertRSI(stock['RSI'][i]), convertRSI(stock['Stochastic'][i]), convertMACDSignal(stock['MACD'][i] - stock['MACD Signal'][i])))
    counter = collections.Counter(joints)
    partialCounter = collections.Counter(partialJoints)
    probabilities = {}
    partialProbabilities = {}
    for key, value in counter.items():
        value = (value * 1.0)/len(joints)
        probabilities[key] = value
    for key, value in partialCounter.items():
        value = (value * 1.0)/len(partialJoints)
        partialProbabilities[key] = value
    pd.DataFrame(probabilities, index = [0]).to_csv('fullJointDistribution.csv', index=False)
    pd.DataFrame(partialProbabilities, index = [0]).to_csv(ticker + 'partialJointDistribution.csv', index=False)
    return conditionalProbabilities(probabilities, partialProbabilities)

# Calculates conditional probability distributions with indicator independence assumption
def conditionalProbabilities2(jointDistribution, stock):
    conditionals = {}
    marginals = analyze(stock)
    # print(marginals["P(R = 3)"])
    for key, value in jointDistribution.items():
        rKey = "R = " + str(key[1])
        sKey = "S = " + str(key[2])
        mKey = "S = " + str(key[3])
        conditionals["P(D = " + str(key[0]) + " | " + rKey + ", " + sKey + ", " + mKey + ")"] = value/(marginals["P(" + rKey + ")"][0] * marginals["P(" + sKey + ")"][0] * marginals["P(" + mKey + ")"][0])
    pd.DataFrame(conditionals, index=[0]).to_csv('conditionalProbabilities2.csv', index=False)

def conditionalProbabilities(jointDistribution, partialJointDistribution):
    conditionals = {}
    for key, value in jointDistribution.items():
        rKey = "R = " + str(key[1])
        sKey = "S = " + str(key[2])
        mKey = "M = " + str(key[3])
        conditionals["P(D = " + str(key[0]) + " | " + rKey + ", " + sKey + ", " + mKey + ")"] = value/(partialJointDistribution[(key[1], key[2], key[3])])
    pd.DataFrame(conditionals, index=[0]).to_csv(ticker + 'conditionalProbabilities.csv', index=False)
    return dictSort(conditionals)

def dictSort(oldDict):
    sortedDict = {}
    for macd in range(2):
        for stochastic in range(5):
            for rsi in range(5):
                for delta in ["Very Much Up", "Up", "Neutral", "Down", "Very Much Down"]:
                    key = "P(D = " + delta + " | R = " + str(rsi) + ", S = " + str(stochastic) + ", M = " + str(macd) + ")"
                    if key in oldDict:
                        sortedDict[key] = oldDict[key]
    pd.DataFrame(sortedDict, index=[0]).to_csv(ticker + 'FinalConditionalProbabilities.csv', index=False)
    return sortedDict

def convertRSI(percentage):
    value = int(percentage * 0.05)
    if value == 5:
        value = 4
    return value

def convertMACDSignal(difference):
    if difference >= 0:
        return 1
    else:
        return 0
    # table['P(R = )']
    
# Converts continuous price data to discrete labels and builds out emission-to-emission
# probability matrix
def convert(percentage, probabilities):
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
    # adate -= timedelta(days=period)
    adate -= US_BUSINESS_DAY * period
    while not is_business_day(adate): # Mon-Fri are 0-4
        # adate -= timedelta(days=1)
        adate -= US_BUSINESS_DAY
    return adate

def next_weekday(adate, period):
    adate += timedelta(days=period)
    while not is_business_day(adate): # Mon-Fri are 0-4
        adate += timedelta(days=1)
    return adate

def is_business_day(date):
    return bool(len(pd.bdate_range(date, date)))

def stockHistory(symbol, startDate, endDate):
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


    stock['delta'] = "Test"
    stock['EMA12'] = 0.0
    stock['EMA26'] = 0.0
    stock['MACD'] = 0.0
    stock['MACD Signal'] = 0.0
    stock['RSI'] = 0.0
    stock['Stochastic'] = 0.0
    stock['Stochastic SMA'] = 0.0

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
        # test.append(stock.iloc[i]['pctChange'])
        result = convert(stock.iloc[i]['pctChange'], probabilities)
        stock['delta'][i] = result[0]
        classes.append(result[2])
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

            # Not sure why there is a leakage here
            # else:
                # print(stock['delta'][i - 1])
        probabilities = result[1]
        # stock.set_value(i, 'delta', convert(stock.iloc[i]['pctChange']))

    probabilitiesPct = []
    summed = probabilities[0] + probabilities[1] + probabilities[2] + probabilities[3] + probabilities[4]
    probabilitiesPct = [probabilities[0]/summed, probabilities[1]/summed, probabilities[2]/summed, probabilities[3]/summed, probabilities[4]/summed]

    stock=ema(stock, 12, 'EMA')
    stock=ema(stock, 26, 'EMA')
    stock=macd(stock)
    stock=ema(stock, 9, 'MACD')
    stock=rsi(stock, 14)
    stock=stochastic(stock, 14)

    # print(stock)
    stock.to_csv(symbol + ".csv")
    return (stock, classes)


def predictor(stock, conditionalProbabilityDistribution, period, date):
    dateDate = datetime.datetime.strptime(date, '%Y-%m-%d').date()
    while not is_business_day(dateDate):
        dateDate += 1
    date = dateDate
    yesterday = prev_weekday(date, period).strftime('%Y-%m-%d')
    print(yesterday)
    if yesterday in stock['RSI']:
        rKey = convertRSI(stock['RSI'][yesterday])
        sKey = convertRSI(stock['Stochastic'][yesterday])
        mKey = convertMACDSignal(stock['MACD'][yesterday] - stock['MACD Signal'][yesterday])


        choices = []

        for delta in ["Very Much Up", "Up", "Neutral", "Down", "Very Much Down"]:
            key = "P(D = " + delta + " | R = " + str(rKey) + ", S = " + str(sKey) + ", M = " + str(mKey) + ")"
            if key in conditionalProbabilityDistribution:
                choices.append((delta, conditionalProbabilityDistribution[key]))
            else:
                choices.append((delta, 0.2))

        prediction = max(choices, key = lambda x: x[1])[0]
        print("Prediction: The price will move " + prediction + " on " + date.strftime('%Y-%m-%d'))
        if date.strftime('%Y-%m-%d') in stock['RSI']:
            realAction = stock['delta'][date.strftime('%Y-%m-%d')]
            print("True action: " + realAction)
            # if prediction in realAction:

            if 'Up' in prediction and 'Up' in realAction:
                return (1.0, 1.0)
            elif 'Down' in prediction and 'Down' in realAction:
                return (1.0, 1.0)
            elif prediction in realAction:
                return (1.0, 1.0)
            else:
                return (0.0, 1.0)
        else:
            return (0.0, 0.0)
    else:
        print("No prediction")
        return (0.0, 0.0)

def testing():
    startDate = '2018-01-01'
    predictionDateString = startDate
    predictionDate = datetime.datetime.strptime(predictionDateString, '%Y-%m-%d').date()
    finalPredictionDate = '2018-12-11'
    period = 1
    stockHistories = stockHistory(ticker, "2010-01-01", "2018-12-31")[0]
    pastHistory = stockHistory(ticker, "2010-01-01", "2017-12-31")[0]
    conditionalProbabilityDistribution = jointAnalysis(pastHistory, period)

    # if predictionDate.weekday() <= 4:
    correct = 0.0
    total = 0.0
    while predictionDate < datetime.datetime.strptime(finalPredictionDate, '%Y-%m-%d').date() + timedelta(days=1):
        if is_business_day(predictionDate):
            correct += predictor(stockHistories, conditionalProbabilityDistribution, period, predictionDate.strftime('%Y-%m-%d'))[0]
            total += predictor(stockHistories, conditionalProbabilityDistribution, period, predictionDate.strftime('%Y-%m-%d'))[1]
        predictionDate = next_weekday(predictionDate, period)
    print(correct)
    print(total)
    print(correct/total)

predictionDate = "2018-01-02"
forecastingPeriod = 1
endDate = prev_weekday(datetime.datetime.strptime(predictionDate, '%Y-%m-%d').date(), forecastingPeriod)
print(endDate)
# stock = pdr.get_data_yahoo(ticker, start="2010-01-01", end=endDate)

testing()

""" Naives Bayes Classification """

stock, classes = stockHistory(ticker, "2010-01-01", "2018-12-31")

# drop high, low, close, and EMA columns since not useful for NB
optstock = stock.drop(['High','Low','Close','delta','EMA12','EMA26'], axis=1)
# optstock = optstock.drop(optstock.index[0:30])

# Delete last row since can have NaN values
optstock=optstock.drop(optstock.index[-1])
classes = classes[:-1]

# get 80% index to divide data into training and testing sets for fitting
cutat = int(len(optstock.index) / 10) * 8
trainingX=optstock.drop(optstock.index[cutat:])
testX=optstock.drop(optstock.index[0:cutat])
# trainingX = optstock[:cutat]
# testX = optstock[cutat:]

trainingclasses = classes[:cutat]
testclasses = classes[cutat:]
# print(len(testclasses))
# print(stock.head())

# Use sklearn if possible to fit the model, test it, and get its accuracy. If not, implement our own NB and CPTs.
NB = GaussianNB()
# print(optstock)
# print(trainingX)
# print(trainingclasses)
NB.fit(trainingX, trainingclasses)
predictedclasses = NB.predict(testX)

print("Accuracy:")
print (metrics.accuracy_score(testclasses, predictedclasses))

# If we implement the NB on our own:
# 1) Training: count the number of rows with each of the 5 perc_change classes
#              build a dictionary with all the possible values/intervals for each feature (that requires discretizing the features as well)
#              count the occurences of each features interval in each perc_change class
#              fit the model using the same equation as in q5 of NaiveBayes.py
#              test it on (testX, predictedclasses) in the same way as q6


##$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$##
