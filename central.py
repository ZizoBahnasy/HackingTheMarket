from pandas_datareader import data as pdr
import pandas as pd

import fix_yahoo_finance as yf, numpy as np
yf.pdr_override() # <== that's all it takes :-)

pd.options.mode.chained_assignment = None

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
    positiveSums = sum(stock['pctChange'][i] for i in range(period) if stock['pctChange'][i] > 0)
    negativeSums = sum(stock['pctChange'][i] for i in range(period) if stock['pctChange'][i] < 0)
    for i in range(period, len(stock.index)):
        avgGain = sum(stock['pctChange'][i - j] for j in range(period, 0, -1) if stock['pctChange'][i - j] > 0)/period
        avgLoss = -1 * sum(stock['pctChange'][i - j] for j in range(period, 0, -1) if stock['pctChange'][i - j] < 0)/period
        stock['RSI'][i] = 100 - (100/(1 + avgGain/avgLoss))
        # stock['RSI'][i] = avgLoss
    # for i in range(len(stock.index)):
        # if i < period:
        #     sum += stock['pctChange']
        # else:
    return stock
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

ticker = "BABA"
# download dataframe
# data = pdr.get_data_yahoo("BABA", start="2018-01-01", end="2018-11-26")
stock = pdr.get_data_yahoo(ticker, start="2010-01-01", end="2018-12-07")
# stock = pdr.get_data_yahoo("TSLA").loc["2018"]

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

for i in range(0, len(stock.index)):
    # test.append(stock.iloc[i]['pctChange'])
    result = convert(stock.iloc[i]['pctChange'], probabilities)
    stock['delta'][i] = result[0]
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


# # Currently calculating probability of same key[0] given key[1] (sums to 1)
# newSum = 0
# for key in [('Very Much Up', 'Very Much Up'), ('Very Much Up', 'Up'), ('Very Much Up', 'Neutral'), ('Very Much Up', 'Down'), ('Very Much Up', 'Very Much Down')]:
#     newSum += transitionsDict[key]
# for key in [('Very Much Up', 'Very Much Up'), ('Very Much Up', 'Up'), ('Very Much Up', 'Neutral'), ('Very Much Up', 'Down'), ('Very Much Up', 'Very Much Down')]:
#     if newSum != 0:
#         transitionsDict[key] = transitionsDict[key]/newSum
#     else:
#         transitionsDict[key] = 0
#
# newSum = 0
# for key in [('Up', 'Very Much Up'), ('Up', 'Up'), ('Up', 'Neutral'), ('Up', 'Down'), ('Up', 'Very Much Down')]:
#     newSum += transitionsDict[key]
# for key in [('Up', 'Very Much Up'), ('Up', 'Up'), ('Up', 'Neutral'), ('Up', 'Down'), ('Up', 'Very Much Down')]:
#     if newSum != 0:
#         transitionsDict[key] = transitionsDict[key]/newSum
#     else:
#         transitionsDict[key] = 0
#
# newSum = 0
# for key in [('Neutral', 'Very Much Up'), ('Neutral', 'Up'), ('Neutral', 'Neutral'), ('Neutral', 'Down'), ('Neutral', 'Very Much Down')]:
#     newSum += transitionsDict[key]
# for key in [('Neutral', 'Very Much Up'), ('Neutral', 'Up'), ('Neutral', 'Neutral'), ('Neutral', 'Down'), ('Neutral', 'Very Much Down')]:
#     if newSum != 0:
#         transitionsDict[key] = transitionsDict[key]/newSum
#     else:
#         transitionsDict[key] = 0
#
# newSum = 0
# for key in [('Down', 'Very Much Up'), ('Down', 'Up'), ('Down', 'Neutral'), ('Down', 'Down'), ('Down', 'Very Much Down')]:
#     newSum += transitionsDict[key]
# for key in [('Down', 'Very Much Up'), ('Down', 'Up'), ('Down', 'Neutral'), ('Down', 'Down'), ('Down', 'Very Much Down')]:
#     if newSum != 0:
#         transitionsDict[key] = transitionsDict[key]/newSum
#     else:
#         transitionsDict[key] = 0
#
# newSum = 0
# for key in [('Very Much Down', 'Very Much Up'), ('Very Much Down', 'Up'), ('Very Much Down', 'Neutral'), ('Very Much Down', 'Down'), ('Very Much Down', 'Very Much Down')]:
#     newSum += transitionsDict[key]
# for key in [('Very Much Down', 'Very Much Up'), ('Very Much Down', 'Up'), ('Very Much Down', 'Neutral'), ('Very Much Down', 'Down'), ('Very Much Down', 'Very Much Down')]:
#     if newSum != 0:
#         transitionsDict[key] = transitionsDict[key]/newSum
#     else:
#         transitionsDict[key] = 0


newSum = 0
for key in [('Very Much Up', 'Very Much Up'), ('Up', 'Very Much Up'), ('Neutral', 'Very Much Up'), ('Down', 'Very Much Up'), ('Very Much Down', 'Very Much Up')]:
    newSum += transitionsDict[key]
for key in [('Very Much Up', 'Very Much Up'), ('Up', 'Very Much Up'), ('Neutral', 'Very Much Up'), ('Down', 'Very Much Up'), ('Very Much Down', 'Very Much Up')]:
    if newSum != 0:
        transitionsDict[key] = transitionsDict[key]/newSum
    else:
        transitionsDict[key] = 0

newSum = 0
for key in [('Very Much Up', 'Up'), ('Up', 'Up'), ('Neutral', 'Up'), ('Down', 'Up'), ('Very Much Down', 'Up')]:
    newSum += transitionsDict[key]
for key in [('Very Much Up', 'Up'), ('Up', 'Up'), ('Neutral', 'Up'), ('Down', 'Up'), ('Very Much Down', 'Up')]:
    if newSum != 0:
        transitionsDict[key] = transitionsDict[key]/newSum
    else:
        transitionsDict[key] = 0

newSum = 0
for key in [('Very Much Up', 'Neutral'), ('Up', 'Neutral'), ('Neutral', 'Neutral'), ('Down', 'Neutral'), ('Very Much Down', 'Neutral')]:
    newSum += transitionsDict[key]
for key in [('Very Much Up', 'Neutral'), ('Up', 'Neutral'), ('Neutral', 'Neutral'), ('Down', 'Neutral'), ('Very Much Down', 'Neutral')]:
    if newSum != 0:
        transitionsDict[key] = transitionsDict[key]/newSum
    else:
        transitionsDict[key] = 0

newSum = 0
for key in [('Very Much Up', 'Down'), ('Up', 'Down'), ('Neutral', 'Down'), ('Down', 'Down'), ('Very Much Down', 'Down')]:
    newSum += transitionsDict[key]
for key in [('Very Much Up', 'Down'), ('Up', 'Down'), ('Neutral', 'Down'), ('Down', 'Down'), ('Very Much Down', 'Down')]:
    if newSum != 0:
        transitionsDict[key] = transitionsDict[key]/newSum
    else:
        transitionsDict[key] = 0

newSum = 0
for key in [('Very Much Up', 'Very Much Down'), ('Up', 'Very Much Down'), ('Neutral', 'Very Much Down'), ('Down', 'Very Much Down'), ('Very Much Down', 'Very Much Down')]:
    newSum += transitionsDict[key]
for key in [('Very Much Up', 'Very Much Down'), ('Up', 'Very Much Down'), ('Neutral', 'Very Much Down'), ('Down', 'Very Much Down'), ('Very Much Down', 'Very Much Down')]:
    if newSum != 0:
        transitionsDict[key] = transitionsDict[key]/newSum
    else:
        transitionsDict[key] = 0




# stock['delta'].apply(convert)

# print(probabilities)
# if daily_pct_change.item() > 0.03:
#     stock['delta'] = "Very Large"
# Inspect daily returns
print("Stock: ")
print(ticker)
print("Probability Matrix (VMU, U, N, D, VMD): ")
print(probabilities)
print(probabilitiesPct)

print("Transition Record:")
print(transitions)
print(transitionsDict)

stock=ema(stock, 12, 'EMA')
stock=ema(stock, 26, 'EMA')
stock=macd(stock)
stock=ema(stock, 9, 'MACD')
stock=rsi(stock, 14)

print(stock)
#
# # Daily log returns
# daily_log_returns = np.log(daily_close.pct_change()+1)
#
# # Print daily log returns
# print(daily_log_returns)

# Shift percentage change method
# # Daily returns
# daily_pct_change = daily_close / daily_close.shift(1) - 1
#
# # Print `daily_pct_change`
# print(daily_pct_change)
#
# def convert(percentages):
#     for percentage in percentages:
#         if percentage > 0.03:
#             return "Very Much Up"
#         if percentage <= 0.03 and percentage > 0:
#             return "Up"
#         if percentage >= -0.03 and percentage < 0:
#             return "Down"
#         if percentage < -0.03:
#             return "Very Much Down"
#         else:
#             return "Neutral"
