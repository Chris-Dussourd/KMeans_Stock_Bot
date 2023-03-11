"""
Functions:
    calc_features - calculate the features for the newest set of data
    find_strong_buys - find which stocks currently have feature data in a strong buy centroid

CD 12/2020 - Created
CD 01/2021 - Update calc_features to store all feature values
"""

import pandas as pd
import numpy as np

def calc_features(transactions,tickers_list):
    """
    Calculate all the features for the current data and store in transactions[ticker]['Features'].
    Store only the features used for calculating centroid data in transactions[ticker]['Features Scaled']

    transactions - dictionary that contains current data for stocks and features list
        transactions['Features List'] - list of features that are needed to calculate the feature list
        transactions[ticker] - dictionary of data for each ticker with a 'Price', 'Bid Price', 'Ask Price', 'Minute Prices', and 'Daily Prices' key
        transactions[ticker]['Feature Scaling'] - contains a DataFrame the scaling variables for features (index = ['min','max'])
    tickers_list - list of tickers to calculate features

    returns - transactions with an updated DataFrame row at index 0 in transactions[ticker]['Features'] and centroid features in transactions[ticker]['Features Scaled']
               
            
    """
    new_data = pd.Series(columns = transactions['Features List'])
    for ticker in tickers_list:

        #Use a bias parameter as a feature
        new_data.loc[0,'Bias'] = 1

        #Add current price of ticker output
        new_data.loc[0,'close'] = transactions[ticker]['Price']

        #Add close price of ticker output 1 minute ago
        new_data.loc[0,'close1min'] = transactions[ticker]['Minute Prices']['close'].iloc[-2]

        #Add close price of ticker output 2 mintues ago
        new_data.loc[0,'close2min'] = transactions[ticker]['Minute Prices']['close'].iloc[-3]

        #Add close price of ticker output 5 mintues ago
        new_data.loc[0,'close5min'] = transactions[ticker]['Minute Prices']['close'].iloc[-5]

        #Add close price of ticker output 10 mintues ago
        new_data.loc[0,'close10min'] = transactions[ticker]['Minute Prices']['close'].iloc[-11]

        #Add close price of ticker output 30 mintues ago
        new_data.loc[0,'close30min'] = transactions[ticker]['Minute Prices']['close'].iloc[-31]

        #Add close price of ticker output 1 hour ago
        new_data.loc[0,'close1hr'] = transactions[ticker]['Minute Prices']['close'].iloc[-61]

        #Add difference the bid and ask price as an estimate
        new_data.loc[0,'DiffHighLow'] = transactions[ticker]['Ask Price'] - transactions[ticker]['Bid Price']

        #Add 5 day simple moving average
        new_data.loc[0,'SMA5'] = calc_sma(transactions[ticker]['Price'],transactions[ticker]['Daily Prices']['close'],5)

        #Add 10 day simple moving average
        new_data.loc[0,'SMA10'] = calc_sma(transactions[ticker]['Price'],transactions[ticker]['Daily Prices']['close'],10)

        #Add 20 day simple moving average
        new_data.loc[0,'SMA20'] = calc_sma(transactions[ticker]['Price'],transactions[ticker]['Daily Prices']['close'],20)
    
        #Add 50 day simple moving average
        new_data.loc[0,'SMA50'] = calc_sma(transactions[ticker]['Price'],transactions[ticker]['Daily Prices']['close'],50)

        #Add 5 day exponential moving average
        new_data.loc[0,'EMA5'] = calc_ema(transactions[ticker]['Price'],transactions[ticker]['Features']['EMA5'][1],5)

        #Add 10 day exponential moving average
        new_data.loc[0,'EMA10'] = calc_ema(transactions[ticker]['Price'],transactions[ticker]['Features']['EMA10'][1],10)

        #Add 20 day exponential moving average
        new_data.loc[0,'EMA20'] = calc_ema(transactions[ticker]['Price'],transactions[ticker]['Features']['EMA20'][1],20)
    
        #Add 50 day exponential moving average
        new_data.loc[0,'EMA50'] = calc_ema(transactions[ticker]['Price'],transactions[ticker]['Features']['EMA50'][1],50)


        #Calculate the 20 day and 50 day standard deviation of the output ticker
        std20 = calc_std(transactions[ticker]['Price'],transactions[ticker]['Daily Prices']['close'],new_data.loc[0,'SMA20'],20)
        std50 = calc_std(transactions[ticker]['Price'],transactions[ticker]['Daily Prices']['close'],new_data.loc[0,'SMA50'],50)

        #Add 20 day Bollinger Percent, Note: we are using 2*standard deviation to get the lower and upper band
        boll_lower_20 = new_data.loc[0,'SMA20']-2*std20
        boll_upper_20 = new_data.loc[0,'SMA20']+2*std20
        boll_percent_20 = (transactions[ticker]['Price']-boll_lower_20)/(boll_upper_20-boll_lower_20)*100
        new_data.loc[0,'Boll20'] = boll_percent_20

        #Add 50 day Bollinger Percent, Note: we are using 2.1*standard deviation to get the lower and upper band
        boll_lower_50 = new_data.loc[0,'SMA50']-2.1*std50
        boll_upper_50 = new_data.loc[0,'SMA50']+2.1*std50
        boll_percent_50 = (transactions[ticker]['Price']-boll_lower_50)/(boll_upper_50-boll_lower_50)*100
        new_data.loc[0,'Boll50'] = boll_percent_50

        #Calculate the MACD (Moving Average Convergence/Divergence Oscillator)
        [macd_line,signal_line,macd_hist] = calc_macd(transactions[ticker]['Price'],transactions[ticker]['Features']['EMA5'][1],transactions[ticker]['Features']['SignalLine'][1])

        #Add MACD line
        new_data.loc[0,'MACDline'] = macd_line

        #Add Signal line (9 day EMA of MACD line)
        new_data.loc[0,'SignalLine'] = signal_line

        #Add MACD histogram (MACD_line - Signal line)
        new_data.loc[0,'MACDhist'] = macd_hist

        #Add Relative Strength Index
        new_data.loc[0,'RSI'] = calc_rsi(transactions[ticker]['Price'],transactions[ticker]['Daily Prices']['close'])

        #Get the 14 day stochastic oscillator
        (stochastic_k_line,stochastic_d_line,_s) = calc_stochastic_oscillator(transactions[ticker]['Price'],transactions[ticker]['Daily Prices'],14,\
        transactions[ticker]['Features']['Stochastic%K'],transactions[ticker]['Features']['Stochastic%K']-transactions[ticker]['Features']['Stochastic%K-%D'])

        #Add %K of the Stochastic Oscillator
        new_data.loc[0,'Stochastic%K'] = stochastic_k_line

        #Add difference between the %K and %D of the Stochastic Oscillator
        new_data.loc[0,'Stochastic%K-%D'] = stochastic_k_line-stochastic_d_line

        #Add the new data to index 0 (0 days in past) in 'Features'
        transactions[ticker]['Features'].loc[0] = new_data.copy(deep=True)

        #Create a new DataFrame that only keeps the features used for centroid cluster assignment
        centroid_features = new_data.copy(deep=True)
        drop_columns = []
        for col in centroid_features:
            if col not in transactions['Features List']:
                drop_columns.append(col)
        centroid_features.drop(columns=drop_columns,inplace=True)

        #Add the scaled data to 'Feautures Scaled'
        transactions[ticker]['Features Scaled'] = (centroid_features-transactions[ticker]['Feature Scaling'].loc['min'])/(transactions[ticker]['Feature Scaling'].loc['max']-transactions[ticker]['Feature Scaling'].loc['min'])



def find_strong_buys(transactions,tickers_list,num_buys):
    """
    Find the current stock feature data that belong to cluster centroids with a strong buy label
    Calculate the Euclidean distance between current data and all centroids. Find the centroid with the minimum distance.

    transactions - dictionary of feature data 
        transactions[ticker]['Features'] - DataFrame of feature data that must contain row index 0 for most recent data
        transactions[ticker]['Thresholds'] - list of the thresholds that divide the clusters into different categories (strong buy uses the last two thresholds)
        transactions[ticker]['Centroids'] - a dictionary where each key is a centroid and the values are feature data that represent the center of a cluster
        transactions[ticker]['Centroid Labels'] - a pandas Series where each index is a centroid number and the values are a label (Strong Buy, Buy, Weak Buy, Neutral, Weak Sell, Sell, Strong Sell)
    tickers_list - the list of tickers to find strong buys
    num_buys - number of buys that can be purchased based on available balance

    returns buy_list - a list of tickers that are in strong buy clusters
    """
    buy_list = []

    for ticker in tickers_list:

        #pandas Series that contains the distance values between each centroid and current feature data
        dist_centroid = pd.Series(index=range(len(transactions[ticker]['Centroids'])))

        #Find the euclidean distance between current features and centroids
        for k in range(len(transactions[ticker]['Centroids'])):
            dist_centroid[k] = np.sqrt(sum(np.square(transactions[ticker]['Features Scaled']-transactions[ticker]['Centroids'][k])))

        #Assign more recent data to a cluster
        assign_centroid = dist_centroid.idxmin()

        #Add it to buy list if it belongs to a strong buy cluster
        if transactions[ticker]['Centroid Labels'].loc[assign_centroid] == 'Strong Buy':
            buy_list.append(ticker)

    #If there is a limit to the number of buys available, only place orders for most volatile tickers (greatest chance of large profit)
    while len(buy_list)>num_buys:
        top_threshold = pd.Series(index=buy_list)
        for ticker in buy_list:
            top_threshold.loc[ticker] = transactions[ticker]['Thresholds'][-1]

        #Remove lowest percent gain ticker
        buy_list.remove(top_threshold.idxmin())

    return buy_list


def get_centroid_label(transactions,ticker):
    """
    Find the cluster centroid the current stock data belongs to and get the label of the centroid.
    Calculate the Euclidean distance between current data and all centroids. Find the centroid with the minimum distance.

    transactions - dictionary of feature data 
        transactions[ticker]['Features'] - DataFrame of feature data that must contain row index 0 for most recent data
        transactions[ticker]['Thresholds'] - list of the thresholds that divide the clusters into different categories (strong buy uses the last two thresholds)
        transactions[ticker]['Centroids'] - a dictionary where each key is a centroid and the values are feature data that represent the center of a cluster
        transactions[ticker]['Centroid Labels'] - a pandas Series where each index is a centroid number and the values are a label (Strong Buy, Buy, Weak Buy, Neutral, Weak Sell, Sell, Strong Sell)
    ticker - the ticker to obtain the centroid label for

    returns centroid_label - centroid label for ticker's current data
    """

    #pandas Series that contains the distance values between each centroid and current feature data
    dist_centroid = pd.Series(index=range(len(transactions[ticker]['Centroids'])))

    #Find the euclidean distance between current features and centroids
    for k in range(len(transactions[ticker]['Centroids'])):
        dist_centroid[k] = np.sqrt(sum(np.square(transactions[ticker]['Features Scaled']-transactions[ticker]['Centroids'][k])))

    #Assign most recent data to a cluster
    assign_centroid = dist_centroid.idxmin()

    #Return the centroid label for most recent stock data
    return transactions[ticker]['Centroid Labels'].loc[assign_centroid]


def calc_sma(price,prices_daily,period):
    """
    Calculate the simple moving average for the current price.
    The period represents the number of days to use in the average.

    price - most recent price to use to calculate the SMA
    prices_daily - pandas Series of day level prices (where index number is the number of days in the past)
    period - the number of days to include in the simple moving average

    returns the simple moving average
    """
    sma = price

    for days_past in range(1,period):
        sma = sma + prices_daily[days_past]
    sma = sma/period

    return sma


def calc_ema(price,ema_yesterday,period):
    """
    Calculate the exponential moving average for the current price.
    Example: EMA = (Current_Price) * Multiplier + EMA_yesterday * (1-multiplier)
                  multiplier = 2/(1+period)

    price - most recent price to use to calculate the EMA
    ema_yesterday - the exponential moving average based on yesterday's close price
    period - the number of days to use in EMA multiplier

    Returns the ema
    """
    mult = 2/(1+period) #Multiplier
    #Calculate the EMA value
    ema = price*mult + ema_yesterday*(1-mult)

    return ema


def calc_std(price,prices_daily,sma,period):
    """
    Calculate the standard deviation for the current price using the daily prices
    The period represents the number of days to use in calculating the standard deviation
    The daily closing prices must have data for at least period days before the time data starts.
        std = (sum((close_price - SMA)^2)/20)^1/2  

    price - most recent price to use to calculate the STD
    prices_daily - pandas Series of day level prices (where index number is the number of days in the past)
    period - the number of days to include in the simple moving average
    sma - the simple moving average of the stock over the same period
    period - the number of days to include in the standard deviation

    returns the standard deviation
    """
    std = (price-sma)**2

    for days_past in range(1,period):
        std = std + (prices_daily[days_past]-sma)**2

    #std = sqrt(sum((x-mu).^2)/N)    
    std = (std/period)**0.5

    return std


def calc_macd(price,ema_yesterday,signal_line_yesterday):
    """
    Calculate the moving average convergence/divergence oscillator (MACD) for the current price.

    price - most recent price to use to calculate the MACD
    ema_yesterday - the exponential moving average based on yesterday's close price
    signal_line_yesterday - the signal line from yesterday (9 day EMA of MACD line)

    Returns the MACD line, the signal line, and the MACD histogram
        macd_line = EMA_12days - EMA_26days (using price and yesterday's ema)
        signal_line = EMA_9days (using macd_line and yesterday's signal line)
        macd_histogram = macd_line - signal_line

    """
    #MACD line is 12 day EMA of price - 26 day EMA of price
    macd_line = calc_ema(price,ema_yesterday,12)-calc_ema(price,ema_yesterday,26)

    #Signal line is 9 day EMA of MACD line
    signal_line = calc_ema(macd_line,signal_line_yesterday,9)

    #The MACD histogram is the macd_line minus the signal_line
    macd_histogram = macd_line - signal_line

    return (macd_line,signal_line,macd_histogram)



def calc_rsi(price,prices_daily):
    """
    Calculate the Relative Stregnth Index (RSI) for current price.
    
    price - most recent price to use to calculate the RSI
    prices_daily - pandas Series of day level prices (where index number is the number of days in the past)

    Returns the RSI for each point using the time_data
    We caculate the RSI in two parts:
    Part 1:
        average_gain_first = (sum of the gains between closing prices over 14 days)/14  - gain = 0 for days that closed lower than the previous day
        average_loss_first = (sum of the losses between closing prices over 14 days)/14 - loss = 0 for days that closed higher than the previous day
    
    Part 2:
        average_gain = (average_gain_prev*13+current_gain)/14 where average_gain_prev is the average gain of the previous day
        average_loss = (average_loss_prev*13+current_loss)/14 where average_loss_prev is the average loss of the previous day

    RSI = 100 - 100/(1+average_gain/average_loss)

    """
    #Get the gains and losses for all days in prices_daily
    price_diff = prices_daily - prices_daily.shift(periods=-1)
    price_diff[0] = price-price_daily[1] #Update first value to use current price
    gains = price_diff.where(price_diff>0,0) #Gains from previous day
    losses = price_diff.where(price_diff<0,0) #Losses from previous day

    #Part 1 calculate the first average gain and loss (end values in series are furthest in the past)
    ave_gain = sum(gains[len(prices_daily)-15:len(prices_daily)-1])
    ave_loss = sum(losses[len(prices_daily)-15:len(prices_daily)-1])

    #Part 2 Calculate each average gain/loss based on previous average gain/loss and current gain/loss
    for period in range(len(prices_daily)-16,-1,-1):
        ave_gain = (ave_gain*13+gains[period])/14
        ave_loss = (ave_loss*13+losses[period])/14

    rsi = 100 - 100/(1 + ave_gain/ave_loss)

    return rsi


def calc_stochastic_oscillator(price,prices_daily_df,period,prev_k_line,prev_d_line):
    """
    Calculate the Stochastic Oscillator for the current price using the daily high/low prices and the last available price.
    
    price - most recent price to use to calculate the stochastic oscillator
    prices_daily_df - DataFrame of day level data that contains column names 'High' and 'Low' 
    period - the number of days to include in the stochastic oscillator (typically 14 days)
    prev_k_line - series of previous k_line values (where index number is the number of days in the past)
    prev_d_line - series of previous d_line values (where index number is the number of days in the past)

    Returns the K_line and D_line of the stochastic oscillator
        K_line = %K = 100*(Current_Price - period day Low)/(period day High - period day Low) - Ex: 14 day high/low
        D_line = %D = 3 day simple moving average of %K
        D_line_slow = %D_slow = 3 day simple moving average of %D
    """
    #K_line = 100*(Current_Price - period day Low)/(period day High - period day Low)
    high = max(prices_daily_df['high'].iloc[-1:-period-1:-1])
    low = max(prices_daily_df['low'].iloc[-1:-period-1:-1])
    k_line = 100*(price-low)/(high-low)

    #D_line = 3 day simple moving average of k line
    d_line = (k_line + prev_k_line[1] + prev_k_line[2])/3

    #D_line_slow = 3 day simple moving average of d line
    d_line = (d_line + prev_d_line[1] + prev_d_line[2])/3

    return k_line,d_line,d_line_slow

        