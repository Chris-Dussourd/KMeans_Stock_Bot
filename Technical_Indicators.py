"""
Functions that calculate stock technical indicators. 

Fuctions:
    calc_sma(price_minute_df,price_daily_df,column_name,period) - Simple Moving Average (SMA)
    calc_ema(price_minute_df,price_daily_df,column_name,period) - Exponential Moving Average (EMA)
    calc_std(price_minute_df,price_daily_df,column_name,sma,period) - Standard Deviation (STD)
    calc_macd(price_minute_df,price_daily_df,column_name) - Convergence/Divergence Oscillator (MACD)
    calc_rsi(price_minute_df,price_daily_df,column_name) - Relative Strength Index (RSI)
    calc_stochastic_oscillator(price_minute_df,price_daily_df,ticker,period) -Stochastic Oscillator (%K, %D) 

CD 12/2020  - Update to take DataFrames as input instead of Numpy arrays. 
CD 01/2021  - Update functions to return daily level values (not just minute level)

"""


import numpy as np
import pandas as pd
import datetime


def calc_sma(price_minute_df,price_daily_df,column_name,period):
    """
    Calculate the simple moving average for each datetime using the daily closing prices and the last available price
    The period represents the number of days to use in the average
    The daily closing prices must have data for at least period days before the price_minute_df['datetimte'] starts.
     For example: if we are calculating the 20 day simple moving average (period = 20), the daily_close values must start 20 days before the price_minute_df.
                  Otherwise, we would not be able to calculate the SMA for the first data point in time data.
                  SMA = (sum(close price on previous 19 days) + current_price)/20

    price_minute_df - DataFrame of minute level data with 'Datetime' and 'Date' as a column name
    price_daily_df - DataFrame of day level data with 'Datetime' and 'Date' as a column name
    column_name - the column name of the DataFrames used to calculate the SMA
    period - the number of days to include in the simple moving average

    returns the simple moving average
    """

    #Get the column used to calculate the SMA and the corresponding Date for the daily data
    temp_daily_df = price_daily_df[['Date',column_name]].copy(deep=True)

    #Shift the daily index period-1 times (each new columns is shifted by another day)
    for shift in range(1,period):
        temp_daily_df[shift] = temp_daily_df[column_name].shift(periods=shift)
    
    #Get the column used to calculate the SMA and the corresponding Date for the minute data
    temp_minute_df = price_minute_df[['Date',column_name]].copy(deep=True)

    #Merge the daily values into the minute level data based on the same date
    merged_df = temp_minute_df.merge(temp_daily_df,on='Date',how='left')

    sma = merged_df[column_name+'_x'] #current price from the minute data (_x - minute, _y - day)
    daily_sma = temp_daily_df[column_name]
    #Calculate the simple moving average by summing each daily shifted value with the current price
    for shift in range(1,period):
        #merged_df[shift] contains the close price for the ticker 'shift' days ago
        #Ex: 5-day SMA = current_price + close 1 day ago + close 2 days ago + close 3 days ago + close 4 days ago
        sma = sma + merged_df[shift]
        daily_sma = daily_sma + temp_daily_df[shift]
    sma = sma/period
    daily_sma = daily_sma/period
    
    return sma,daily_sma


def calc_ema(price_minute_df,price_daily_df,column_name,period):
    """
    Calculate the exponential moving average for each datetime using the daily closing prices and the previous EMA
    The period is used to calculate the multiplier. The multiplier decreases exponentially for each day in the past. 
        Lower periods means faster multiplier decay (the more recent close prices have much higher weights)
        Higher periods means slower multiplier decay (the more recent close prices have less of a higher weight, further out prices are more relevant)

     For example: EMA = (Current_Price or Close_Price) * Multiplier + EMA_yesterday * (1-multiplier)
                  multiplier = 2/(1+period)
                  first EMA = first close price available (the further out the close_price, the more accurate the EMA)

    price_minute_df - DataFrame of minute level data with 'Datetime' and 'Date' as a column name
    price_daily_df - DataFrame of day level data with 'Datetime' and 'Date' as a column name
    column_name - the column name of the DataFrames used to calculate the EMA
    period - the number of days to include in the exponential moving average

    Returns 
        Exponential moving average (EMA) based on price_minute_df
        Daily exponential moving average based on price_daily_df
    """

    daily_ema = []
    mult = 2/(1+period) #Multiplier
    #Calculate the EMA value for each datetime in price_daily_df
    for daily_index in range(len(price_daily_df['datetime'])):

        #Calculate the period day EMA for all daily values available.
        if daily_index == 0:
            daily_ema.append(price_daily_df[column_name][0]) #For the first EMA just set to the close price
        else:
            daily_ema.append(price_daily_df[column_name][daily_index] * mult + daily_ema[daily_index-1] * (1-mult))

    #Convert daily ema into a numpy array
    daily_ema = np.array(daily_ema)

    #Create a DataFrame based off the daily ema data and the corresponding Date
    daily_ema_df = pd.DataFrame({'Date': price_daily_df['Date'], 'daily_ema': daily_ema})

    #Shift the daily_ema_df by day
    daily_ema_df['prev_daily_ema'] = daily_ema_df['daily_ema'].shift(periods=1)

    #Create a copy of the minute data for the column used to calculate the EMA and the corresponding Date
    temp_minute_df = price_minute_df[['Date',column_name]].copy(deep=True)

    #Merge the daily_ema_df into the minute level data
    merged_df = temp_minute_df.merge(daily_ema_df,how='left',on='Date')

    #Calculate the minute level EMA using the current price and previous day's EMA
    ema = merged_df[column_name]*mult + merged_df['prev_daily_ema']*(1-mult)

    return ema,daily_ema




def calc_std(price_minute_df,price_daily_df,column_name,sma,daily_sma,period):
    """
    Calculate the standard deviation for each datetime using the daily closing prices and the last available price
    The period represents the number of days to use in calculating the standard deviation
    The daily closing prices must have data for at least period days before the time data starts.
     For example: if we are calculating the 20 day standard deviation (period = 20), the daily_close values must start 20 days before the first datetime.
                  Otherwise, we would not be able to calculate the std for the first data point in price_minute_df['datetime'].
                  std = (sum((close_price - SMA)^2)/20)^1/2  
                   - we sum up the square difference between close price and sma except for the current day (we use the current price if close price is not available yet)


    price_minute_df - DataFrame of minute level data with 'Datetime' and 'Date' as a column name
    price_daily_df - DataFrame of day level data with 'Datetime' and 'Date' as a column name
    column_name - the column name of the DataFrames used to calculate the standard deviation
    sma - the simple moving average of the stock over the same period, the sma is calculated for each of the datetime points
        (Ex: if period = 20, sma should be the 20 day simple moving average)
    daily_sma - the simple moving average of the stock over the same period using daily price values
    period - the number of days to include in the standard deviation

    returns the standard deviation

    """

    #Get the column used to calculate the standard deviation and the corresponding Date for the daily data
    temp_daily_df = price_daily_df[['Date',column_name]].copy(deep=True)

    #Shift the daily index period-1 times (each new columns is shifted by another day)
    for shift in range(1,period):
        temp_daily_df[shift] = temp_daily_df[column_name].shift(periods=shift)
    
    #Get the column used to calculate the standard deviation and the corresponding Date for the minute data
    temp_minute_df = price_minute_df[['Date',column_name]].copy(deep=True)

    #Merge the daily values into the minute level data based on the same date
    merged_df = temp_minute_df.merge(temp_daily_df,on='Date',how='left')

    #Squared difference between current price and the SMA
    std = (merged_df[column_name+'_x']-sma)**2
    daily_std = (temp_daily_df[column_name]-daily_sma)**2
    #Calculate the standard deviation by summing each squared difference between previous close prices and the SMA
    for shift in range(1,period):
        #merged_df[shift] contains the close price for the ticker 'shift' days ago
        #Ex: 5-day SMA = current_price + close 1 day ago + close 2 days ago + close 3 days ago + close 4 days ago
        std = std + (merged_df[shift]-sma)**2
        daily_std = daily_std + (temp_daily_df[shift]-daily_sma)**2

    #std = sqrt(sum((x-mu).^2)/N)    
    std = (std/period)**0.5
    daily_std = (daily_std/period)**0.5

    return std,daily_std


def calc_macd(price_minute_df,price_daily_df,column_name):
    """
    Calculate the moving average convergence/divergence oscillator (MACD) for each datetime point.
    The daily closing prices must have data for at least 26 days before the price_minute_df['datetime'] data starts.

    price_minute_df - DataFrame of minute level data with 'Datetime' and 'Date' as a column name
    price_daily_df - DataFrame of day level data with 'Datetime' and 'Date' as a column name
    column_name - the column name of the DataFrames used to calculate the MACD line 

    Returns the MACD line, the signal line, and the MACD histogram
        macd_line = EMA_12days - EMA_26days (using close prices in price_minute_df and price_daily_df)
        signal_line = EMA_9days (using the macd line)
        macd_histogram = macd_line - signal_line

    """
    #The MACD line is the 12 day exponential moving average minus the 26 day exponential moving average
    (ema12,daily_ema12) = calc_ema(price_minute_df,price_daily_df,column_name,12)
    (ema26,daily_ema26) = calc_ema(price_minute_df,price_daily_df,column_name,26)
    macd = pd.DataFrame({'datetime': price_minute_df['datetime'], 'Date': price_minute_df['Date'], 'macd_line': ema12-ema26})
    daily_macd = pd.DataFrame({'datetime': price_daily_df['datetime'], 'Date': price_daily_df['Date'], 'macd_line': daily_ema12-daily_ema26})

    #The signal line is the 9 day exponential moving average of the MACD line
    (signal_line,daily_signal_line) = calc_ema(macd,daily_macd,'macd_line',9)

    #The MACD histogram is the macd_line minus the signal_line
    macd_histogram = macd['macd_line'] - signal_line
    daily_macd_histogram = daily_macd['macd_line'] - daily_signal_line

    return (macd['macd_line'],signal_line,macd_histogram,daily_macd['macd_line'],daily_signal_line,daily_macd_histogram)



def calc_rsi(price_minute_df,price_daily_df,column_name):
    """
    Calculate the Relative Stregnth Index (RSI) for each datetime using the daily closing prices and the last available price.
    The daily closing prices must have data for at least 14 days before the time data starts.
    
    price_minute_df - DataFrame of minute level data with 'Datetime' and 'Date' as a column name
    price_daily_df - DataFrame of day level data with 'Datetime' and 'Date' as a column name
    column_name - the column name of the DataFrames used to calculate the RSI

    Returns the RSI for each point using the price_minute_df
        We caculate the RSI in two parts:
        RSI_First = 100 - 100/(1+average_gain_first/average_loss_first)
            average_gain_first = (sum of the gains between closing prices over 14 days)/14  - gain = 0 for days that closed lower than the previous day
            average_loss_first = (sum of the losses between closing prices over 14 days)/14 - loss = 0 for days that closed higher than the previous day
        
        RSI = 100 - 100/(1+average_gain/average_loss)
            average_gain = (average_gain_prev*13+current_gain)/14 where average_gain_prev is the average gain of the previous day
            average_loss = (average_loss_prev*13+current_loss)/14 where average_loss_prev is the average loss of the previous day

    """
    #We can't calculate the average gain, average loss, or rsi for the first 14 days - need 15 days of data (14 gains/losses over a 1 day lookback period)
    daily_ave_gain,daily_ave_loss = (np.empty(14),np.empty(14))
    daily_ave_gain[:],daily_ave_loss[:] = (np.nan,np.nan)

    #Get the column used to calculate the RSI and the corresponding Date for the daily data
    temp_daily_df = price_daily_df[['Date',column_name]].copy(deep=True)

    #Shift the price by one day to get the previous day prices
    temp_daily_df['Prev_Price'] = temp_daily_df[column_name].shift(periods=1)

    #Calculate the difference between price and previous price and divide into gains and losses
    temp_daily_df['Diff_Price'] = temp_daily_df[column_name]-temp_daily_df['Prev_Price']
    temp_daily_df['Gain'] = temp_daily_df['Diff_Price'].where(temp_daily_df['Diff_Price']>0,0) #Gains from previous day
    temp_daily_df['Loss'] = -temp_daily_df['Diff_Price'].where(temp_daily_df['Diff_Price']<0,0) #Losses from previous day

    #Calculate the first average gain and average loss using the first 14 days of price gains/losses
    daily_ave_gain = np.append(daily_ave_gain,sum(temp_daily_df['Gain'][1:15]))
    daily_ave_loss = np.append(daily_ave_loss,sum(temp_daily_df['Loss'][1:15]))

    #Calculate the RSI for the rest of the datetimes in price_daily_df
    for daily_index in range(15,len(price_daily_df['datetime'])):

        #Add the latest gain/loss to the average gain/loss value (use a moving average with a multiplier of 1/14)
        daily_ave_gain = np.append(daily_ave_gain,temp_daily_df['Gain'][daily_index]/14 + daily_ave_gain[daily_index-1]*13/14)
        daily_ave_loss = np.append(daily_ave_loss,temp_daily_df['Loss'][daily_index]/14 + daily_ave_loss[daily_index-1]*13/14)

    #Calculate the daily RSI. Set the RSI to np.nan for 0 average loss values (can't divide by zero)
    daily_rsi = np.where(daily_ave_loss>0, 100 - 100/(1 + daily_ave_gain/daily_ave_loss), np.nan)

    #Create a DataFrame based off the daily average gain and average loss and the corresponding Date
    daily_ave_gainloss_df = pd.DataFrame({'Date': price_daily_df['Date'], 'ave_gain': daily_ave_gain, 'ave_loss': daily_ave_loss})

    #Shift the price and the average gain/loss values by one day
    daily_ave_gainloss_df['prev_price'] = temp_daily_df['Prev_Price']
    daily_ave_gainloss_df['prev_ave_gain'] = daily_ave_gainloss_df['ave_gain'].shift(periods=1)
    daily_ave_gainloss_df['prev_ave_loss'] = daily_ave_gainloss_df['ave_loss'].shift(periods=1)

    #Create a copy of the minute data for the column used to calculate the RSI and the corresponding Date
    temp_minute_df = price_minute_df[['Date',column_name]].copy(deep=True)

    #Merge the daily previous price and the daily average gain/loss values into the minute level data
    merged_df = temp_minute_df.merge(daily_ave_gainloss_df,how='left',on='Date')

    #Calculate the most recent gain loss from previous day (_x is minute, _y is day)
    merged_df['Diff_Price'] = merged_df[column_name]-merged_df['prev_price']
    merged_df['Gain'] = merged_df['Diff_Price'].where(merged_df['Diff_Price']>0,0) #Gains from previous day
    merged_df['Loss'] = -merged_df['Diff_Price'].where(merged_df['Diff_Price']<0,0) #Losses from previous day

    #Calculate the minute level average gains/losses using the current gain/loss and previous day's average gains/losses
    minute_ave_gain = merged_df['Gain']*(1/14) + merged_df['prev_ave_gain']*(13/14)
    minute_ave_loss = merged_df['Loss']*(1/14) + merged_df['prev_ave_loss']*(13/14)

    #Calculate the RSI for the minute level data
    rsi = np.where(minute_ave_loss>0, 100 - 100/(1 + minute_ave_gain/minute_ave_loss),np.nan)

    return rsi,daily_rsi


def calc_stochastic_oscillator(price_minute_df,price_daily_df,ticker,period):
    """
    Calculate the Stochastic Oscillator for each datetime using the daily high/low prices and the last available price.
    The daily high/low prices must have data for at least period days before the price_minute_df['datetime'] data starts.
    
    price_minute_df - DataFrame of minute level data with 'Datetime' and 'Date' as a column name
    price_daily_df - DataFrame of day level data with 'Datetime' and 'Date' as a column name
    ticker - the stock ticker used for calculating the stochastic oscillator
        the DataFrames must contain the high, low, and close price for this ticker
        In other words, price_minute_df and price_daily_df must contain the columns 'high_'+ticker, 'low_'+ticker, and 'close_'+ticker
    period - the number of days to include in the stochastic oscillator (typically 14 days)

    Returns the K_line and D_line of the stochastic oscillator
        K_line = %K = 100*(Current_Price - period day Low)/(period day High - period day Low) - Ex: 14 day high/low
        D_line = %D = 3 day simple moving average of %K
        D_line_slow = %D_slow = 3 day simple moving average of %D
    """

    #Get the high, low, and close price columns and the corresponding Date for the daily data
    temp_daily_df = price_daily_df[['Date','datetime','high_'+ticker,'low_'+ticker,'close_'+ticker]].copy(deep=True)

    #Shift the daily index period-1 times (so each row will now contain the high/low price for a day and period previous days)
    for shift in range(1,period):
        temp_daily_df['h'+str(shift)] = temp_daily_df['high_'+ticker].shift(periods=shift)
        temp_daily_df['l'+str(shift)] = temp_daily_df['low_'+ticker].shift(periods=shift)
    
    #Create new columns that contain the previous period-1 day high and low to be used with the minute data
    temp_daily_df['high_prev'] = temp_daily_df.loc[:,'h1':'h'+str(period-1)].max(axis=1)
    temp_daily_df['low_prev'] = temp_daily_df.loc[:,'l1':'l'+str(period-1)].min(axis=1)

    #Create new columns that contain the period high and low used for calculating the daily_k_line
    temp_daily_df['period_high_daily'] = temp_daily_df[['high_'+ticker,'high_prev']].max(axis=1)
    temp_daily_df['period_low_daily'] = temp_daily_df[['low_'+ticker,'low_prev']].min(axis=1)

    #Calculate the daily k_line
    temp_daily_df['daily_k_line'] = 100*(temp_daily_df['close_'+ticker]-temp_daily_df['period_low_daily'])/(temp_daily_df['period_high_daily']-temp_daily_df['period_low_daily'])

    #Shift the daily_k_line to get the daily_k_line for the previous two days (used for d_line calculation)
    temp_daily_df['daily_k_line_prev1'] = temp_daily_df['daily_k_line'].shift(periods=1)
    temp_daily_df['daily_k_line_prev2'] = temp_daily_df['daily_k_line'].shift(periods=2)

    #Calculate daily d_line (and the previous two day d_lines) to get the minute level slow d_line (see below)
    temp_daily_df['daily_d_line'] = (temp_daily_df['daily_k_line']+temp_daily_df['daily_k_line_prev1']+temp_daily_df['daily_k_line_prev2'])/3
    temp_daily_df['daily_d_line_prev1'] = temp_daily_df['daily_d_line'].shift(periods=1)
    temp_daily_df['daily_d_line_prev2'] = temp_daily_df['daily_d_line'].shift(periods=2)

    #Get the high, low, and close price columns and the corresponding Date for the minute data
    temp_minute_df = price_minute_df[['Date','high_'+ticker,'low_'+ticker,'close_'+ticker]].copy(deep=True)

    #Group by the date and get the cumulative max/min for each day (to get the high/low of a day at any given time)
    temp_minute_df['dayhigh'] = temp_minute_df.groupby(['Date'])['high_'+ticker].cummax()
    temp_minute_df['daylow'] = temp_minute_df.groupby(['Date'])['low_'+ticker].cummin()

    #Merge the daily data into the minute level data
    merged_df = temp_minute_df.merge(temp_daily_df,on='Date',how='left')

    #Calculate the minute period day high/low by finding the max/min of the previous period-1 day high/low with the cumulative max/min on minute data
    merged_df['period_high'] = merged_df[['high_prev','dayhigh']].max(axis=1)
    merged_df['period_low'] = merged_df[['low_prev','daylow']].min(axis=1)

    #Calculate the minute k_line using the period day high and period day low (_x - minute, _y - day)
    k_line = 100*(merged_df['close_'+ticker+'_x'] - merged_df['period_low'])/(merged_df['period_high']-merged_df['period_low'])

    #Calculate the d line by getting the simple moving average of the minute k line data and the previous two daily k line values
    d_line = (k_line+merged_df['daily_k_line_prev1']+merged_df['daily_k_line_prev2'])/3

    #Calculate the slow d line by getting the simple moving average of the d line and the d lines of the previous two days
    d_line_slow = (d_line+merged_df['daily_d_line_prev1']+merged_df['daily_d_line_prev2'])/3
    daily_d_line_slow = (temp_daily_df['daily_d_line']+temp_daily_df['daily_d_line_prev1']+temp_daily_df['daily_d_line_prev2'])/3

    return k_line,d_line,d_line_slow,temp_daily_df['daily_k_line'],temp_daily_df['daily_d_line'],daily_d_line_slow