
"""
Saves minute level or daily price history data for tickers from TD Ameritrade Website. 
    Note: TD Ameritrade only allows access to minute level data for one or two months prior to the current date.

CD 12/2020 - Move codeinto a function and add support for getting price history between two dates
           - Use the file_structure_dict as the folder structure to save files or folder_path if passed in
           - Clean up code and add in raised errors if user tries to get data for dates too far in the past
"""

from TDAmeritrade_API import get_access,get_price_history_lookback,get_price_history_dates
from file_organization import folder_directory,folder_structure,file_structure,create_filename
import time,datetime,os
import pickle

def save_pricehistory(start_date,end_date,frequencyType,tickers_list=[],folder_path=''):
    """
    CD 12/2020  - Place code into a function and allow support for getting between two dates 
                - Add support for file and folder structure from file_structure_dict.py
                - Clean up code and add in raised errors if user tries to get data for dates too far in the past

    Save price history data for all tickers in tickers_list or in folder_structure as a pickle file

    start_date - 'YYYYMMDD' string to get price data from
    end_date - 'YYYYMMDD' string to get price data to
    frequency - supports minute level and day level price/volume data, 'minute' or 'daily'
                minute: only can obtain detailed data two months before current date
                day: only can obtain detailed data two years before current date
    tickers_list - pass in a list of tickers to get data for instead of getting them from folder_structure
                 - if empty, just use the folder_structure from file_structure_dict.py instead
    folder_path - pass in a folder_path to save price history for tickers in tickers_list, only use if tickers_list is not empty
                - if this is null, tickers in tickers_list will not be saved (only returned)
                - data saved to this folder will be named 'Ticker_StartDate_EndDate_Frequency.p' Ex: 'GOOG_20201201_20201231_Frequency.p'

    Example dictionary saved to a pickle file and returned:
    price_history['AAPL'] = {
        'datetime':  []  - recorded in milliseconds per epoch
        'open':
        'high': []
        'low': []
        'volume': []

    Saves price history of each ticker individually as a pickle file according to folder_structure
    Returns the price history of all tickers
    }
        
    """

    #Get an access token in order to get price data
    access_token=''
    expire_time=0
    [access_token,expire_time]=get_access(access_token,expire_time)
    last_access = time.time()
    price_history_all = {}

    #Time of last API call. Make sure only 2 API calls per second (max API calls allowed by TD Ameritrade)
    time_start = time.time()

    #use folder_structure tickers if tickers_list is empty
    tickers_loop = folder_structure.keys() if not tickers_list else tickers_list

    #Loop through each ticker to save minute level data for this set
    for ticker in tickers_loop:

        #File path to save these tickers to
        file_path = os.path.join(folder_directory,folder_structure[ticker]) if not tickers_list else folder_path

        #Price history is a dictionary that stores minute level data of each ticker.
        price_history = {} #Reset for each ticker (each file should only contain one ticker)
        price_history[ticker]=[]

        #Extract out dates from start_date and end_date
        start = datetime.date(int(start_date[0:4]),int(start_date[4:6]),int(start_date[6:]))
        stop = datetime.date(int(end_date[0:4]),int(end_date[4:6]),int(end_date[6:]))

        # *** USE START AND END DATE IF GETTING MINUTE LEVEL DATA ***
        #Get data at the minute level with each API call
        if frequencyType == 'minute':
            lookback = False
            frequency=1

            #Make sure start_date is within two months of current date (TD Ameritrade doesn't store further out)
            #if (datetime.date.today()-start).days > 62:
                #raise ValueError('TD Ameritrade can only return two months of minute level data from the current date. Use a start_date closer to the current date.')
            
            start_dates_list=[]
            end_dates_list=[]
            #The API will return minute level day for 10 workdays (~14 days including weekends) max, so make multiple calls of 14 day periods to get the full time range if necessary.
            for i in range((stop-start).days//14+1):
                start_temp = start+datetime.timedelta(days=i*14) #Start period
                stop_temp = start+datetime.timedelta(days=(i+1)*14)
                stop_temp = stop if stop_temp>stop else stop_temp #Use stop when stop_temp is further out than stop (stay within our start/end date)

                start_dates_list.append(datetime.datetime.combine(start_temp,datetime.datetime.min.time()).timestamp()*1000)
                end_dates_list.append(datetime.datetime.combine(stop_temp,datetime.datetime.max.time()).timestamp()*1000)

        # *** USE LOOKBACK DATE IF GETTING DAILY, WEEKLY, OR MONTHLY DATA ***
        else:
            lookback = True
            frequency=1

            #Get data looking back a period including start_date.
            if (datetime.date.today()-start).days < 28: #Min days in a month is 28
                periodType='month'
                period=1
            elif (datetime.date.today()-start).days <  59: #Min days in two consecutive months is 59
                periodType='month'
                period=2
            elif (datetime.date.today()-start).days < 89: #Min days in three consecutive months is 89
                periodType='month'
                period=3
            elif (datetime.date.today()-start).days < 181: #Min days in six consecutive months is 181
                periodType='month'
                period=6
            elif (datetime.date.today()-start).days < 365:
                periodType='year'
                period=1
            elif (datetime.date.today()-start).days < 730:
                periodType='year'
                period=2
            elif (datetime.date.today()-start).days < 1826:
                periodType='year'
                period=5
            elif (datetime.date.today()-start).days < 3652:
                periodType='year'
                period=10
            elif (datetime.date.today()-start).days < 5478:
                periodType='year'
                period=15
            elif (datetime.date.today()-start).days < 7304:
                periodType='year'
                period=20
            else:
                raise ValueError('TD Ameritrade can only return twenty years of daily level data from the current date. Use a start_date closer to the current date.')

        #Create a price history dictionary. Each key contains a list of values (dates, prices, or volume)
        price_history[ticker]={'datetime': [],'open': [],'close': [],'high': [],'low': [],'volume': []}

        #Get the start_date and end_date as timestamp values in milliseconds
        start_timestamp = datetime.datetime.combine(start,datetime.datetime.min.time()).timestamp()*1000
        end_timestamp = datetime.datetime.combine(stop,datetime.datetime.max.time()).timestamp()*1000

        if lookback: #Daily, monthly, or yearly data

            time_end = time.time() #time_end - time_start = time since last API call
            if time_end-time_start < 0.5:
                #Only make 2 API calls per second
                time.sleep(0.5-(time_end-time_start))

            #Refresh access token when we reach 27 minutes (it expires after 30)
            if time.time()-last_access > 27*60:
                [access_token,expire_time]=get_access()
                last_access = time.time()

            time_start = time.time()
            price_history_temp = get_price_history_lookback(access_token,ticker,periodType,period,frequencyType,frequency)
            for price_info in price_history_temp['candles']:
                #Only save data if within the start_date and end_date time period (subtract one day since TD Ameritrade's daily chart is off by one day)
                if price_info['datetime'] >= start_timestamp-86400000 and price_info['datetime'] <= end_timestamp-86400000:
                    #Save the info in the dictionary 
                    price_history[ticker]['datetime'].append(price_info['datetime'])
                    price_history[ticker]['open'].append(price_info['open'])
                    price_history[ticker]['close'].append(price_info['close'])
                    price_history[ticker]['high'].append(price_info['high'])
                    price_history[ticker]['low'].append(price_info['low'])
                    price_history[ticker]['volume'].append(price_info['volume'])

            #The daily chart is off by one day for some reason (it says the open/close/high/low price is for 8/3 when it's acutally for 8/4).
            #Update the price history datetime by one day (86400000 in datetime units)
            price_history[ticker]['datetime'] = [x + 86400000 for x in price_history[ticker]['datetime']]

        else: #Minute level data

            for i in range(len(start_dates_list)):

                #Start date and end date since epoch (dates are used for determining days, weeks, months, or years - not specific to hours
                startDate=int(start_dates_list[i])
                endDate=int(end_dates_list[i])

                time_end = time.time() #time_end - time_start = time since last API call
                if time_end-time_start < 0.5:
                    #Only make 2 API calls per second
                    time.sleep(0.5-(time_end-time_start))

                #Refresh access token when we reach 27 minutes (it expires after 30)
                if time.time()-last_access > 27*60:
                    [access_token,expire_time]=get_access()
                    last_access = time.time()

                time_start = time.time()
                price_history_temp = get_price_history_dates(access_token,ticker,startDate,endDate,frequencyType,frequency)

                for price_info in price_history_temp['candles']:
                    #Only save data if within the start_date and end_date time period
                    if price_info['datetime'] >= start_timestamp and price_info['datetime'] <= end_timestamp:
                        #Save the info in the dictionary 
                        price_history[ticker]['datetime'].append(price_info['datetime'])
                        price_history[ticker]['open'].append(price_info['open'])
                        price_history[ticker]['close'].append(price_info['close'])
                        price_history[ticker]['high'].append(price_info['high'])
                        price_history[ticker]['low'].append(price_info['low'])
                        price_history[ticker]['volume'].append(price_info['volume'])

        if not tickers_list:
            #Used in create_filename to order the variables in a particular structure
            file_info_dict = {
                'vStartDate': start_date,
                'vEndDate': end_date,
                'vTicker': ticker,
                'vFrequency': frequencyType.capitalize()
            }  
            #Get the filename based off the file_structure
            file_name = create_filename(file_structure,file_info_dict)
        elif len(folder_path) != 0:
            #Use a default filename
            file_name = '{}_{}_{}_{}.p'.format(ticker,start_date,end_date,frequencyType.capitalize())

        #Only save files if we are using the file_structure_dict or if folder_path is not null
        if len(file_path) > 0:
            pickle.dump(price_history,open(os.path.join(file_path,file_name),"wb"))

        price_history_all[ticker] = price_history[ticker]

    return price_history_all


# datetime.datetime.fromtimestamp(price_history[ticker]['datetime'][0]/1000)

if __name__ == "__main__":
    start_date = '20201201'
    end_date = '20201231'
    frequencyType = 'minute'
    #tickers_list = ['IMMP','INO']
    #folder_path = ''
    price_history_all = save_pricehistory(start_date,end_date,frequencyType)

    datapoints_saved = {}
    first_datetime = {}
    last_datetime = {}
    first_price = {}
    last_price = {}
    #Get the amount of data that is saved for each ticker
    for ticker in price_history_all:
        datapoints_saved[ticker] = len(price_history_all[ticker]['datetime'])
        first_datetime[ticker] = datetime.datetime.fromtimestamp(price_history_all[ticker]['datetime'][0]/1000)
        last_datetime[ticker] = datetime.datetime.fromtimestamp(price_history_all[ticker]['datetime'][-1]/1000)
        first_price[ticker] = price_history_all[ticker]['open'][0]
        last_price[ticker] = price_history_all[ticker]['close'][-1]

    test=1
