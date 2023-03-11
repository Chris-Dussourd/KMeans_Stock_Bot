
"""
TD Ameritrade_get_prices

Thread that gets the current prices by getting quotes from the TD Ameritrade API.

CD 12/2020 - Created

"""

import datetime,json,pickle,time
from TDAmeritrade_algorithm_centroids import track_buy_orders,track_sell_orders,check_request_times
from TDAmeritrade_API import get_access


def get_prices(lock,transactions,errors):
    """
    Calls get_quotes every minute to get minute level price data. 
    Loads the transactions dictionary to set up price information for stocks. 

    Keys in transactions[ticker] dictionary that apply to this function
        Bid Price - most recent highest bid price 
        Ask Price - most recent lowest ask price
        Price - estimated current price; (bid_price + ask_price)/2
        Minute Prices - holds at least a day's worth of minute level data (deletes old data from two days ago before market open)
        Daily Prices - holds at least 6 months worth of day level data (deletes a month's worth a data at the beginning of a new month)


    Keys of the transactions dictionary that apply to this function
        Access Token - the token we use to access the TD Ameritrade site
        Access Expire Time - time the access token expires
        Max Buys - the number of buy orders we allow the bot to place without any sells (prevents continuous purchase of stock heading to zero)
        Last Requests - the last requests we made to the TD Ameritrade API (make sure we don't make more than two calls per second)

    Stores latest results of transaction dictionary as a pickle file

    """
    #Run this code until we encounter an error or trading hours have ended (include extended hours)
    open_time = datetime.datetime(2021,1,1,4,0,0,0).time()
    close_time = datetime.datetime(2021,1,1,17,0,0,0).time()
    open_time_seconds = datetime.timedelta(hours=open_time.hour,minutes=open_time.minute,seconds=open_time.second).total_seconds()
    holidays = get_holidays(datetime.datetime(2021,1,1),datetime.datetime(2021,12,31))

    while len(errors)==0:
        
        #Make sure we have an up to date datetime list in 'Minute Prices'
        ready_prices=False

        current = datetime.datetime.now()
        current_seconds = datetime.timedelta(hours=current.hour,minutes=current.minute,seconds=current.second).total_seconds()
        #Sleep until trading opens for the day
        if open_time_seconds>=current_seconds+60:
            time.sleep(open_time_seconds-current_seconds-60)
        if close_time<datetime.datetime.now().time():
            time.sleep(open_time_seconds+86300-current_seconds)

        #If the day is Saturday, sleep for two more days
        if datetime.datetime.now().weekday()==5:
            test=1
            time.sleep(2*86300)

        #If the day is a holiday, sleep for one more day
        if datetime.datetime.now().date in holidays:
            test=1
            time.sleep(86300)
            
        #Remove old minute and day level data
        if len(errors)==0 and open_time<=datetime.datetime.now().time()+datetime.timedelta(minutes=1) and close_time>=datetime.datetime.now().time() and ~ready_prices:
            for ticker in transactions['Tickers Prices']:
                #Remove minute data from two trading days ago
                if len(transactions[ticker]['Minute Prices']) > 2*780:
                    transactions[ticker]['Minute Prices'].drop(range(780)).reset_index(drop=True)

                #Remove day level data from 7 months ago (approximately 21 trading days in a month)
                if len(transactions[ticker]['Daily Prices']) > 7*21:
                    transactions[ticker]['Daily Prices'].drop(range(21)).reset_index(drop=True)
            ready_prices=True


        while len(errors)==0 and open_time<=datetime.datetime.now().time() and close_time>=datetime.datetime.now().time() and ready_prices:
            
            lock.acquire()
            try:
                #Get quotes information from API call
                new_quotes = get_multi_quotes(transactions['Access Token'],','.join(transactions['Tickers Prices']))
                transactions['Last Requests'][transactions['Min Request Index']]=time.time()
                #Save time that the quote was obtained
                price_time = time.time()
                price_minute = datetime.datetime.now().replace(seconds=0,microseconds=0) + datetime.timedelta(minutes=1) #Round up to minute

                for ticker in transactions['Tickers Prices']:

                    #For each ticker store the latest bid and ask price
                    if ticker in new_quotes and 'bidPrice' in new_quotes[ticker] and 'askPrice' in new_quotes[ticker]:
                        transactions[ticker]['Bid Price'] = new_quotes[ticker]['bidPrice']
                        transactions[ticker]['Ask Price'] = new_quotes[ticker]['askPrice']
                        #Use the mean of the bid and ask price as the estimated current price
                        transactions[ticker]['Price'] = (new_quotes[ticker]['bidPrice']+new_quotes[ticker]['askPrice'])/2

                        #Save minute level data at current datatime, fill in earlier datetimes
                        last_filled_minute = transactions[ticker]['Minute Prices'].iloc[-1]['datetime']
                        last_filled_price = transactions[ticker]['Minute Prices'].iloc[-1]['close']

                        #Fill all minutes leading up to current minute
                        while (price_minute != last_filled_minute+datetime.timedelta(minutes=1) and price_minute != last_filled_minute):
                            last_filled_minute = last_filled_minute+datetime.timedelta(minutes=1)
                            transactions[ticker]['Minute Prices'] = transactions[ticker]['Minute Prices'].append(pd.Series({'datetime': last_filled_minute, 'close': last_filled_price}))

                        #Fill current minute
                        transactions[ticker]['Minute Prices'] = transactions[ticker]['Minute Prices'].append(pd.Series({'datetime': price_minute, 'close': transactions[ticker]['Price']}))

            except:
                transactions['API Errors'].append('Could not get quotes')
                
            last_requests=transactions['Last Requests']
            lock.release()
            #Make sure we are not making too many requests to the TD Ameritrade API
            transactions['Min Request Index']=check_request_times(last_requests)

            #Sleep till next minute
            current_time = time.time()
            if (price_time+60-current_time)>0:
                time.sleep(price_time+60-current_time)


def get_holidays(start_date,end_date):
    """
    CD 12/2020 Get holiday datetimes within start_date and end_date

    start_date - earlier datetime to get holiday dates from
    end_date - later datetime to get holiday dates to

    Return list of holidays (as a date) within start_date and end_date
    """
    holiday_list = []
    #Loop over each month between start and end date
    date = datetime.date(year=start_date.year,month=start_date.month,day=start_date.day)
    while date <= end_date:
        if date.month == 1:
            #New Years Day
            holiday_list.append(datetime.date(date.year,1,1))

            #Martin Luter King Day (3rd Monday of January)
            #Earliest is Jan 15th and latest is Jan 21st, number of days ahead the next Monday is from January 14th
            days_ahead = 7-datetime.date(date.year,1,14).weekday() #Weekday=0 for Monday
            holiday_list.append(datetime.date(date.year,1,14)+datetime.timedelta(days=days_ahead))
        elif date.month == 2:
            #Washington's Birthday (3rd Monday of February)
            #Earliest is Feb 15th and latest is Feb 21st, number of days ahead the next Monday is from February 14th
            days_ahead = 7-datetime.date(date.year,2,14).weekday() #Weekday=0 for Monday
            holiday_list.append(datetime.date(date.year,2,14)+datetime.timedelta(days=days_ahead))

        elif date.month == 3 or date.month==4:
            #Good Friday (Friday before Easter which falls on first Sunday following full moon on or after March 21) 
            holiday_list.append(calc_easter(date.year)-datetime.timedelta(days=2))

        elif date.month == 5:
            #Memorial Day (last Monday of May)
            #Earliest is May 25th and latest is May 31st, number of days ahead the next Monday is from May 24th
            days_ahead = 7-datetime.date(date.year,5,24).weekday() #Weekday=0 for Monday
            holiday_list.append(datetime.date(date.year,5,24)+datetime.timedelta(days=days_ahead))

        elif date.month == 7:
            #Independence Day
            if datetime.date(date.year,7,4).weekday() == 5:
                #Observed on Friday July 3rd
                holiday_list.append(datetime.date(date.year,7,3))
            elif datetime.date(date.year,7,4).weekday() == 6:
                #Observed on Monday July 5th
                holiday_list.append(datetime.date(date.year,7,5))
            else:
                holiday_list.append(datetime.date(date.year,7,4))
        
        elif date.month == 9:
            #Labor Day (1st Monday of September)
            #Earliest is Sep 1st and latest is Sep 6th, number of days ahead the next Monday is from August 31st
            days_ahead = 7-datetime.date(date.year,8,31).weekday() #Weekday=0 for Monday
            holiday_list.append(datetime.date(date.year,8,31)+datetime.timedelta(days=days_ahead))

        elif date.month == 11:
            #Thanksgiving Day (4th Thursday of November)
            #Earliest Monday before Thanksgiving is Nov 19th and latest is Nov 25th, add three extra days to make it Thursday (10 instead of 7)
            days_ahead = 10-datetime.date(date.year,11,18).weekday() #Weekday=0 for Monday
            holiday_list.append(datetime.date(date.year,11,18)+datetime.timedelta(days=days_ahead))

        elif date.month == 12:
            #Christmas Day
            if datetime.date(date.year,12,25).weekday() == 5:
                #Observed Friday December 24th
                holiday_list.append(datetime.date(date.year,12,24))
            elif datetime.date(date.year,12,25).weekday() == 6:
                #Observed Monday December 26th
                holiday_list.append(datetime.date(date.year,12,26))
            else:
                holiday_list.append(datetime.date(date.year,12,25))

        next_month = 1 if date.month==12 else date.month+1
        date = date.replace(month=next_month)

    #Remove any holidays that don't fall within range (can happen for first month or last month in loop)
    for date in holiday_list:
        if (date<start_date) or (date>end_date):
            holiday_list.remove(date)

    return holiday_list


def calc_easter(year):
    """
    Returns Easter as a date object.

    Obtained from https://code.activestate.com/recipes/576517-calculate-easter-western-given-a-year/?in=lang-python
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = (19 * a + b - b // 4 - ((b - (b + 8) // 25 + 1) // 3) + 15) % 30
    e = (32 + 2 * (b % 4) + 2 * (c // 4) - d - (c % 4)) % 7
    f = d + e - 7 * ((a + 11 * d + 22 * e) // 451) + 114
    month = f // 31
    day = f % 31 + 1    
    return datetime.date(year, month, day)