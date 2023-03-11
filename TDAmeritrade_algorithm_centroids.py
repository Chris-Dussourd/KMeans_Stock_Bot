
"""
TD Ameritrade_algorithm_centroids

1. Get a list of tickers in transactions['Tickers Algorithm'] that have enough volume to buy and that are not too expensive for available funds.
2. Calculates feature data for all stocks in the list of tickers.
3. Calculates the cluster centroid that each stock currently belongs to. Buy stock that belong to a 'Strong Buy' centroid.
4. Sell stock if one of the following three metrics are met (only sell stock owned for at least 1 day to avoid breaking day trading rules):
    a. Profit or Break Even acheived and current feature data is in a sell cluster centroid
    b. A Limit order is hit that was set to twice the last Threshold value (thresholds are profit/loss percent used to divide centroid categories, last one contains highest profit)
    c. Profit or Break Even acheived and predict period is over
5. Save latest data at market close

CD 12/2020 - Revise algorithm to use centroid values from K Means clustering algorithm .

"""

import datetime,time,dateutil.parser
import pytz, pickle,math
import numpy as np
import pandas as pd
from TDAmeritrade_API import *
from CalculateFeatures import calc_features, find_strong_buys, get_centroid_label
from TDAmeritrade_get_prices import get_holidays

def ordering_bot(lock,transactions,errors):
    """
    Runs a bot to buy and sell stock. 
    Loads the transactions dictionary to set up parameters for when to buy and sell stock. 

    Within the transactions dictionary there are dictionaries for each stock. The keys of each stock dictionary are:
        Thresholds - the thresholds that divide the clusters into different categories (strong buy uses the last two thresholds)
        Centroids - a dictionary where each key is a centroid and the values are feature data that represent the center of a cluster
        Centroid Labels - a pandas Series where each index is a centroid number and the values are a label (Strong Buy, Buy, Weak Buy, Neutral, Weak Sell, Sell, Strong Sell)
        Features - a DataFrame that contains the most recent feature info on a ticker, use this to calculate real time feature values
        Features Scaled - a DataFrame that contains the most recent feature info scaled, used to find the centroid label of current data
        Feature Scaling - a DataFrame of the minimum and maximum values for each feature (use to scale features)
        Average Volume - the average volume of the ticker over the past month

        Bid Price - most recent highest bid price 
        Ask Price - most recent lowest ask price
        Price - estimated current price; (bid_price + ask_price)/2


    Keys of the transactions dictionary that apply to all stocks
        Access Token - the token we use to access the TD Ameritrade site
        Access Expire Time - time the access token expires
        Max Buys - the number of buy orders we allow the bot to place without any sells (prevents continuous purchase of stock heading to zero)
        Last Requests - the last requests we made to the TD Ameritrade API (make sure we don't make more than two calls per second)

        Available Balance - money available to make buy orders
        Mimimum Purchase - lowest amount of money we will spend on one transaction
        Maximum Purchase - maximum amount of money we can spend on one transaction
        Number Transactions - number of transactions allowed to divide the available balance into (unless under minimum purchase or over maximum purchase)
        Save Proportion - proportion of profit to save (rest goes back into available balance for bot)

        Predict Period - the period in the future the k means clustering algorithm is predicting in the future
        Predict Period Type - the type of period ('day' and 'minute' supported) in the future the k means algorithm is predicting
        Features List - list of features that were used to calculate centroids

        Buy Orders - DataFrame of current buy orders; columns - Ticker, Datetime, Limit Buy ID, Limit Buy Price, Limit Buy Quantity
        Current Stock - DataFrame of current stock owned; columns - Ticker, Buy Datetime, Buy Price, Buy Quantity, Limit Sell ID, Limit Sell Price
        Past Orders - DataFrame of past orders; columns - Ticker, Buy Datetime, Buy Price, Buy Quantity, Sell Datetime, Sell Price, Sell Quantity

    Stores latest results of transaction dictionary as a pickle file

    """

    #Run this code until we encounter an error or trading hours have ended (include extended hours)
    open_time = datetime.datetime(2021,1,1,4,0,0,0).time()
    close_time = datetime.datetime(2021,1,1,17,0,0,0).time()
    open_time_seconds = datetime.timedelta(hours=open_time.hour,minutes=open_time.minute,seconds=open_time.second).total_seconds()
    holidays = get_holidays(datetime.datetime(2021,1,1),datetime.datetime(2021,12,31))

    #Continue looping while we don't have any errors
    while len(errors)==0:

        #Make sure we have an up to date tickers list for the day before starting loop
        ready=False

        #Reset API errors list
        transactions['API Errors']=[]

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
            time.sleep(2*86400)

        #Get list of tickers right before open
        if len(errors)==0 and open_time<=datetime.datetime.now().time()+datetime.timedelta(minutes=1) and close_time>=datetime.datetime.now().time() and ~ready:
            
            lock.acquire()
            try:
                #Get the access token and the expire time of the access token.
                (transactions['Access Token'],transactions['Access Expire Time']) = get_access(transactions['Access Token'],transactions['Access Expire Time'])
                transactions['Last Requests'][transactions['Min Request Index']]=time.time()

                ############################### Get list of tickers #############################
                transactions['Tickers Today'] = transactions['Tickers Algorithm']
                new_quotes = get_multi_quotes(transactions['Access Token'],','.join(transactions['Tickers Algorithm']))
                transactions['Last Requests'][transactions['Min Request Index']]=time.time()

                for ticker in transactions['Tickers Today']:
                    if ticker not in new_quotes:
                        #Remove tickers that were not obtained with a TD API Call
                        transactions['Tickers Today'].remove(ticker)

                    elif 'lastPrice' in new_quotes[ticker]:
                        #Find total balance by getting money spent on current transactions and available balance
                        total_balance = transactions['Available Balance'] + sum(transactions['Buy Orders']['Limit Buy Price']*transactions['Buy Orders']['Limit Buy Quantity']) \
                            + sum(transactions['Current Stock']['Buy Price']*transactions['Current Stock']['Buy Quantity'])
                        if new_quotes[ticker]['lastprice'] > max(total_balance/10,100):
                            #Remove stock that is too expensive from list
                            transactions['Tickers Today'].remove(ticker)

                        elif min(transactions['Maximum Purchase'],total_balance/transactions['Number Transactions']) > \
                            new_quotes[ticker]['lastprice']*transactions[ticker]['Average Volume']*0.01:
                            #Remove stock from ticker if the quantity bought per transaction would be more than 1 percent of the average day's volume
                            transactions['Tickers Today'].remove(ticker)


                #Tickers list is up to date and ready to loop for the day
                ready=True

            except:
                transactions['API Errors'].append('Could not get quotes')

            last_requests=transactions['Last Requests']
            lock.release()
            #Make sure we are not making too many requests to the TD Ameritrade API
            transactions['Min Request Index']=check_request_times(last_requests)
                         

        while len(errors)==0 and open_time<=datetime.datetime.now().time() and close_time>=datetime.datetime.now().time() and ready:
            
            ############################### Get a new access token ##########################
            #Get the access token and the expire time of the access token.
            (transactions['Access Token'],transactions['Access Expire Time']) = get_access(transactions['Access Token'],transactions['Access Expire Time'])



            ############################### Get Feature and Cluster Data #################
            lock.acquire()
            #Cost spent per transaction 
            cost_per_transaction = max(transactions['Minimum Purchase'],min(transactions['Maximum Purchase'],total_balance/transactions['Number Transactions']))

            # Find stocks with strong buy centroids if we have money for the transaction 
            if transactions['Available Balance'] > cost_per_transaction:

                #Calculate the latest feature data for each ticker and store in transactions
                calc_features(transactions,transactions['Tickers Today'])

                #Find current stocks to buy using centroid labels
                num_buys = math.floor(transactions['Available Balance']/cost_per_transaction)
                buy_list = find_strong_buys(transactions,transactions['Tickers Today'],num_buys)



            ############################### Place Limit Buy Orders #########################
            lock.acquire()
            #Place buy orders for tickers
            for ticker in buy_list:
                #Place the new limit buy order for each ticker (for first buy and if all stock has been sold)
                place_buy_order(transactions,ticker)
                transactions['Last Requests'][transactions['Min Request Index']]=time.time()

                #Remove the ticker from buy list and ticker list after placing order
                transactions['Tickers Today'].remove(ticker)
                buy_list.remove(ticker)

                last_requests=transactions['Last Requests']
                #Make sure we are not making too many requests to the TD Ameritrade API
                transactions['Min Request Index']=check_request_times(last_requests)
            lock.release()



            ############################### Track Orders ###################################
            lock.acquire()
            for ticker in transactions['Buy Orders']['Ticker']:
                #Check if the bid price dipped below the limit buy order
                if transactions[ticker]['Bid Price']<=transactions[ticker]['Limit Buy Price']:
                    #Track the buy order, remove from 'Buy Orders' and add to 'Current Stock' if purchased
                    track_buy_orders(transactions,ticker)
                    transactions['Last Requests'][transactions['Min Request Index']]=time.time()
                
                last_requests=transactions['Last Requests']
                #Make sure we are not making too many requests to the TD Ameritrade API
                transactions['Min Request Index']=check_request_times(last_requests)

            for ticker in transactions['Current Stock']['Ticker']:
                #Check if there are limit sell orders and the ask price rose above the limit sell order
                if transactions['Current Stock']['Ticker'==ticker]['Limit Sell Price']>0 and \
                    transactions[ticker]['Ask Price']>=transactions['Current Stock']['Ticker'==ticker]['Limit Sell Price']:
                    #Track the sell order (remove from Current Stock if filled). Add ticker to transactions['Tickers Today'] since we can buy it again
                    track_sell_orders(transactions,ticker)
                    transactions['Last Requests'][transactions['Min Request Index']]=time.time()
            
                last_requests=transactions['Last Requests']
                #Make sure we are not making too many requests to the TD Ameritrade API
                transactions['Min Request Index']=check_request_times(last_requests)
            lock.release()


            ############################### Place Limit Sell Orders #########################
            lock.acquire()
            for ticker in transactions['Current Stock']['Ticker']:

                #If stock is owned for more than one day and no order currently placed, place a limit order for this ticker
                if transactions['Current Stock']['Ticker'==ticker]['Buy Date'] < datetime.datetime.now().date and \
                    transactions['Current Stock']['Ticker'==ticker]['Limit Sell ID']==0:

                    #Place sell order at twice the largest threshold
                    sell_price = transactions['Current Stock']['Ticker'==ticker]['Buy Price']*(1+2*transactions[ticker]['Thresholds'][-1])
                    place_sell_order(transactions,ticker,sell_price)
                    transactions['Last Requests'][transactions['Min Request Index']]=time.time()

                #If stock is owned for more than one day and order is currently placed
                elif transactions['Current Stock']['Ticker'==ticker]['Buy Datetime'].date < datetime.datetime.now().date and \
                    transactions['Current Stock']['Ticker'==ticker]['Limit Sell ID']>0:

                    if transactions['Predict Period Type'] == 'day':
                        predict_delta = datetime.timedelta(days=transactions['Predict Period'])
                    elif transactions['Predict Period Type'] == 'minute':
                        predict_delta = datetime.timedelta(minutes=transactions['Predict Period'])

                    #Buy transaction occurred within predict period days ago (prediction period is not up)
                    if datetime.datetime.now()-transactions['Current Stock']['Ticker'==ticker]['Buy Datetime'] < predict_delta:

                        #Get the cluster that the ticker data currently belongs to
                        centroid_label = get_centroid_label(transactions,ticker)

                        #Replace limit sell order if profit/break-even acheived, order is within predict period, and current data is in a sell cluster
                        if transactions['Current Stock']['Ticker'==ticker]['Buy Price']<transactions[ticker]['Bid Price'] and \
                            centroid_label in ['Weak Sell','Sell','Strong Sell']:

                            #Sell at the mean between bid and ask price
                            sell_price = transactions[ticker]['Price']
                            replace_sell_order(transactions,ticker,sell_price)
                            transactions['Last Requests'][transactions['Min Request Index']]=time.time()

                    else:

                        #Once prediction period is over place a limit sell order at break-even price or mean of bid and ask price
                        sell_price = max(transactions['Current Stock']['Ticker'==ticker]['Buy Price'],transactions[ticker]['Price'])
                        if sell_price < transactions['Current Stock']['Ticker'==ticker]['Limit Sell Price']:
                            replace_sell_order(transactions,ticker,sell_price)
                            transactions['Last Requests'][transactions['Min Request Index']]=time.time()

                last_requests=transactions['Last Requests']
                #Make sure we are not making too many requests to the TD Ameritrade API
                transactions['Min Request Index']=check_request_times(last_requests)
            lock.release()
            

            ############################### Cancel Buy Orders ########################
            lock.acquire()
            for ticker in transactions['Buy Orders']['Ticker']:
                #Cancel buy orders if it has been unfilled in 10 minutes and it is not longer a strong buy centroid
                if datetime.datetime.now() - transactions['Current Stock']['Ticker'==ticker]['Buy Datetime'] > datetime.timedelta(minutes=10):

                    #Get the cluster that the ticker data currently belongs to
                    centroid_label = get_centroid_label(transactions,ticker)

                    if centroid_label != 'Strong Buy':
                        #Cancel buy order on TD Ameritrade and remove it from buy arrays
                        cancel_buy_orders(transactions,ticker)   
                        transactions['Last Requests'][transactions['Min Request Index']]=time.time()

                last_requests=transactions['Last Requests']
                #Make sure we are not making too many requests to the TD Ameritrade site
                transactions['Min Request Index']=check_request_times(last_requests)
            lock.release()

        lock.acquire()

        #Get the access token and the expire time of the access token.
        (transactions['Access Token'],transactions['Access Expire Time']) = get_access(transactions['Access Token'],transactions['Access Expire Time'])

        #Track all the buy orders to make sure the transactions are up to date
        for ticker in transactions['Buy Orders']['Ticker']:

            track_buy_orders(transactions,ticker)
            transactions['Last Requests'][transactions['Min Request Index']]=time.time()

            last_requests=transactions['Last Requests']
            #Make sure we are not making too many requests to the TD Ameritrade API
            transactions['Min Request Index']=check_request_times(last_requests)

        #Track all the sell orders to make sure the transactions are up to date 
        for ticker in transactions['Current Stock']['Ticker']:

            #Check if there are limit sell orders
            if transactions['Current Stock']['Ticker'==ticker]['Limit Sell Price']>0:

                #Track the sell order (remove from limit sell arrays if filled)
                track_sell_orders(transactions,ticker)
                transactions['Last Requests'][transactions['Min Request Index']]=time.time()
            
                last_requests=transactions['Last Requests']
                #Make sure we are not making too many requests to the TD Ameritrade API
                transactions['Min Request Index']=check_request_times(last_requests)


        #Get the close price for all tickers and indexes (to be used for feature calculations)
        last_quotes = get_multi_quotes(transactions['Access Token'],','.join(transactions['Tickers Algorithm']+['SPX','NDX','DJX']))
        for ticker in (transactions['Tickers Algorithm']+['SPX','NDX','DJX']):
            #Get the close price for all tickers
            if ticker in last_quotes and 'regularMarketLastPrice' in last_quotes[ticker]:
                transactions[ticker]['Price'] = last_quotes['regularMarketLastPrice']

        #Calculate the latest feature data using the day's close price and save it to the transactions dictionary
        calc_features(transactions,transactions['Tickers Algorithm'])

        #Shift the feature data by one day (to get ready for market open tomorrow). Todays close will be 1 day in the past tomorrow.
        for ticker in transactions['Tickers Algorithm']:
            transactions[ticker]['Features'] = transactions[ticker]['Features'].shift(periods=1)
            #Remove the last row
            transactions[ticker]['Features'].drop(transactions[ticker]['Features'].tail(1).index,inplace=True)
        
        #Save transactions dictionary
        pickle.dump(transactions,open("Transactions.p","wb"))

        lock.release()


def check_request_times(last_requests):
    """
    Checks if we are making too many api requests and get the latest request time index
    last_requests - last requests made to the site. 
        If all of these are within one second, wait before making another call.

    Returns the index of the latest request time in transaction['Last Requests']
    """
    copy_requests = last_requests.copy()
    copy_requests.sort()

    if (copy_requests[1]-copy_requests[0])<1 and (copy_requests[1]-copy_requests[0])>0:
        time.sleep(1-(copy_requests[1]-copy_requests[0])) #Sleep to make sure we don't have too many requests per second.
    return last_requests.index(min(last_requests)) #Min Index



def place_buy_order(transactions,ticker):
    """
    Code to interface with TD Ameritrade to place a limit order to buy stock

    transactions - dictionary of info related to bot transactions
            'Previous Sell' - last sale price of stock
            'New Buy Proportion' - proportion below previous sell to buy stock
            'First Buy' - buy price to make the first purchase of stock
            

    Returns the status code of the request.
    """
    if transactions[ticker]['Stock Owned']==0 and transactions[ticker]['Previous Sell']>0:
        #We sold the stock. Set new buy price at New Buy Proportion below previous sell
        buy_price = transactions[ticker]['Previous Sell']*(1-transactions[ticker]['New Buy Proportion'])
    elif transactions[ticker]['Stock Owned']>0:
        #Place another limit buy order Buy proportion below previous buy
        buy_price = transactions[ticker]['Previous Buy']*(1-transactions[ticker]['Buy Proportion'])
    else:
        #We haven't bought or sold any of this stock yet
        buy_price = transactions[ticker]['First Buy']

    #Round buy price according to number of digits allowed (2 digits for >$1 stocks, 4 digits for <$1 stocks)
    buy_price = round(buy_price,transactions[ticker]['Max Digits'])

    #Make sure we have the funds to buy before placing the order
    if transactions[ticker]['Available Balance'] > buy_price*transactions[ticker]['Order Quantity']:

        #Build the order leg collections for placing orders on TD Ameritrade
        orderLegCollection_buy = [{ 'instrument':{'symbol': ticker,'assetType':'EQUITY'},
                                    'instruction':'BUY',
                                    'quantity':transactions[ticker]['Order Quantity']}]

        #Create main order request
        buy_request = build_order_request('SEAMLESS','GOOD_TILL_CANCEL','LIMIT',orderLegCollection_buy,'SINGLE',str(buy_price))

        try:
            #Make the actual post request
            post_order_response = post_order(transactions['Access Token'],buy_request)

            #Check request to make sure it successfully posted
            if post_order_response.status_code==201:
                transactions[ticker]['Available Balance']-= buy_price*transactions[ticker]['Order Quantity']
                #Reset the previous sell since we have a new buy order in place
                transactions[ticker]['Previous Sell']=0

                response_headers=post_order_response.headers
                #Get the order id of the buy order from the headers
                if 'Location' in response_headers:
                    order_id = int(response_headers['Location'].split('orders/')[1])

                    transactions[ticker]['Limit Buy ID'] = order_id
                    transactions[ticker]['Limit Buy Price'] = buy_price
                        
            return post_order_response.status_code
        except:
            transactions['API Errors'].append('Could not place buy order for {} at price {}'.format(ticker,buy_price))
            return 0


def track_buy_orders(transactions,ticker):
    """
    Determines if we successfully bought stock with our limit order. Updates transactions limit buy list and tickers list.
    transactions - dictionary containing information about transactions. 
        'Buy Orders' - DataFrame of current buy orders with columns Ticker, DateTime, Limit Buy ID, Limit Buy Price, Limit Buy Quantity
        'Current Stock' - DataFrame of current stock owned with columns Ticker, Buy Datetime, Buy Price, Buy Quantity, Limit Sell ID, Limit Sell Price
    ticker - the stock ticker that was previously bought

    returns:
        transactions - updates transactions['Buy Orders'] by removing ticker successfully bought
                       updates transactions['Current Stock'] by adding successfully bought stock to DataFrame

    CD 12/2020 - Update to use DataFrames

    """
    buy_order = transactions['Buy Orders']['Ticker'==ticker]
    buy_index = transactions['Buy Orders']['Ticker'==ticker].index()
    try:
        #Get the order
        limit_order=get_order_by_id(transactions['Access Token'],buy_order['Limit Buy ID'])
    
        #Check if the order was filled
        if limit_order['status']=='FILLED':

            #Add recently filled order to Current Stock
            transactions['Current Stock'] = transactions['Current Stock'].append({'Ticker': ticker,'Buy Date': datetime.datettime.now(),\
                'Buy Price': limit_order['price'],'Buy Quantity': limit_order['quantity'],'Limit Sell ID': 0,'Limit Sell Price': 0},ignore_index=True)

            #Remove buy order from transactions dictionary
            transactions['Buy Orders'].drop(index=buy_index)

    except:
        transactions['API Errors'].append('Could not get buy orders')



def track_sell_orders(transactions,ticker):
    """
    Determines if stock was successfully sold with our limit order. Updates transactions limit sell list.
    transactions - dictionary containing information about transactions. 
        'Current Stock' - DataFrame of current stock owned with columns Ticker, Buy Datetime, Buy Price, Buy Quantity, Limit Sell ID, Limit Sell Price
        'Past Orders' - DataFrame of past orders with columns Ticker, Buy Datetime, Buy Price, Buy Quantity, Sell Datetime, Sell Price, Sell Quantity
        'Tickers Today' - list of tickers monitored today to see if a buy order should be placed
    ticker - the stock ticker we are trading

    """
    sell_order = transactions['Current Stock']['Ticker'==ticker]
    sell_index = transactions['Current Stock']['Ticker'==ticker].index()
    try:
        #Get the order
        limit_order=get_order_by_id(transactions['Access Token'],sell_order['Limit Sell ID'])
    
        #Check if the order was filled
        if limit_order['status']=='FILLED':

            #Add recently filled order to Past Orders
            transactions['Past Orders'] = transactions['Past Orders'].append({'Ticker': ticker,'Buy Datetime': sell_order['Buy Datetime'],'Buy Price': sell_order['Buy Price'],\
                'Buy Quantity': sell_order['Buy Quantity'],'Sell Datetime': datetime.datetime.now(),'Sell Price': limit_order['price'],'Sell Quantity': limit_order['quantity']},ignore_index=True)

            #Remove from Current Stock
            transactions['Current Stock'].drop(index=sell_index)

            #Add to transactions['Tickers Today'] (the stock can be bought again now that it has been sold)
            transactions['Tickers Today'].append(ticker)

    except:
        transactions['API Errors'].append('Could not get sell orders')



def place_sell_order(transactions,ticker,sell_price):
    """
    Uses the threshold values to place a limit sell order for bought stock.
    Try to maximize profit by selling at twice the last threshold value.

    transactions - dictionary of info related to bot transactions
            transactions['Current Stock'] - DataFrame of current stock owned with columns Ticker, Buy Datetime, Buy Price, Buy Quantity, Limit Sell ID, Limit Sell Price
    ticker - the stock currently owned
    sell_price - the price to set when placing limit sell order
    """
    sell_order = transactions['Current Stock']['Ticker'==ticker]
    sell_index = transactions['Current Stock']['Ticker'==ticker].index()

    orderLegCollection_sell = [{'instrument':{'symbol': ticker,'assetType':'EQUITY'},
                            'instruction':'SELL',
                            'quantity':sell_order['Buy Quantity']}]

    #Create main order request
    sell_request = build_order_request('SEAMLESS','GOOD_TILL_CANCEL','LIMIT',orderLegCollection_sell,'SINGLE',str(sell_price))

    try:
        #Make the actual post request
        post_order_response = post_order(transactions['Access Token'],sell_request)

        #Check request to make sure it successfully posted
        if post_order_response.status_code==201:

            response_headers=post_order_response.headers
            #Get the order id of the sell order from the headers
            if 'Location' in response_headers:
                order_id = int(response_headers['Location'].split('orders/')[1])

                transactions['Current Orders'].loc[sell_index,'Limit Sell ID'] = order_id
                transactions['Current Orders'].loc[sell_index,'Limit Sell Price'] = sell_price
    
    except:
        transactions['API Errors'].append('Could not place buy order for {} at price {}'.format(ticker,sell_price))



def replace_sell_order(transactions,ticker,sell_price):
    """
    Replace the current sell order for the ticker with a new one at the new sell price

    transactions[Current Stock] - DataFrame of info on buy price and current sell orders
            'Limit Sell ID' - ID of a current sell order
            'Limit Sell Price' - price of a previous sell order
    transactions['Tickers Today'] - list of tickers monitored today to see if a buy order should be placed
    ticker - the stock being traded
    sell_price - the price to set for the replaced order
    """
    sell_order = transactions['Current Stock']['Ticker'==ticker]
    sell_index = transactions['Current Stock']['Ticker'==ticker].index()

    orderLegCollection_sell = [{'instrument':{'symbol': ticker,'assetType':'EQUITY'},
                            'instruction':'SELL',
                            'quantity':sell_order['Buy Quantity']}]

    #Create main order request
    sell_request = build_order_request('SEAMLESS','GOOD_TILL_CANCEL','LIMIT',orderLegCollection_sell,'SINGLE',str(sell_price))

    try:
        #Make the actual put request to replace order
        put_order_response = replace_order(transactions['Access Token'],sell_order['Limit Sell ID'],sell_request)

        #Check request to make sure it successfully posted
        if put_order_response.status_code==201:

            response_headers=put_order_response.headers
            #Get the order id of the buy order from the headers
            if 'Location' in response_headers:
                order_id = int(response_headers['Location'].split('orders/')[1])

                transactions['Current Orders'].loc[sell_index,'Limit Sell ID'] = order_id
                transactions['Current Orders'].loc[sell_index,'Limit Sell Price'] = sell_price
    
        #Try tracking the order to get latest information
        else:
            track_sell_orders(transactions,ticker)
    except:
        transactions['API Errors'].append('Could not replace sell order for {} at price {}'.format(ticker,sell_price))



def cancel_buy_orders(transactions,ticker):
    """
    Cancels a currently placed buy order
    transactions['Buy Orders'] - DataFrame containing currently placed limit buy orders.
        'Limit Buy ID' - order ID of current limit buy
        'Limit Buy Price' - price of current limit buy
        'Limit Buy Quantity' - quantity of stock ordered
    transactions['Tickers Today'] - list of tickers monitored today to see if a buy order should be placed
    ticker - the stock with the buy order being cancelled

    """
    buy_order = transactions['Buy Orders']['Ticker'==ticker]
    buy_index = transactions['Buy Orders']['Ticker'==ticker].index()
    try:
        #Cancel the order with the min price
        delete_response = delete_order(transactions['Access Token'],buy_order['Limit Buy ID'])

        if delete_response.status_code==200:
            #Add back the cost of the limit buy back to available balance, remove from 'Buy Orders', and add ticker back to list
            transactions['Available Balance'] += buy_order['Limit Buy Price']*buy_order['Order Quantity']
            transactions['Buy Orders'].drop(index=buy_index)
            transactions['Tickers Today'].append(ticker)

    except:
        transactions['API Errors'].append('Could not cancel order for {} at price {}'.format(ticker,transactions[ticker]['Limit Buy Price']))



        