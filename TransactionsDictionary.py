"""
Initialize or update transaction history dictionary used by the trading bot.
Add general trading information as well as centroids and centroid labels from K means algorithm results.

CD 12/2020 Add centroids, lables, and features_df to transactions

CD 01/2021 - Add retrieving minute level, day level, and last update dates for centroids and features
           - Add update_transactions_date to let bot update latest day-level prices, minute-level prices, and features DataFrame
"""

import math,pickle,datetime,os
import pandas as pd
from file_organization import folder_directory,folder_structure,folder_directory_centroid
from TDAmeritrade_API import get_access#,get_account

#Initialize dictionary
def initialize_transactions():
    """
    Store information related to trading on each stock into a dictionary
    Store the dictionary as a pickle file ("transactions.p")
    """
    #Create a dictionary to save successful buys, sells, and money available to bot.
    transactions={}

    #Maximum number of times the bot can buy the same stock (only 1 is supported right now)
    transactions['Max Buys']=1

    #Available balance for the bot to use 
    transactions['Available Balance'] = 300

    #Minimum amount of money we can spend on one transaction
    transactions['Minimum Purchase'] = 100

    #Maximum amount of money we can spend on one purchase
    transactions['Maximum Purchase'] = 10000

    #Number of transactions to divide the available balance into unless it is under the 'Minimum Purchase' or over the 'Maximum Purchase'
    transactions['Number Transactions'] = 10 #Diversify Purchases into multiple transactions

    #Proportion of money from profits saved for taxes
    transactions['Save Proportion'] = 0.5

    #Features used for calculating centroids
    transactions['Features List'] = ['Bias','close','close1min','close2min','close5min','close10min','close30min','close1hr','SMA5','SMA10','SMA20','SMA50',\
                                    'EMA5','EMA10','EMA20','EMA50','Boll20','Boll50','MACDline','SignalLine','MACDhist','RSI','Stochastic%K','Stochastic%K-%D']

    #TD Ameritrade Limitations
    requests_allowed=2      #Minimum number of requests allowed per second to TD Ameritrade API
    transactions['Last Requests']=[0]*requests_allowed
    #Initialize min request index to 0 since no transactions have started yet
    transactions['Min Request Index']=0
    
    #Initialize access information in dictionary
    transactions['Access Token']=''
    transactions['Access Expire Time']=0

    #Store API call errors to see if we need to manually place orders
    transactions['API Errors']=[]

    #Get the last update time for the transactions dictionary
    transactions['Last Update']=datetime.datetime.now()

    #Initialize list of tickers that are monitored for today
    transactions['Tickers Today']=[]

    #Create a list of tickers to get minute level and day level prices (must contain all tickers in 'Tickers Algorithm')
    transactions['Tickers Prices']=folder_structure.keys()

    #Create a list of options to get minute level and day level prices
    transactions['Options']

    #Create a list of tickers for algorithm to monitor
    transactions['Tickers Algorithm']=[]
    for ticker in folder_structure:
        
        #Initialize dictionary for each ticker, minute prices, and daily prices for each ticker in transactions menu
        transactions[ticker] = {}
        transactions[ticker]['Minute Prices'] = pd.DataFrame({'datetime':[],'close':[]})
        transactions[ticker]['Daily Prices'] = pd.DataFrame({'datetime':[],'high':[],'low':[],'open':[],'close':[]})

        #Make sure folder for this ticker is in the directory
        if ticker in os.listdir(folder_directory_centroid):

            #Path to get to each ticker data set
            ticker_path = os.path.join(folder_directory_centroid,ticker)

            #Get all of the filenames for centroid labels, centroid values, and latest feature info
            filenames_centroids = [f for f in os.listdir(ticker_path) if os.path.isfile(os.path.join(ticker_path,f))]

            #Add all the centroid values and labels to the transactions dictionary
            if 'centroids.p' in filenames_centroids and 'centroidslabels.p' in filenames_centroids and 'thresholds.p' in filenames_centroids \
                and 'features_df.p' in filenames_centroids and 'featurescaling.p' in filenames_centroids and 'minute_data.p' in filenames_centroids \
                    and 'day_data.p' in filenames_centroids and 'monthly_ave_volume' in filenames_centroids and 'centroiddate_featuredate.p' in filenames_centroids:

                #Add ticker to list since centroids and centroid labels are available
                transactions['Tickers Algorithm'].append(ticker)

                #Centroids and lables for centroids (strong buy, buy, weak buy, neutral, weak sell, sell, and strong sell)
                transactions[ticker]['Centroids'] = pickle.load(open(os.path.join(ticker_path,'centroids.p'),"rb"))
                transactions[ticker]['Centroid Labels'] = pickle.load(open(os.path.join(ticker_path,'centroid_labels.p'),"rb"))

                #Thresholds contains a list of percent gain/losses for each centroid category
                transactions[ticker]['Thresholds'] = pickle.load(open(os.path.join(ticker_path,'thresholds.p'),"rb"))

                #Add a DataFrame of Lastest Feature Data for tickers (used to calculate new feature values real time)
                transactions[ticker]['Features'] = pickle.load(open(os.path.join(ticker_path,'features_df.p'),"rb"))

                #Add a DataFrame with two rows (Min and Max) that contain the min and max for each feature (use for feature scaling)
                transactions[ticker]['Feature Scaling'] = pickle.load(open(os.path.join(ticker_path,'featurescaling.p'),"rb"))

                #Add a list of minute level prices to transactions dictionary for feature calculation
                transactions[ticker]['Minute Prices'] = pickle.load(open(os.path.join(ticker_path,'minute_data.p'),"rb"))

                #Add a list of day level prices to transactions dictionary for feature calculation
                transactions[ticker]['Daily Prices'] = pickle.load(open(os.path.join(ticker_path,'day_data.p'),"rb"))

                #Add the average monthly volume of the ticker
                transactions[ticker]['Average Volume'] = pickle.load(open(os.path.join(ticker_path,'monthly_ave_volume.p'),"rb"))

                #Add the last update dates for centroid data and feature data
                (transactions[ticker]['Centroid Last Update'],transactions[ticker]['Feature Last Update']) = pickle.load(open(os.path.join(ticker_path,'centroiddate_featuredate.p'),"rb"))
            
    #Create a DataFrame that includes information on current and past orders
    transactions['Buy Orders'] = pd.DataFrame(columns = ['Ticker','Datetime','Limit Buy ID','Limit Buy Price','Limit Buy Quantity'])
    transactions['Current Stock'] = pd.DataFrame(columns = ['Ticker','Buy Datetime','Buy Price','Buy Quantity','Limit Sell ID','Limit Sell Price'])
    transactions['Past Orders'] = pd.DataFrame(columns = ['Ticker','Buy Datetime','Buy Price','Buy Quantity','Sell Datetime','Sell Price','Sell Quantity'])

    #Period of time being predicted in the future (used for when to sell the stock)
    transactions['Predict Period'] = 2
    transactions['Predict Period Type'] = 'day'

    pickle.dump(transactions,open("Transactions.p","wb"))


def add_stock_transactions(ticker):
    """
    Adds a new stock to the transactions dictionary
    ticker - the stock symbol 
    transactions - the transactions dictionary to update
    """

    transactions = pickle.load(open("Transactions.p","rb"))

    #Make sure folder for this ticker is in the directory
    if ticker in os.listdir(folder_directory_centroid):

        #Path to get to each ticker data set
        ticker_path = os.path.join(folder_directory_centroid,ticker)

        #Get all of the filenames for centroid labels, centroid values, and latest feature info
        filenames_centroids = [f for f in os.listdir(ticker_path) if os.path.isfile(os.path.join(ticker_path,f))]

        #Add all the centroid values and labels to the transactions dictionary
        if 'centroids.p' in filenames_centroids and 'centroidslabels.p' in filenames_centroids and 'thresholds.p' in filenames_centroids \
                and 'features_df.p' in filenames_centroids and 'featurescaling.p' in filenames_centroids and 'minute_data.p' in filenames_centroids \
                    and 'day_data.p' in filenames_centroids and 'monthly_ave_volume' in filenames_centroids and 'centroiddate_featuredate.p' in filenames_centroids:

            #Add ticker to list since centroids and centroid labels are available
            transactions['Tickers Algorithm'].append(ticker)
            transactions[ticker]={}

            #Centroids and lables for centroids (strong buy, buy, weak buy, neutral, weak sell, sell, and strong sell)
            transactions[ticker]['Centroids'] = pickle.load(open(os.path.join(ticker_path,'centroids.p'),"rb"))
            transactions[ticker]['Centroid Labels'] = pickle.load(open(os.path.join(ticker_path,'centroid_labels.p'),"rb"))

            #Thresholds contains a list of percent gain/losses for each centroid category
            transactions[ticker]['Thresholds'] = pickle.load(open(os.path.join(ticker_path,'thresholds.p'),"rb"))

            #Add a DataFrame of Lastest Feature Data for tickers (used to calculate new feature values real time)
            transactions[ticker]['Features'] = pickle.load(open(os.path.join(ticker_path,'features_df.p'),"rb"))

            #Add a DataFrame with two rows (Min and Max) that contain the min and max for each feature (use for feature scaling)
            transactions[ticker]['Feature Scaling'] = pickle.load(open(os.path.join(ticker_path,'featurescaling.p'),"rb"))

            #Add a list of minute level prices to transactions dictionary for feature calculation
            transactions[ticker]['Minute Prices'] = pickle.load(open(os.path.join(ticker_path,'minute_data.p'),"rb"))

            #Add a list of day level prices to transactions dictionary for feature calculation
            transactions[ticker]['Daily Prices'] = pickle.load(open(os.path.join(ticker_path,'day_data.p'),"rb"))

            #Add the average monthly volume of the ticker
            transactions[ticker]['Average Volume'] = pickle.load(open(os.path.join(ticker_path,'monthly_ave_volume.p'),"rb"))

            #Add the last update dates for centroid data and feature data
            (transactions[ticker]['Centroid Last Update'],transactions[ticker]['Feature Last Update']) = pickle.load(open(os.path.join(ticker_path,'centroiddate_featuredate.p'),"rb"))

            #Save Updates
            pickle.dump(transactions,open("Transactions.p","wb"))


def update_transactions_date(transactions,start_date,end_date):
    """
    Update the transactions dictionary with day-level prices, minute-level prices, and latest feature data

    transactions - the dictionary of transction info
        transactions[ticker] - for each ticker update the following keys
            'Minute Prices' - minute level price data
            'Daily Prices' - day level price data
            'Features' - latest feature data
            'Feature Last Update' - set to end_date
    start_date - update values in transaction dictionary starting at this date
    end_date - update values in transaction dictionary ending at this date
    """

    #Loop through each ticker in algorithm
    for ticker in transactions['Tickers Algorithm']:

        #Make sure folder for this ticker is in the directory
        if ticker in os.listdir(folder_directory_centroid):

            #Path to get to each ticker data set
            ticker_path = os.path.join(folder_directory_centroid,ticker)

            #Add a list of minute level prices to transactions dictionary for feature calculation
            #transactions[ticker]['Minute Prices'] = pickle.load(open(os.path.join(ticker_path,'minute_data.p'),"rb"))

            #Add a list of day level prices to transactions dictionary for feature calculation
            #transactions[ticker]['Daily Prices'] = pickle.load(open(os.path.join(ticker_path,'day_data.p'),"rb"))

            #Add a DataFrame of Lastest Feature Data for tickers (used to calculate new feature values real time)
            #transactions[ticker]['Features'] = pickle.load(open(os.path.join(ticker_path,'features_df.p'),"rb"))

            #Add the last update dates for centroid data and feature data
            transactions[ticker]['Feature Last Update'] = end_date


def monthly_transactions_update(transactions):
    """
    Update the transactions dictionary with centroid, price, and feature data for previous month
    Note: It is recommended to run this on a weekend since the centroid data code takes time

    transactions - the dictionary of transction info
        transactions[ticker] - for each ticker update the following keys
            'Centroids' - the centroids found using the k means algorithm
            'Centroid Labels' - the 
            'Minute Prices' - minute level price data
            'Daily Prices' - day level price data
            'Features' - latest feature data
            'Feature Last Update' - set to end_date
    start_date - update values in transaction dictionary starting at this date
    end_date - update values in transaction dictionary ending at this date
    """

    #Loop through each ticker in algorithm
    for ticker in transactions['Tickers Algorithm']:

        #Make sure folder for this ticker is in the directory
        if ticker in os.listdir(folder_directory_centroid):

            #Path to get to each ticker data set
            ticker_path = os.path.join(folder_directory_centroid,ticker)

            #Add ticker to list since centroids and centroid labels are available
            transactions['Tickers Algorithm'].append(ticker)
            transactions[ticker]={}

            #Centroids and lables for centroids (strong buy, buy, weak buy, neutral, weak sell, sell, and strong sell)
            transactions[ticker]['Centroids'] = pickle.load(open(os.path.join(ticker_path,'centroids.p'),"rb"))
            transactions[ticker]['Centroid Labels'] = pickle.load(open(os.path.join(ticker_path,'centroid_labels.p'),"rb"))

            #Thresholds contains a list of percent gain/losses for each centroid category
            transactions[ticker]['Thresholds'] = pickle.load(open(os.path.join(ticker_path,'thresholds.p'),"rb"))

            #Add a DataFrame of Lastest Feature Data for tickers (used to calculate new feature values real time)
            transactions[ticker]['Features'] = pickle.load(open(os.path.join(ticker_path,'features_df.p'),"rb"))

            #Add a DataFrame with two rows (Min and Max) that contain the min and max for each feature (use for feature scaling)
            transactions[ticker]['Feature Scaling'] = pickle.load(open(os.path.join(ticker_path,'featurescaling.p'),"rb"))

            #Add a list of minute level prices to transactions dictionary for feature calculation
            transactions[ticker]['Minute Prices'] = pickle.load(open(os.path.join(ticker_path,'minute_data.p'),"rb"))

            #Add a list of day level prices to transactions dictionary for feature calculation
            transactions[ticker]['Daily Prices'] = pickle.load(open(os.path.join(ticker_path,'day_data.p'),"rb"))

            #Add the average monthly volume of the ticker
            transactions[ticker]['Average Volume'] = pickle.load(open(os.path.join(ticker_path,'monthly_ave_volume.p'),"rb"))

            #Add the last update dates for centroid data and feature data
            (transactions[ticker]['Centroid Last Update'],transactions[ticker]['Feature Last Update']) = pickle.load(open(os.path.join(ticker_path,'centroiddate_featuredate.p'),"rb"))

            #Save Updates
            pickle.dump(transactions,open("Transactions.p","wb"))

"""
def recover_transactions(transactions):


    [access_token,expire_time]=get_access('','')
    account_data = get_account(access_token)

    positions = account_data['securitiesAccount']['positions']
    orders = account_data['securitiesAccount']['orderStrategies']

    #Update limit buy and sell orders using currently queued orders
    for order in orders:

        if order['status'] == 'WORKING':

            ticker = order['orderLegCollection'][0]['instrument']['symbol']
            if ticker in transactions['Tickers']:

                if order['orderLegCollection'][0]['instruction']=='BUY':

                    transactions[ticker]['Limit Buy ID'] = order['orderId']
                    transactions[ticker]['Limit Buy Price'] = order['price']

                elif order['orderLegCollection'][0]['instruction']=='SELL':

                    transactions[ticker]['Limit Sell ID'] = order['orderId']
                    transactions[ticker]['Limit Sell Price'] = order['price']


    #Update stock owned and after buy price
    for position in positions:

        ticker = position['instrument']['symbol']
        if ticker in transactions['Tickers']:
            
            #Update stock owned and average price
            transactions[ticker]['Available Balance'] -= (position['longQuantity']-transactions[ticker]['Stock Owned'])*position['averagePrice']
            transactions[ticker]['Stock Owned'] = position['longQuantity']
            transactions[ticker]['Average Buy'] = position['averagePrice']
            #Update previous buy depending on how much stock we bought
            if position['longQuantity']==transactions[ticker]['Order Quantity']:
                transactions[ticker]['Previous Buy'] = position['averagePrice']
            elif position['longQuantity']>2*transactions[ticker]['Order Quantity']:
                transactions[ticker]['Previous Buy'] = position['averagePrice']*(1-transactions[ticker]['Buy Proportion'])
            else:
                transactions[ticker]['Previous Buy'] = position['averagePrice']*(1-transactions[ticker]['Buy Proportion']/2)

    
"""


