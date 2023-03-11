"""
This bot uses the centroids from the K Means clustering algorithm to make trading decisions. 

It divides the money available and places buy orders on stocks with 'Strong Buy' labels.
Sells the stock when price is above purchase price and the stock data changes to a neutral or sell cluster centroid. 
Collects price and option information on the minute.

Description of threading - This bot creates 2 threads.
   Thread 1: Places buy and sell orders on TD Ameritrade Site   
   Thread 2: Get the latest stock and option prices from TD Ameritrade
   Thread 3: Looks for user's input and stops for debugging

"""


import threading,pyodbc,pickle
from TDAmeritrade_algorithm_centroids import ordering_bot,check_request_times
from TDAmeritrade_user import user_interface
from TDAmeritrade_get_prices import get_prices
#from TDAmeritrade_odbc import store_results,retrieve_open_orders
#from TDAmeritrade_excel import save_results



def time_diff(time_1,time_2):
    """
    Check the difference in days (to neareast hour) between time_2 and time_1
    time_1 - datetime object
    time_2 - datetime object
    Returns the difference between the two times to the nearest second
    """
    time_delta=time_2-time_1
    time_difference=time_delta.days*24*60*60+time_delta.seconds
    return time_difference #In seconds


def run_bot(transactions):
    """
    Runs a bot to buy and sell stock. 
    Loads the transactions dictionary to set up parameters for when to buy and sell stock. 

    Within the transactions dictionary there are dictionaries for each stock. The keys of each stock dictionary are:
        Thresholds - the thresholds that divide the clusters into different categories (strong buy uses the last two thresholds)
        Centroids - a dictionary where each key is a centroid and the values are feature data that represent the center of a cluster
        Centroid Labels - a pandas Series where each index is a centroid number and the values are a label (Strong Buy, Buy, Weak Buy, Neutral, Weak Sell, Sell, Strong Sell)
        Features - a DataFrame that contains the latest feature info on a ticker, use this to calculate real time feature values

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

        Current Stock - DataFrame of current stock owned; columns - Ticker, Buy Date, Buy Price, Buy Quantity, Limit Sell ID, Limit Sell Price
        Past Orders - DataFrame of past orders; columns - Ticker, Buy Date, Buy Price, Buy Quantity, Sell Date, Sell Price, Sell Quantity

    Stores latest results of transaction dictionary as a pickle file

    """
    #Keep track of any errors
    errors=[]

    #Start multithreading
    lock=threading.Lock() #Lock for transactions dictionary
    #Thread 1 - Place buy and sell orders on TD Ameritrade site
    ordering_thread=threading.Thread(target=ordering_bot,args=(lock,transactions,errors,))
    #Thread 2 -  Get the latest stock and option prices from TD Ameritrade
    price_thread=threading.Thread(target=get_prices,args=(lock,transactions,errors,))
    #Thread 3 - Check for user input and provide user with details on transactions and status
    user_thread=threading.Thread(target=user_interface,args=(lock,transactions,errors,))

    #Start the threads
    ordering_thread.start()
    price_thread.start()
    user_thread.start()

    #Wait for the threads to join
    ordering_thread.join()
    price_thread.join()
    user_thread.join()

    print("-------------")
    print("-------------")
    print("-------------")
    print("Here is a list of errors that ended the run.")
    print(errors)
    print("-------------")
    print("-------------")
    print("-------------")
    print(transactions)

    return errors
    








    
            
