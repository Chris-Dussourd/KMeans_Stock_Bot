

#Functions to interact with the user of bot
# update_user - Gives an update message to user about current status of bot and transactions made.

import datetime,json,pickle,time
from TDAmeritrade_algorithm_centroids import track_buy_orders,track_sell_orders,check_request_times
from TDAmeritrade_API import get_access


def user_interface(lock,transactions,errors):
    """
    User interaction. User can get information about the transactions
    """
    user_input="Random"
    while len(errors)==0:
        try:
            while user_input!="Check All" and user_input!="Stop" and user_input!="Check Ticker" \
                and user_input!="Track Orders" and user_input!="Update":
                
                user_input=input("Enter 'Check All', 'Check Ticker', 'Track Orders', 'Update' or 'Stop': ")
            
            if user_input=="Check All":
                lock.acquire()
                print("--------------------")
                print("--------------------")
                print(json.dumps(transactions))
                print("--------------------")
                print("--------------------")
                lock.release()
                #Reset user_input
                user_input="Random"

            elif user_input=="Check Ticker":

                lock.acquire()
                ticker = input("Please enter the ticker to check:  ")
                print("--------------------")
                print("--------------------")
                if ticker in transactions:
                    print(json.dumps(transactions[ticker]))
                print("--------------------")
                print("--------------------")
                lock.release()
                #Reset user_input
                user_input="Random"

            elif user_input=="Update":
                lock.acquire()
                while user_input == "Update":

                    ticker = input("Please enter the ticker to update:  ")
                    key = input("Please enter the key of the ticker dictionary to update:  ")
                    value_temp = input("Please enter the value to change it to:  ")
                    value_type = input("Please enter the type the value is (int,float,str):  ")
                    if value_type=="int":
                        value=int(value_temp)
                    elif value_type=="float":
                        value=float(value_temp)
                    else:
                        value=value_temp
                    if ticker in transactions and key in transactions[ticker]:
                        transactions[ticker][key]=value
                        print(transactions[ticker][key])
                    
                    user_input = input("Type 'Update' to keep updating or anything else to continue bot:  ")

                lock.release()

            elif user_input=="Track Orders":
                lock.acquire()

                ticker = input("What ticker do you want to track orders for:  ")
                instruction = input("Do you want to track buy or sell orders (enter: buy or sell):  ")
                if ticker in transactions:
                    if instruction=='buy':
                        track_buy_orders(transactions,ticker)
                    elif instruction=='sell':
                        track_sell_orders(transactions,ticker)

                lock.release()
                #Reset user_input
                user_input="Random"

            elif user_input=="Stop":
                errors.append('User asked to stop bot.')
        except:
            errors.append('Error in user interface.')


def update_transactions():

    #Load transactions dictionary
    transactions = pickle.load(open("Transactions.p","rb"))

    #Prompt user for information on what they want to update.
    user_input = 'Update'

    while user_input == 'Update':

        ticker = input('Please enter the ticker to update:  ')
        key = input('Please enter the key of the ticker to update:  ')
        value_temp = input("Please enter the value to change it to:  ")
        value_type = input("Please enter the type the value is (int,float,str):  ")
        if value_type=="int":
            value=int(value_temp)
        elif value_type=="float":
            value=float(value_temp)
        else:
            value=value_temp
        if ticker in transactions and key in transactions[ticker]:
            transactions[ticker][key]=value
            print(transactions[ticker][key])
        
        user_input = input("Type 'Update' to keep updating or anything else stop:  ")
        
    #Save transactions dictionary
    pickle.dump(transactions,open("Transactions.p","wb"))



def print_transactions():

    #Load transactions dictionary
    transactions = pickle.load(open("Transactions.p","rb"))

    for ticker in transactions['Tickers']:
        print('--------------------')
        print(ticker)
        print(transactions[ticker])
        print('--------------------')


def update_all_orders():

    #Load transactions dictionary
    transactions = pickle.load(open("Transactions.p","rb"))
    min_request_index=0

    #Make sure the access token is valid
    (transactions['Access Token'],transactions['Access Expire Time']) = get_access(transactions['Access Token'],transactions['Access Expire Time'])

    for ticker in transactions['Tickers']:
        
        track_buy_orders(transactions,ticker)
        track_sell_orders(transactions,ticker)

        transactions['Last Requests'][min_request_index]=time.time()
        last_requests=transactions['Last Requests']
        #Make sure we are not making too many requests to the TD Ameritrade API
        min_request_index=check_request_times(last_requests)

    #Save transactions dictionary
    pickle.dump(transactions,open("Transactions.p","wb"))


    