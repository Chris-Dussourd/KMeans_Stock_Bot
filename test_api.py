
import requests,json,copy,datetime,auth_info
from TDAmeritrade_API import get_access,get_orders,delete_order,post_order,replace_order,build_order_request

errors=[]

access_token = ''
access_expire_time = 0

#Get the access token, access dict contains the token and expire time
(access_token,access_expire_time) = get_access(access_token,access_expire_time)
print(access_token)
print(access_expire_time)


"""
#Get movers for $COMPX
headers={'Authorization': 'Bearer {}'.format(access_token)}
symbol='$DJI'
mover_url = 'https://api.tdameritrade.com/v1/marketdata/{}/movers'.format(symbol)
params = {'apikey':client_id,
          'direction':'down',
          'change':'percent'}

mover_data_json = requests.get(url=mover_url,headers=headers,params=params)
mover_data=mover_data_json.json()
print(mover_data)
"""


#Get an option chain 
fromDate = '2021-01-01'
toDate = '2021-12-31'

headers={'Authorization': 'Bearer {}'.format(access_token)}
symbol='NIO'
quote_url = 'https://api.tdameritrade.com/v1/marketdata/chains'
params={'symbol':symbol,
        'fromDate': fromDate,
        'toDate': toDate,
        'strikeCount':10}

option_data_json=requests.get(url=quote_url,headers=headers,params=params)
option_data=option_data_json.json()
print(option_data)


"""
#Get quote from Amazon
headers={'Authorization': 'Bearer {}'.format(access_token)}
symbol='AMZN'
quote_url = 'https://api.tdameritrade.com/v1/marketdata/{}/quotes'.format(symbol)
params={'apikey':auth_info.client_id}

quote_data_json=requests.get(url=quote_url,headers=headers,params=params)
quote_data=quote_data_json.json()
print(quote_data)
"""

"""
#Get option quotes

headers={'Authorization': 'Bearer {}'.format(access_token)}
symbol='NIO_011521C48,NIO_011521C49'
quote_url = 'https://api.tdameritrade.com/v1/marketdata/quotes'
params={'apikey':auth_info.client_id,'symbol': symbol}

quote_data_json=requests.get(url=quote_url,headers=headers,params=params)
quote_data=quote_data_json.json()
print(quote_data)
"""


"""
#Get orders from my account
#start_date = datetime.datetime(2020,5,4).isoformat()
#end_date = datetime.datetime(2020,5,4).isoformat()
start_date = '2020-05-05'
end_date = '2020-05-05'
status='QUEUED'

orders_response = get_orders(access_token,start_date,end_date,status)
print('Symbol       Instruction      Price     Quantity    OrderType      ComplexOrderStrategyType     Status   EnteredTime     ClosedTime')
print(orders_response)
"""


"""
#Modify an order on my account
order_id = 1045843829
#Define parameters for the new order. 
symbol = 'NMHLY'
session = 'NORMAL' # during all trading hours (extended hours)
duration = 'GOOD_TILL_CANCEL'
orderType = 'LIMIT'
quantity = '300'
price = '0.81'
assetType = 'EQUITY'
instruction = 'SELL'
orderStrategyType = 'SINGLE'

#Add the information into json format that api understands
orderLegCollection = [{'instrument':{'symbol': symbol,'assetType':assetType},
                       'instruction':instruction,
                       'quantity':quantity}]

order_info=build_order_request(session,duration,orderType,orderLegCollection,orderStrategyType,price)

#Post the order on TD Ameritrade and check the response
replace_order_response=replace_order(access_token,order_id,order_info)

response_headers=replace_order_response.headers
#Get the order id from the headers
if 'Location' in response_headers:
    order_id = response_headers['Location'].split('orders/')[1]
    print(order_id)
"""


"""
#Delete a placed order on my account
order_id = 1045843721
order_status = delete_order(access_token,order_id)
print(order_status)
"""

"""
#Get account information (balances, open orders, positions)
headers={'Authorization': 'Bearer {}'.format(access_token)}
account_url = 'https://api.tdameritrade.com/v1/accounts/{}'.format(misc_info.account_num)

params = {'fields':'positions,orders'}
account_info_json = requests.get(url=account_url,headers=headers,params=params)
account_info = account_info_json.json()

print(account_info)
"""


"""
#Place a new order on my account
#Define parameters for the order. We placing a limit order for 1 share of NAIL at $11.15 per share.
symbol = 'GE'
session = 'SEAMLESS' # during all trading hours (extended hours)
duration = 'GOOD_TILL_CANCEL'
orderType = 'LIMIT'
quantity = '1'
price = '3.001'
assetType = 'EQUITY'
instruction = 'BUY'
orderStrategyType = 'SINGLE'

#Add the information into json format that api understands
orderLegCollection = [{'instrument':{'symbol': symbol,'assetType':assetType},
                       'instruction':instruction,
                       'quantity':quantity}]

order_info=build_order_request(session,duration,orderType,orderLegCollection,orderStrategyType,price)

#Post the order on TD Ameritrade and check the response
place_order_response=post_order(access_token,order_info)

response_headers=place_order_response.headers
#Get the order id from the headers
if 'Location' in response_headers:
    order_id = response_headers['Location'].split('orders/')[1]
    print(order_id)
"""


"""
#Complex orders - one triggers another with automatic 
#Define parameters for the order. We placing a limit order for 1 share of GE at $3 per share.
symbol = 'GE'
duration = 'DAY'
orderType = 'LIMIT'
buy_quantity = '3'
sell_quantity = '1'
buy_price = '3'
sell_gain = 0.01 #Amount we are gaining from selling it.
assetType = 'EQUITY'

#Add the information into json format that api understands
orderLegCollection_buy = [{'instrument':{'symbol': symbol,'assetType':assetType},
                           'instruction':'BUY',
                           'quantity':buy_quantity}]

orderLegCollection_sell = [{'instrument':{'symbol': symbol,'assetType':assetType},
                           'instruction':'SELL',
                           'quantity':sell_quantity}]

childOrders_template = {'session':'NORMAL',
                        'duration':duration,
                        'orderType':orderType,
                        'orderLegCollection':orderLegCollection_sell,
                        'orderStrategyType':'SINGLE'}

child_orders=[]
for n in range(int(int(buy_quantity)/int(sell_quantity))):
    sell_price = (n+1)*sell_gain+float(buy_price)
    #Convert sell price to string
    sell_price_str=str(round(sell_price,3))

    #Build child order array
    child_orders.append(build_order_request('NORMAL','GOOD_TILL_CANCEL','LIMIT',orderLegCollection_sell,'SINGLE',sell_price_str))


#Create the buy order request and add in the grandchild orders to the request
order_info = build_order_request('NORMAL','DAY','LIMIT',orderLegCollection_buy,'TRIGGER',buy_price)
#Child orders (grandchildren of original order) of the child buy orders
order_info['childOrderStrategies']=child_orders

#Post the order on TD Ameritrade and check the response
place_order_response=post_order(access_token,order_info)

print(place_order_response)

response_headers=place_order_response.headers
#Get the order id from the headers
if 'Location' in response_headers:
    order_id = response_headers['Location'].split('orders/')[1]
    print(order_id)
"""
