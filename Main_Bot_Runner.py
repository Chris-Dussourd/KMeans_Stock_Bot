from Bot_Centroids import run_bot
from TransactionsDictionary import initialize_transactions#,recover_transactions
import pickle, json, datetime, time

initialize = 0
#recover=0

if initialize:
    initialize_transactions()
else:
    #Load transactions dictionary
    transactions = pickle.load(open("Transactions.p","rb"))

#Recover old info
#if recover:
#    recover_transactions(transactions)
errors=[]

while len(errors)==0:
    #Reset API errors list
    transactions['API Errors']=[]

    errors = run_bot(transactions)

    