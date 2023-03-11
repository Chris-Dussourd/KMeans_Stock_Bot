
"""
Main file to be called for performing Machine Learning on stock price data 
    1. Extract and align price into minute level and day level pandas DataFrames using tickers_input
    2. Add stock features to input data by calculating the features in features_list. Scale features.
    3. Extract output data by shifting input, removing nan, and categorizing output 
    4. Pass input into a K-means learning algorithm
    5. Save variables in pickle files and the transactions dictionary to be used by trading bot.

CD 02/2021 - Update to be called by the stock bot and use the transaction dictionary.
           - Only perform K-means and add ability to run K-means multiple times.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime,math,pickle,os,calendar,time
from file_organization import folder_structure,folder_directory_centroid
from StockFeatures import add_features,feature_scaling
from pathlib import Path
from KMeans_functions import k_means_clustering,assign_data,label_centroids,find_best_labels
from PreprocessStockData import extract_input_data,extract_output_data,remove_nan_rows,categorize_output


def kmeans_machine_learning(start_date,end_date,predict_period,predict_periodType,run_time,tickers_list=[]):
    """
    Runs the K-means machine learning algorithm for tickers in the folder structure or in tickers_list (if passed in).
    Saves the results of the algorithm in a pickle file to be used by a trading bot.

    start_date - start time period to get stock price data
    end_date - end time period to get stock price data
    predict_period - time to predict in the future 
    predict_periodType - period type ('minute' or 'day) 
    algorithm - the learning algorithm to use for stock price prediction
    run_time - number of seconds available for running the algorithm; determines how many times the k-means algorithm is repeated.
    tickers_list - list of tickers to run the algorithm on
    """
    #Algorithm start time
    start_time = time.time()

    if not tickers_list:
        tickers_list = folder_structure

    ticker_num = 1 #Count the tickers to divide the time for k-means algorithm equally between them
    time_per_ticker = run_time/len(tickers_list)

    #Save number of strong buys for each ticker
    strong_buy_dict = {}

    #Loop through and perform K-means on all output tickers
    #The algorithm attempts to predict the price of ticker_output. It assumes ticker_output is also in tickers_input
    for ticker_output in tickers_list:
    
        #Use these tickers to help guess the price of a chosen ticker. The first ticker must be the output ticker
        tickers_input = [ticker_output]


        #Loads stock price data in data range from pickle files 
        #Aligns data between tickers and creates minute level and day level pandas DataFrames
        (price_minute_df,price_daily_df) = extract_input_data(tickers_input,start_date,end_date)

        #features_list = ['close','close1min','close2min','close5min','close10min','close30min','close1hr','SMA5','SMA10','SMA20','SMA50',\
        #    'EMA5','EMA10','EMA20','EMA50','Boll20','Boll50','MACDline','SignalLine','MACDhist','RSI','Stochastic%K','Stochastic%K-%D']
        features_list = ['Boll20','Boll50','MACDline','SignalLine','MACDhist','RSI','Stochastic%K','Stochastic%K-%D']

        num_clusters = 100 #Number of clusters used for K-means algorithm
        twotests = True #Use two test sets for testing algorithm

        #Add all features to the input DataFrame (the function also adds the 'Bias' feature)
        (input_data_df_all,input_data_daily_df_all) = add_features(price_minute_df,price_daily_df,ticker_output,'all','close_'+ticker_output)
        
        #Create dataframes of input data only used for the machine learning
        input_data_df = input_data_df_all.copy(deep=True)
        input_data_daily_df = input_data_daily_df_all.copy(deep=True)
        drop_columns=[]
        for col in input_data_df_all:
            if col.split('_')[0] not in features_list and col != 'Date' and col != 'datetime' and col != 'close_'+ticker_output:
                drop_columns.append(col)
        input_data_df.drop(columns=drop_columns,inplace=True)
        input_data_daily_df.drop(columns=drop_columns,inplace=True)

        #Scale Features before running learning algorithm
        (input_scaled_df,input_data_df,minmax_df,input_scaled_daily_df,input_data_daily_df) = feature_scaling(input_data_df,input_data_daily_df,'min-max')

        #Get the output data by shifting the input data
        (output_data_df,output_scaled) = extract_output_data(input_data_df,input_scaled_df,'close_'+ticker_output,predict_period,predict_periodType)

        #Remove nan rows from output and input data
        (input_data_df,input_scaled_df,output_data_df,output_scaled) = remove_nan_rows(input_data_df,input_scaled_df,output_data_df,output_scaled)

        #Remove 'Date' and 'datetime' column in input_data (the date is not being used as a feature)
        input_scaled_df = input_scaled_df.drop(columns=['datetime','Date'])
        input_scaled_daily_df = input_scaled_daily_df.drop(columns=['datetime','Date'])
        
        if twotests:
            train_division = 3/5
            test_division = 4/5
            #Separate input and output data into training (3/5 of data), test1 set (1/5 of data), and test2 set (1/5 of data)
            train_input_df = input_scaled_df.loc[0:math.floor(input_scaled_df.shape[0]*train_division)]
            train_output = output_scaled.loc[0:math.floor(output_scaled.shape[0]*train_division)]
            test_input_df = input_scaled_df.loc[math.floor(input_scaled_df.shape[0]*train_division+1):math.floor(input_scaled_df.shape[0]*test_division)]
            test_output = output_scaled.loc[math.floor(output_scaled.shape[0]*train_division+1):math.floor(output_scaled.shape[0]*test_division)]
            test2_input_df = input_scaled_df.loc[math.floor(input_scaled_df.shape[0]*test_division+1):]
            test2_output = output_scaled.loc[math.floor(output_scaled.shape[0]*test_division+1):]

        else:
            train_division = 2/3
            test_division = 1
            #Separate input and output data into training (2/3 of data) and test sets (1/3 of data)
            train_input_df = input_scaled_df.loc[0:math.floor(input_scaled_df.shape[0]*train_division)]
            train_output = output_scaled.loc[0:math.floor(output_scaled.shape[0]*train_division)]
            test_input_df = input_scaled_df.loc[math.floor(input_scaled_df.shape[0]*train_division+1):]
            test_output = output_scaled.loc[math.floor(output_scaled.shape[0]*train_division+1):]
            test2_input_df = pd.DataFrame()
            test2_output = pd.DataFrame()

        num_cat=8
        #Categorize output data. Divide into equal segments based on percent gained/lossed
        (train_output_cat,test_output_cat,test2_output_cat,threshold) = categorize_output(input_data_df,output_data_df,train_output,test_output,'close_'+ticker_output,num_cat,test2_output)

        #Remove the close price to be predicted from input data if it's not in the features list
        if 'close' not in features_list:
            train_input_df.drop(columns='close_'+ticker_output,inplace=True)
            test_input_df.drop(columns='close_'+ticker_output,inplace=True)
            test2_input_df.drop(columns='close_'+ticker_output,inplace=True)

        #Scale output categories from 0 to 1
        train_data_df = train_input_df.copy(deep=True)
        train_data_df['Output Cat'] = train_output_cat/train_output_cat.max() #Scale it from 0 to 1
        train_datetimes = input_data_df.loc[0:math.floor(input_data_df.shape[0]*train_division)]['datetime']

        strong_buy_count_best = 0 #Initialize previous strong buy count as zero
        #Run K-means clustering on data while we have time to run it or until algorithm has run 10 times
        run_num = 0
        while time.time() - start_time < time_per_ticker*ticker_num and run_num < 10
            cluster_centroids = k_means_clustering(train_data_df,num_clusters)

            #For each cluster, get the output categories and input datetimes in the training set
            (cluster_train_output_cat_temp,cluster_train_datetime_temp) = assign_data(train_data_df,'Output Cat',cluster_centroids,train_datetimes)

            #Remove cluster centroids that have no data assigned to them
            k_new = 0
            drop_clusters = []
            cluster_train_output_cat = {}
            cluster_train_datetime = {}
            for k in cluster_train_output_cat_temp:
                #Add cluster to new dictionary if temporary one is not empty for cluster k.
                if ~cluster_train_output_cat_temp[k].empty:
                    cluster_train_output_cat[k_new] = cluster_train_output_cat_temp[k]
                    cluster_train_datetime[k_new] = cluster_train_datetime_temp[k]
                    k_new=k_new+1
                else:
                    drop_clusters.append(k)
            #Remove centroids with no training data assigned
            cluster_centroids.drop(drop_clusters).reset_index(drop=True)
                
            #For each cluster, get the output categories and input datetimes in the test set
            test_data_df = test_input_df.copy(deep=True).reset_index(drop=True)
            test_data_df['Output Cat'] = (test_output_cat/test_output_cat.max()).reset_index(drop=True) #Scale it from 0 to 1
            test_datetimes = input_data_df.loc[math.floor(input_data_df.shape[0]*train_division+1):math.floor(input_data_df.shape[0]*test_division)]['datetime'].reset_index(drop=True)
            (cluster_test_output_cat,cluster_test_datetime) = assign_data(test_data_df,'Output Cat',cluster_centroids,test_datetimes)

            if twotests:
                #For each cluster, get the output categories and input datetimes in the test2 set
                test2_data_df = test2_input_df.copy(deep=True).reset_index(drop=True)
                test2_data_df['Output Cat'] = (test2_output_cat/test2_output_cat.max()).reset_index(drop=True) #Scale it from 0 to 1
                test2_datetimes = input_data_df.loc[math.floor(input_data_df.shape[0]*test_division+1):]['datetime'].reset_index(drop=True)
                (cluster_test2_output_cat,cluster_test2_datetime) = assign_data(test2_data_df,'Output Cat',cluster_centroids,test2_datetimes)

            if twotests:
                centroid_labels = label_centroids(threshold,cluster_train_output_cat,cluster_test_output_cat,cluster_test2_output_cat)
            else:
                centroid_labels = label_centroids(threshold,cluster_train_output_cat,cluster_test_output_cat)

            #Count the number of strong buys
            strong_buy_count = centroid_labels[centroid_labels=='Strong Buy'].count()

            #Save this run of k-means if this is the first run
            if run_num==0:
                #Save only the top three strong buy labels. Label the others as buys
                centroid_labels_best = find_best_strong_buy(centroid_labels,3)

            #Save this run if there are more strong buys
            elif strong_buy_count_best < 3 and strong_buy_count > strong_buy_count_best:
                #Save only the top three strong buy labels. Try adding the previous best centroid labels to this set.
                centroid_labels_best = find_best_strong_buy(centroid_labels,3,centroid_labels_best)

            #Try adding this run to the previous best run
            elif strong_buy_count>0:
                #Add current strong buy centroid labels to previous best if they improve algorithm
                centroid_labels_best = find_best_strong_buy(centroid_labels_best,3,centroid_labels)
                
            #Find the count of the best strong buy centroids
            strong_buy_count_best = centroid_labels_best[centroid_labels_best=='Strong Buy'].count()

            run_num = run_num + 1

        #Keep track of strong buys per ticker
        strong_buy_dict[ticker_output] = strong_buy_count_best

        #Save additional information for a stock bot to use
        path_centroid = os.path.join(folder_directory_centroid,ticker_output)
        
        #Save the centroid values themselves
        pickle.dump(cluster_centroids,open(os.path.join(path_centroid,'centroids.p'),"wb"))
        
        #Save the centroid labels (strong buy, buy, weak buy, neutral, weak sell, sell, strong sell)
        pickle.dump(centroid_labels,open(os.path.join(path_centroid,'centroid_labels.p'),"wb"))

        #Save thresholds values - percent gain/loss values that split output into each output category
        pickle.dump(threshold,open(os.path.join(path_centroid,'thresholds.p'),"wb"))

        #Save the unscaled features for the last three values
        features_df = input_data_daily_df_all.copy(deep=True).iloc[-3:]
        #Remove the ticker output from the name of each feature
        for col in features_df:
            if len(col.split('_')) > 1:
                if col.split('_')[1]==ticker_output:
                    features_df = features_df.rename(columns={col: col.split('_')[0]})
                    minmax_df = minmax_df.rename(columns={col: col.split('_')[0]})
        features_df.index = range(1,4)
        #Save DataFrame of feature data used to calculate real-time values
        pickle.dump(features_df,open(os.path.join(path_centroid,'features_df.p'),"wb"))

        #Save the features scaling values for each feature
        pickle.dump(minmax_df,open(os.path.join(path_centroid,'featurescaling.p'),"wb"))
        
        main_columns = ['close_'+ticker_output,'open_'+ticker_output,'high_'+ticker_output,'low_'+ticker_output]
        #Get minute data for the past day
        minute_data = price_minute_df[main_columns].iloc[-780:]
        minute_data = minute_data.rename(columns={'close_'+ticker_output: 'close','open_'+ticker_output: 'open','high_'+ticker_output: 'high','low_'+ticker_output: 'low'})
        minute_data.index = range(1,781)
        #Save minute level data for the ticker
        pickle.dump(minute_data,open(os.path.join(path_centroid,'minute_data.p'),"wb"))

        #Get daily data indexed by days in the past (where the last data)
        day_data = price_daily_df[main_columns].iloc[::-1]
        day_data = day_data.rename(columns={'close_'+ticker_output: 'close','open_'+ticker_output: 'open','high_'+ticker_output: 'high','low_'+ticker_output: 'low'})
        day_data.index = range(1,len(day_data)+1)
        #Save day level data for the ticker indexed by days in the past
        pickle.dump(day_data,open(os.path.join(path_centroid,'day_data.p'),"wb"))

        #Calculate the average daily volume for the previous month (~30 days)
        average_volume = sum(price_daily_df['volume_'+ticker_output].iloc[-30:])/30
        #Save the monthly average volume
        pickle.dump(average_volume,open(os.path.join(path_centroid,'monthly_ave_volume.p'),"wb"))

        centroid_date = price_daily_df['Date'].iloc[-1]
        #Save the dates that centroid data and feature/price data was last updated (they are the same because we are updating both) 
        pickle.dump((centroid_date,centroid_date),open(os.path.join(path_centroid,'centroiddate_featuredate.p'),"wb"))

        ticker_num = ticker_num+1


if __name__=="__main__":

    #Pull the saved data from stocks for a certain time period. CD 12/2020 Support a specific start_date and end_date
    start_date = datetime.date(2020,8,1)
    end_date = datetime.date(2020,12,31)

    #Amount of time in the future that we are trying to guess the price. CD 12/2020 Suppport minutes and days in future
    predict_period = 2  #number of minutes or days in future
    predict_periodType = 'day' #only 'minute' or 'day' supported  


    stock_machine_learning(start_date,end_date,predict_period,predict_periodType)

