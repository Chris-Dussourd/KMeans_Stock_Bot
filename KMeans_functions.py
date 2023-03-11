"""
K-means clustering. Takes in pandas DataFrames for input variables and returns cluster centroids.

CD 12/2020 - Create k_means_clustering and assign_data

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def k_means_clustering(data_df,clusters):
    """
    Performs K-means clustering on the data.

    data_df - DataFrame, each column is a new feature and each row is a new sample.
    clusters - number of clusters to divide the data into

    Returns cluster centroids
    """

    #Initialize random centroid (k X n centroid matrix)
    centroids = data_df.sample(n=clusters).reset_index(drop=True)

    #Initialize Euclidean distance between each sample and centroid (m samples X k centroids)
    dist_centroid = pd.DataFrame(0,index = range(data_df.shape[0]), columns = range(clusters))

    #Loop through K-Means clustering for multiple iterations
    iterations = 100
    for iter in range(1,iterations+1):

        #Find the euclidean distance between input samples and centroids (m samples x k centroids)
        for k in range(clusters):
            dist_centroid[k] = np.sqrt(np.square(data_df.sub(centroids.loc[k,:],axis=1).values).sum(axis=1))

        #Assign each data point to its nearest cluster (min distance)
        assign_centroid = dist_centroid.idxmin(axis=1)

        #Calculate new centroid values (mean of all data points currently assigned)
        for k in range(clusters):
            centroids.loc[k,:] = data_df[assign_centroid==k].mean(axis=0)

    return centroids


def assign_data(data_df,column_name,centroids,datetime_series):
    """
    Assign the data to the nearest centroid by finding the minimum Euclidean distance.
    Create a dictionary where each key is a cluster and the values are the data points in column_name belonging to the cluster 
    Create a dictionary where each key is a cluster and the values are datetimes belonging to the cluster

    data_df - DataFrame of data to assign to the centroids (each column is a feature, each row is a sample)
    column_name - the data points to store in the cluster dictionary
    centroids - DataFrame of centroids, each row is a cluster, and each column is a feature
    datetime_series - series of datetimes that align with the data_df index


    Returns 
        cluster_data - dictionary, each key is a cluster number (from 0 to k), values are the data points in column name belonging to the cluster
        cluster_datetimes - dictionary, each key is a cluster, values are datetimes belonging to the cluster
    """

    #Initialize Euclidean distance between each sample and centroid (m samples X k centroids)
    dist_centroid = pd.DataFrame(index = range(data_df.shape[0]), columns = range(centroids.shape[0]))

    #Find the euclidean distance between input samples and centroids (m samples x k centroids)
    for k in range(centroids.shape[0]):
        dist_centroid[k] = np.sqrt(np.square(data_df.sub(centroids.loc[k,:],axis=1).values).sum(axis=1))

    #Assign each data point to its nearest cluster (min distance)
    assign_centroid = dist_centroid.idxmin(axis=1)

    #Store the data points from column name into the cluster dictionary
    cluster_data = {}
    cluster_datetimes = {}
    for k in range(centroids.shape[0]):
        cluster_data[k] = data_df[assign_centroid==k][column_name]
        cluster_datetimes[k] = datetime_series[assign_centroid==k]

    return (cluster_data,cluster_datetimes)


def label_centroids(threshold,cluster_train_cat,cluster_test1_cat,cluster_test2_cat={}):
    """
    Label each centroid as buy, weak buy, neutral, weak sell, or sell.
    Assumes the lower category numbers are for losses and the higher category numbers are for gains

    threshold - list of threshold values that separate each of the categories
    cluster_train_cat - dictionary, each key is a cluster, each value is a list of categories belonging to cluster in the training set
    cluster_test1_cat - dictionary, each key is a cluster, each value is a list of categories belonging to cluster in the first test set
    cluster_test2_cat - dictionary, each key is a cluster, each value is a list of categories belonging to cluster in the second test set (not required)

    Returns:
        centroid_labels - pandas series, each index is a cluster number, each value is a label for the cluster (strong buy, buy, weak buy, neutral, weak sell, sell, strong sell)
    """
    centroid_labels = pd.Series(index=range(len(cluster_train_cat.keys())))

    #Number of categories (categories are divided up by threshold values)
    num_cat = len(threshold)+1

    cat_split_buysell = 0
    #Category that splits the buy and sell signal (in between positive and negative thresholds)
    for thresh_index in range(len(threshold)):
        if threshold[thresh_index] > 0 and cat_split_buysell == 0: #First threshold greater than zero
            cat_split_buysell = thresh_index

    #Loop over each cluster
    for k in cluster_train_cat:
        train_counts = cluster_train_cat[k].value_counts()
        test1_counts = cluster_test1_cat[k].value_counts()
        test2_counts = pd.Series({}) if not cluster_test2_cat else cluster_test2_cat[k].value_counts()

        #Change index to category numbers (it was scaled before performing K-means clustering)
        train_counts.index = train_counts.index*(num_cat-1)
        if not test1_counts.empty:
            test1_counts.index = test1_counts.index*(num_cat-1)
        if not test2_counts.empty:
            test2_counts.index = test2_counts.index*(num_cat-1)

        #Add all labels not indexed with a value of zero (zero categories in the cluster)
        for cat in np.arange(float(num_cat)):
            if cat not in train_counts:
                train_counts[cat]=0
            if not test1_counts.empty and cat not in test1_counts:
                test1_counts[cat]=0
            if not test2_counts.empty and cat not in test2_counts:
                test2_counts[cat]=0

        #If test1 and test2 counts have data, set centroid label (else set label to neutral)
        if not test1_counts.empty and (not test2_counts.empty or not cluster_test2_cat):

            #Buy clusters have there most frequent category above the split buy/sell category for train, test1, and test2 samples
            if train_counts.index[0]>cat_split_buysell and test1_counts.index[0]>cat_split_buysell and (test2_counts.empty or test2_counts.index[0]>cat_split_buysell):
                
                #Strong buy clusters have the top two categories have 90% of the samples
                if (train_counts[num_cat-1]+train_counts[num_cat-2])/sum(train_counts)>0.9 and (test1_counts[num_cat-1]+test1_counts[num_cat-2])/sum(test1_counts)>0.9 and \
                    (test2_counts.empty or (test2_counts[num_cat-1]+test2_counts[num_cat-2])/sum(test2_counts)>0.9):
                    centroid_labels[k] = 'Strong Buy'

                #Buy clusters have top two categories 60% of the samples
                elif (train_counts[num_cat-1]+train_counts[num_cat-2])/sum(train_counts)>0.6 and (test1_counts[num_cat-1]+test1_counts[num_cat-2])/sum(test1_counts)>0.6 and \
                    (test2_counts.empty or (test2_counts[num_cat-1]+test2_counts[num_cat-2])/sum(test2_counts)>0.6):
                    centroid_labels[k] = 'Buy'
                
                #The rest are weak buy clusters
                else:
                    centroid_labels[k] = 'Weak Buy'


            #Sell clusters have their most frequent category below the split buy/sell category for train, test1, and test2 samples
            elif train_counts.index[0]<cat_split_buysell and train_counts.index[0]<cat_split_buysell and (test2_counts.empty or test2_counts.index[0]>cat_split_buysell):

                #Strong sell clusters have the bottom two categories have 90% of the samples
                if (train_counts[0]+train_counts[1])/sum(train_counts)>0.9 and (test1_counts[0]+test1_counts[1])/sum(test1_counts)>0.9 and \
                    (test2_counts.empty or (test2_counts[0]+test2_counts[1])/sum(test2_counts)>0.9):
                    centroid_labels[k] = 'Strong Sell'

                #Strong sell clusters have the bottom two categories have 60% of the samples
                elif (train_counts[0]+train_counts[1])/sum(train_counts)>0.6 and (test1_counts[0]+test1_counts[1])/sum(test1_counts)>0.6 and \
                    (test2_counts.empty or (test2_counts[0]+test2_counts[1])/sum(test2_counts)>0.6):
                    centroid_labels[k] = 'Sell'

                #The rest are weak sell clusters
                else:
                    centroid_labels[k] = 'Weak Sell'

            #Clusters are neutral if most frequent cluster is the  split buy/sell category or if train, test1, and test2 samples have different signals
            else:
                centroid_labels[k] = 'Neutral'
        
        #Clusters are neutral if missing test1 or test2 data
        else:
            centroid_labels[k] = 'Neutral'

    return centroid_labels



def find_best_labels(threshold,cluster_train_cat,cluster_test1_cat,cluster_test2_cat={}):
    """
    Label each centroid as buy, weak buy, neutral, weak sell, or sell.
    Assumes the lower category numbers are for losses and the higher category numbers are for gains

    threshold - list of threshold values that separate each of the categories
    cluster_train_cat - dictionary, each key is a cluster, each value is a list of categories belonging to cluster in the training set
    cluster_test1_cat - dictionary, each key is a cluster, each value is a list of categories belonging to cluster in the first test set
    cluster_test2_cat - dictionary, each key is a cluster, each value is a list of categories belonging to cluster in the second test set (not required)

    Returns:
        centroid_labels - pandas series, each index is a cluster number, each value is a label for the cluster (strong buy, buy, weak buy, neutral, weak sell, sell, strong sell)
    """
    centroid_labels = pd.Series(index=range(len(cluster_train_cat.keys())))

    #Number of categories (categories are divided up by threshold values)
    num_cat = len(threshold)+1

    cat_split_buysell = 0
    #Category that splits the buy and sell signal (in between positive and negative thresholds)
    for thresh_index in range(len(threshold)):
        if threshold[thresh_index] > 0 and cat_split_buysell == 0: #First threshold greater than zero
            cat_split_buysell = thresh_index

    #Loop over each cluster
    for k in cluster_train_cat:
        train_counts = cluster_train_cat[k].value_counts()
        test1_counts = cluster_test1_cat[k].value_counts()
        test2_counts = pd.Series({}) if not cluster_test2_cat else cluster_test2_cat[k].value_counts()

        #Change index to category numbers (it was scaled before performing K-means clustering)
        train_counts.index = train_counts.index*(num_cat-1)
        if not test1_counts.empty:
            test1_counts.index = test1_counts.index*(num_cat-1)
        if not test2_counts.empty:
            test2_counts.index = test2_counts.index*(num_cat-1)

        #Add all labels not indexed with a value of zero (zero categories in the cluster)
        for cat in np.arange(float(num_cat)):
            if cat not in train_counts:
                train_counts[cat]=0
            if not test1_counts.empty and cat not in test1_counts:
                test1_counts[cat]=0
            if not test2_counts.empty and cat not in test2_counts:
                test2_counts[cat]=0

        #If test1 and test2 counts have data, set centroid label (else set label to neutral)
        if not test1_counts.empty and (not test2_counts.empty or not cluster_test2_cat):

            #Buy clusters have there most frequent category above the split buy/sell category for train, test1, and test2 samples
            if train_counts.index[0]>cat_split_buysell and test1_counts.index[0]>cat_split_buysell and (test2_counts.empty or test2_counts.index[0]>cat_split_buysell):
                
                #Strong buy clusters have the top two categories have 90% of the samples
                if (train_counts[num_cat-1]+train_counts[num_cat-2])/sum(train_counts)>0.9 and (test1_counts[num_cat-1]+test1_counts[num_cat-2])/sum(test1_counts)>0.9 and \
                    (test2_counts.empty or (test2_counts[num_cat-1]+test2_counts[num_cat-2])/sum(test2_counts)>0.9):
                    centroid_labels[k] = 'Strong Buy'

                #Buy clusters have top two categories 60% of the samples
                elif (train_counts[num_cat-1]+train_counts[num_cat-2])/sum(train_counts)>0.6 and (test1_counts[num_cat-1]+test1_counts[num_cat-2])/sum(test1_counts)>0.6 and \
                    (test2_counts.empty or (test2_counts[num_cat-1]+test2_counts[num_cat-2])/sum(test2_counts)>0.6):
                    centroid_labels[k] = 'Buy'
                
                #The rest are weak buy clusters
                else:
                    centroid_labels[k] = 'Weak Buy'


            #Sell clusters have their most frequent category below the split buy/sell category for train, test1, and test2 samples
            elif train_counts.index[0]<cat_split_buysell and train_counts.index[0]<cat_split_buysell and (test2_counts.empty or test2_counts.index[0]>cat_split_buysell):

                #Strong sell clusters have the bottom two categories have 90% of the samples
                if (train_counts[0]+train_counts[1])/sum(train_counts)>0.9 and (test1_counts[0]+test1_counts[1])/sum(test1_counts)>0.9 and \
                    (test2_counts.empty or (test2_counts[0]+test2_counts[1])/sum(test2_counts)>0.9):
                    centroid_labels[k] = 'Strong Sell'

                #Strong sell clusters have the bottom two categories have 60% of the samples
                elif (train_counts[0]+train_counts[1])/sum(train_counts)>0.6 and (test1_counts[0]+test1_counts[1])/sum(test1_counts)>0.6 and \
                    (test2_counts.empty or (test2_counts[0]+test2_counts[1])/sum(test2_counts)>0.6):
                    centroid_labels[k] = 'Sell'

                #The rest are weak sell clusters
                else:
                    centroid_labels[k] = 'Weak Sell'

            #Clusters are neutral if most frequent cluster is the  split buy/sell category or if train, test1, and test2 samples have different signals
            else:
                centroid_labels[k] = 'Neutral'
        
        #Clusters are neutral if missing test1 or test2 data
        else:
            centroid_labels[k] = 'Neutral'

    return centroid_labels