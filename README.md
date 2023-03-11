# KMeans_Stock_Bot

> I abandoned this project two years ago. I saw myself becoming more obsessed with money and decided to take a step back from creating these algorithms. I also got a coding job. This project is incomplete.

This bot was intended to use the centroids from the K Means clustering algorithm to make trading decisions. 

It divides the money available and places buy orders on stocks with 'Strong Buy' labels.
It sells the stock when price is above purchase price and the stock data changes to a neutral or sell cluster centroid. 
Collects price and option information on the minute.

Description of threading - This bot creates 2 threads.
 - Thread 1: Places buy and sell orders on TD Ameritrade Site   
 - Thread 2: Get the latest stock and option prices from TD Ameritrade
 - Thread 3: Looks for user's input and stops for debugging
 
 It uses features based on common stock calculations (MACD, RSI, Stochastic Oscillator, etc).
 
 Lastly, it creates and keeps clean a file directory that stores pricing information. Old data is removed and new data is stored using pickle.
