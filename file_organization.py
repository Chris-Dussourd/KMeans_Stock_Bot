"""
CD 12/2020 - This contains the file and folder structure to save and to access stock price data as a python dictionary
           - Include folder structure for centroid labels
    
    folder_directory - directory name that currently holds price history files
    folder_strucutre - dictionary where the keys are ticker symbols and the values is the folder path for the ticker's price history
        - combine folder_directory and folder_sturcture[ticker] to get the path that contains the price history files

    file_structure - list of variables (prepended by v) and strings (prepended by s) that labels info contained in filenames
        The actual filenames are separated by the '_' delimiter and end with a pickle file type '.p'
        Ex: ['sPrices','vStartDate','vEndDate','vFrequency]  --> Prices_20200801_20200831_Minute.p
        possible variables - StartDate, EndDate, Frequency, and Ticker
    
    folder_directory_old - start of directory that previously held price history files
    folder_structure_old - dictionary where the keys are ticker names and the values is the folder path for the ticker's price history
        - combine folder_directory_old and folder_structure_old[ticker] to get the previous path containing the price history files
    file_structure_old - list of variables (prepended by v) and strings (prepended by s) saved in previous filenames
        Ex: ['vStartDate','vEndDate','vFrequency','vTicker','sPriceHistory']

    update_directory - updates the path that the price history files are stored in
        Before running:
        - populate folder_directory_old and folder_structure_old with the current file locations
        - populate folder_directory and folder_structure with the new directory and folder location to move files to
        Creates the a new directory/folder if folder_directory or folder_structure[ticker] do not exist
    
    update_filenames - updates the filename to a new file_structure
        Before running:
        - populate file_structure_old with the template that the current files are stored in
        - populate file_structure with the new template you wish to use 
            Note: vStartDate, vEndDate, and vFrequency must be in the old filename if we want those variables in the new filename

    create_filename(structure,file_info_dict) - uses the file structure to create a filename
        - variables (vStartDate,vEndDate,vTicker,vFrequency) need to be stored as keys in the file_info_dict

    delete_old_files(startDate,endDate,frequencyType) - deletes old files that are no longer useful (or files that overlap other files in dates)
        - it will delete files of startDate, endDate, frequencyType for all tickers in folder structure if the file exists
"""

import os
import shutil

#Add folder structure for centroid labels
folder_directory_centroid = ''

#Directory 
folder_directory = ''

#The structure of the file, lists of strings (prefixed with an s) and variables (prefixed with a v). Each item in list is separated by an underscore.
#Ex: ['vStartDate','vEndDate','vFrequency','sPriceHistory'] - 20200801_20200831_Minute_PriceHistory
#The code supports the following four variables - StartDate, EndDate, Frequency, and Ticker
#Recommend using: StartDate, EndDate, and Frequency in the filename. Use ticker in filename if storing more than one ticker in a folder.
file_structure = ['vStartDate','vEndDate','vFrequency']

#Dictionary that stores the folder structure (path) each ticker is stored under
folder_structure = {
    'HD': 'HD',
    'LOW': 'LOW',
    'MMM': 'MMM',
    'PHG': 'PHG',
    'NIO': 'NIO',
    'TM': 'TM',
    'TSLA': 'TSLA',
    'WKHS': 'WKHS',
    'AMD': 'AMD',
    'IBM': 'IBM',
    'INTC': 'INTC',
    'NVDA': 'NVDA',
    'ECL': 'ECL',
    'PHO': 'PHO',
    'RSG': 'RSG',
    'VEGN': 'VEGN',
    'WM': 'WM',
    'AAPL': 'AAPL',
    'HPQ': 'HPQ',
    'MSFT': 'MSFT',
    'MSI': 'MSI',
    'JKS': 'JKS',
    'NEE': 'NEE',
    'REGI': 'REGI',
    'SEDG': 'SEDG',
    'SPWR': 'SPWR',
    'GOOG': 'GOOG',
    'HXL': 'HXL',
    'MTDR': 'MTDR',
    'NAIL': 'NAIL',
    'CMG': 'CMG',
    'COST': 'COST',
    'KR': 'KR',
    'UNFI': 'UNFI',
    'CVS': 'CVS',
    'JNJ': 'JNJ',
    'MRK': 'MRK',
    'SYK': 'SYK',
    'DJX': 'DJX',
    'NDX': 'NDX',
    'SPX': 'SPX',
    'AMZN': 'AMZN',
    'FB': 'FB',
    'QCOM': 'QCOM',
    'DFS': 'DFS',
    'T': 'T',
    'V': 'V',
    'VZ': 'VZ'}

#old directory, populate this if you would like to update the directory your files are stored
folder_directory_old = ''

#file structure old
file_structure_old = ['vFrequency','vStartDate','vEndDate','sTest']

#old folder structure, populate this if you would like to update the file structure
folder_structure_old = {
    'HD': 'Price_History_Monthly_Building',
    'LOW': 'Price_History_Monthly_Building',
    'MMM': 'Price_History_Monthly_Building',
    'PHG': 'Price_History_Monthly_Building',
    'NIO': 'Price_History_Monthly_Car',
    'TM': 'Price_History_Monthly_Car',
    'TSLA': 'Price_History_Monthly_Car',
    'WKHS': 'Price_History_Monthly_Car',
    'AMD': 'Price_History_Monthly_Chips',
    'IBM': 'Price_History_Monthly_Chips',
    'INTC': 'Price_History_Monthly_Chips',
    'NVDA': 'Price_History_Monthly_Chips',
    'ECL': 'Price_History_Monthly_Climate',
    'PHO': 'Price_History_Monthly_Climate',
    'RSG': 'Price_History_Monthly_Climate',
    'VEGN': 'Price_History_Monthly_Climate',
    'WM': 'Price_History_Monthly_Climate',
    'AAPL': 'Price_History_Monthly_Computer',
    'HPQ': 'Price_History_Monthly_Computer',
    'MSFT': 'Price_History_Monthly_Computer',
    'MSI': 'Price_History_Monthly_Computer',
    'JKS': 'Price_History_Monthly_Energy',
    'NEE': 'Price_History_Monthly_Energy',
    'REGI': 'Price_History_Monthly_Energy',
    'SEDG': 'Price_History_Monthly_Energy',
    'SPWR': 'Price_History_Monthly_Energy',
    'GOOG': 'Price_History_Monthly_Favs',
    'HXL': 'Price_History_Monthly_Favs',
    'MTDR': 'Price_History_Monthly_Favs',
    'NAIL': 'Price_History_Monthly_Favs',
    'CMG': 'Price_History_Monthly_Food',
    'COST': 'Price_History_Monthly_Food',
    'KR': 'Price_History_Monthly_Food',
    'UNFI': 'Price_History_Monthly_Food',
    'CVS': 'Price_History_Monthly_Health',
    'JNJ': 'Price_History_Monthly_Health',
    'MRK': 'Price_History_Monthly_Health',
    'SYK': 'Price_History_Monthly_Health',
    'DJX': 'Price_History_Monthly_Indexes',
    'NDX': 'Price_History_Monthly_Indexes',
    'SPX': 'Price_History_Monthly_Indexes',
    'AMTD': 'Price_History_Monthly_Internet',
    'AMZN': 'Price_History_Monthly_Internet',
    'FB': 'Price_History_Monthly_Internet',
    'QCOM': 'Price_History_Monthly_Internet',
    'DFS': 'Price_History_Monthly_Services',
    'T': 'Price_History_Monthly_Services',
    'V': 'Price_History_Monthly_Services',
    'VZ': 'Price_History_Monthly_Services'}


def update_directory():
    """
    Move files from the old file directory into a new file directory

    folder_directory_old - start of directory that currently holds price history files
    folder_structure_old - dictionary where the keys are ticker names and the values is the folder path for the ticker's price history
        - combine folder_directory_old and folder_structure_old[ticker] to get the path containing the files
    
    folder_directory - new directory name (creates folders if they don't exist)
    folder_strucutre - dictionary where the keys are ticker symbols and the values is the new folder path for the ticker's price history
        - combine folder_directory and folder_sturcture[ticker] to get the new path to store the files
    """

    #If the folder name and directory changed, move files to new folder
    if (folder_directory != folder_directory_old) or (folder_structure != folder_structure_old):

        #Move each ticker files to a new directory and/or folder
        for ticker in folder_structure_old:

            path_old = os.path.join(folder_directory_old,folder_structure_old[ticker])

            #Get all of the filenames in this path
            filenames = [f for f in os.listdir(path_old) if os.path.isfile(os.path.join(path_old,f))]

            #If a stock ticker is in the filename, only keep filenames of the current ticker
            if 'vTicker' in file_structure:

                index_ticker = file_structure.index('vTicker')
                #Create list of filenames where the ticker is in the filename
                temp_filenames = [f for f in filenames if f.split('_')[index_ticker]==ticker]
                filenames = temp_filenames

            path = os.path.join(folder_directory,folder_structure[ticker])
            #Create new folder if it doesn't exist
            if not os.path.isdir(path):
                os.makedirs(path)

            #Move files to new folder structure
            for f in filenames:
                shutil.move(os.path.join(path_old,f),path)


def update_filename():
    """
    Rename the filename using file_structure

    file_structure - list of variables (prepended by v) and strings (prepended by s)
        Ex: ['sPrices','vStartDate','vEndDate','vFrequency]
        possible variables - StartDate, EndDate, Frequency, and Ticker

    file_structure_old - list of variables (prepended by v) and strings (prepended by s) saved in current filenames
        Ex: ['vStartDate','vEndDate','vFrequency','vTicker','sPriceHistory']
    
    Update the old filenames by getting the variables in old file structure and re-ordering to form a new file structure
    The actual filenames are separated by the '_' delimiter and end with a pickle file type '.p'
        Ex: 'Prices_20200801_20200831_Minute.p' using structure ['sPrices','vStartDate','vEndDate','vFrequency']
    """

    #If the file structure has changed rename the files
    if file_structure != file_structure_old:

        old_vars = []
        #Check old file structure items are prepended with a v or s, and save the variables
        for file_part in file_structure_old:
            if file_part[0] != 's' and file_part[0] != 'v':
                raise ValueError('The old file structure is not formatted correctly. All items must be prepended with a v or s.')
            if file_part[0] == 'v':
                old_vars.append(file_part)

        new_vars = [] 
        #Check file structure items are prepended with a v or s, and save the variables
        for file_part in file_structure:
            if file_part[0] != 's' and file_part[0] != 'v':
                raise ValueError('The new file structure is not formatted correctly. All items must be prepended with a v or s.')
            if file_part[0] == 'v':
                new_vars.append(file_part)
                #Raise error if old file structure doesn't have enough information to create new structure
                """
                if (file_part not in old_vars) and (file_part != 'vTicker'):
                    error_message = 'There is not enough information to create a new file structure. Cannot populate variable "{}".'.format(file_part)
                    raise ValueError(error_message)
                """
        
        #Get the indices for variable names (vTicker, vStartDate, vEndDate, vFrequency)
        
        #Loop over all tickers in folder_structure
        for ticker in folder_structure:

            path = os.path.join(folder_directory,folder_structure[ticker])
            #Update every file name using the new structure
            for f in os.listdir(path):
                if os.path.isfile(os.path.join(path,f)):
                    
                    #Remove the file extension '.p' and divide the filename into pieces using the delimiter '_'
                    delimited_file = f[:-2].split('_')

                    #Only update filenames that are for this ticker
                    if ('vTicker' not in file_structure_old) or ('vTicker' in file_structure_old and delimited_file[file_structure_old.index('vTicker')]==ticker):
                        
                        filename_new = []
                        #Loop through all items in new file_structure to populate file name
                        for file_part in file_structure:
                            if file_part[0] == 's':
                                filename_new.append(file_part[1:]) #Add string to filename

                            elif file_part == 'vStartDate':
                                if 'vStartDate' in file_structure_old:
                                    filename_new.append(delimited_file[file_structure_old.index('vStartDate')]) #Add start date from old filename                                   
                            
                            elif file_part == 'vEndDate':
                                if 'vEndDate' in file_structure_old:
                                    filename_new.append(delimited_file[file_structure_old.index('vEndDate')]) #Add end date from old filename

                            elif file_part == 'vTicker':
                                filename_new.append(ticker) #Add ticker to new filename
                            
                            elif file_part == 'vFrequency':
                                if 'vFrequency' in file_structure_old:
                                    filename_new.append(delimited_file[file_structure_old.index('vFrequency')])
                            
                        filename_new_str = '_'.join(filename_new) + '.p' #Assume always storing as a pickle file
                        #Rename the actual file
                        os.rename(os.path.join(path,f),os.path.join(path,filename_new_str))


def create_filename(structure,file_info_dict):
    """
    Returns the filename that is consitent with the structure passed in.

    structure - list of variables (prepended by v) and strings (prepended by s)
        Ex: ['sPrices','vStartDate','vEndDate','vFrequency]
        possible variables - StartDate, EndDate, Frequency, and Ticker

    file_info_dict - dictionary mapping variable names to values
        Example:
        'vStartDate': 20200801
        'vEndDate': 20200831
        'vTicker': 'GOOG'
        'vFrequency': 'Minute'
    """
    filename_list=[]
    for file_part in structure:
        if file_part[0] == 's':
            filename_list.append(file_part[1:]) #Add string to filename

        elif file_part == 'vStartDate':
            filename_list.append(file_info_dict['vStartDate']) #Add start date from old filename                                   
        
        elif file_part == 'vEndDate':
            filename_list.append(file_info_dict['vEndDate']) #Add end date from old filename

        elif file_part == 'vTicker':
            filename_list.append(file_info_dict['vTicker']) #Add ticker to new filename
        
        elif file_part == 'vFrequency':
            filename_list.append(file_info_dict['vFrequency'])

        else:
            raise ValueError('The file structure is not formatted correctly or you have incorrect variables.')

    return '_'.join(filename_list) + '.p' #Assume always storing as a pickle file



def delete_old_files(startDate,endDate,frequencyType):
    """
    Deletes old files that are no longer useful (or files that overlap other files in dates).
    Deletes files of startDate, endDate, frequencyType for all tickers in folder structure if the file exists.

    Passed in variables:
    startDate - the date that the price information is stored from
    endDate - the date that the price information is stored to
    frequencyType - 'minute' or 'daily' for how often data points are recorded

    Global variables in file needed for this functions:
    folder_directory - new directory name (creates folders if they don't exist)
    folder_strucutre - dictionary where the keys are ticker symbols and the values is the new folder path for the ticker's price history
        - combine folder_directory and folder_sturcture[ticker] to get the new path to store the files
    file_structure - the structure of the file name based on variables (prepended by v) and strings (prepended by s) 
        Ex: ['sPrices','vStartDate','vEndDate','vFrequency]
        possible variables - StartDate, EndDate, Frequency, and Ticker


    """

    #Move each ticker files to a new directory and/or folder
    for ticker in folder_structure:

        path = os.path.join(folder_directory,folder_structure[ticker])

        #Get all of the filenames in this path
        filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]

        #If a stock ticker is in the filename, only keep filenames of the current ticker
        if 'vTicker' in file_structure:

            index_ticker = file_structure.index('vTicker')
            #Create list of filenames where the ticker is in the filename
            temp_filenames = [f for f in filenames if f.split('_')[index_ticker]==ticker]
            filenames = temp_filenames

        if 'vStartDate' in file_structure:

            index_startDate = file_structure.index('vStartDate')
            #Create list of filenames where the startDate is equal to the one passed in
            temp_filenames = [f for f in filenames if f.split('_')[index_startDate]==startDate]
            filenames = temp_filenames

        if 'vEndDate' in file_structure:

            index_endDate = file_structure.index('vEndDate')
            #Create list of filenames where the endDate is equal to the one passed in
            temp_filenames = [f for f in filenames if f.split('_')[index_endDate]==endDate]
            filenames = temp_filenames

        if 'vFrequency' in file_structure:

            index_frequency = file_structure.index('vFrequency')
            #Create list of filenames where the frequency is equal to the one passed in
            temp_filenames = [f for f in filenames if f.split('_')[index_frequency]==frequencyType]

        #Remove/delete files to new folder structure
        for f in filenames:
            os.remove(os.path.join(path,f))