#Import all necessary libraries
import numpy as np
import pandas as pd
import random
from urllib.request import urlopen

#------------------------------------------------------------------------------------------#

#Data pre-processing

#Handle missing value
def missingValue(data):
    for i in data.columns:
        #If the feature is not the target column AND
        #There is missing value in this column
        if i != "label" and data[i].isnull().values.any():
            unique = data[i].unique()
            #It is categorical if:
                #The value is a String OR
                #There are less than or equal to 20 unique value
                #THEN Replace categorical value's missing value as their most frequent value
            #ELSE Replace numerical value's missing value as their mean
            data = (data.fillna(data.mode().iloc[0]) if ((isinstance(unique[0], str)) or (len(unique) <= 20)) else data[i].fillna(data[i].mean(), inplace=True))
    return data

#Read in data
def load(url, name):
    #Read in data and store it
    #Replace ? with nan
    df = pd.read_csv(urlopen(url), na_values = "?")
    #Adding column names for the dataset
    #All column names should not contain space
    #Last row is the target and must be name label
    df.columns = name
    return df