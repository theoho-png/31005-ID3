#Import all necessary libraries
import numpy as np
import pandas as pd
import random
from urllib.request import urlopen

#------------------------------------------------------------------------------------------#

#Data pre-processing

#Read in data
def load(url, name):
    #Read in data and store it
    df = pd.read_csv(urlopen(url))
    #Adding column names for the dataset
    #All column names should not contain space
    #Last row is the target and must be name label
    df.columns = name
    return df

#Spliting the data to training and testing data
def splitDF(data, size):
    
    #random.seed(0)
    #len(df)/100)*size: automatically calculate the number of rows with size%
    #index.tolist() contain all the index of the dataframe
    #random.sample randomly select rows into test_indices
    indices = random.sample(data.index.tolist(), int((len(data)/100)*size))

    #Return test data 
    #And train data by removing all the test data from the dataframe
    return data.loc[indices], data.drop(indices)

#------------------------------------------------------------------------------------------#

#Determind Split

#Calculate Entropy
def entropy(data):
    
    #Return the row count of distinct classes in the last column 
    _, count = np.unique(data[:, -1], return_counts=True)

    #Calculate the Entropy using the Shannon's formula
    #count / count.sum(): the probabilities of the class
    return sum(count / count.sum() * -np.log2(count / count.sum()))

#Calculate information gain
def infoGain(data,dataBelow, dataAbove):
    
    #Calculate the entropy of the subset:
    #proportion of x * entropy of x
    #len(dataBelow) + len(dataAbove): return total data points
    subset = ((len(dataBelow) / (len(dataBelow) + len(dataAbove))) 
            * entropy(dataBelow)
            + (len(dataAbove) / (len(dataBelow) + len(dataAbove)) 
            * entropy(dataAbove)))

    #Entropy of the set - entropy of a subset
    return entropy(data) - subset

#Spliting data based on the value of each column
def split(data, column, value):
    
    columnValues = data[:, column]

    #Boolean indexing
    if FEATURE_TYPES[column] == "numerical": #Feature is numerical
        dataBelow = data[columnValues <= value] #If the value is smaller than split value, it is dataBelow
        dataAbove = data[columnValues >  value] #If the value is bigger than split value, it is dataAbove
    #Feature is categorical  
    else:
        dataBelow = data[columnValues == value] #If the value is the split value, it is dataBelow
        dataAbove = data[columnValues != value] #If the value is not the split value, it is dataAbove
    
    return dataBelow, dataAbove

#Determine potential splits
def potentialSplit(data):
    
    #Create a dictionary call potentialSplit
    potentialSplit = {}

    #data.shape return the number of rows and columns in the data
    _, columnCount = data.shape

    #Exclude the last column which is the target class
    #Append distinct value as potential split into the dictionary
    for i in range(columnCount - 1):
        #Append distinct value as potential split into the dictionary
        potentialSplit[i] = np.unique(data[:, i])

    return potentialSplit

#Determind the best column and value to split
def bestSplit(data, potentialSplits):
    
    bestScore = 0
    #For each index in potentialSplits
    for i in potentialSplits:
        #For each value in that index
        for j in potentialSplits[i]:

            #Spilt the data in dataBelow and dataAbove based on the value
            dataBelow, dataAbove = split(data, i, j)
            score = infoGain(data, dataBelow, dataAbove) #Determind split based on information gain

            if score >= bestScore: #Storing the split value and column with the highest information gain
                bestScore = score
                bestColumn = i
                bestValue = j
    
    return bestColumn, bestValue

#------------------------------------------------------------------------------------------#

#Decision Tree Algorithm

# Determind if the feature is categorical
def featureType(data):
    
    types = []
    #Loop through all the column in data
    for i in data.columns:
        #If the feature is not the target column
            #It is categorical if:
             #Tthe value is a String OR
             #There are less than or equal to 20 unique value
        if i != "label":
            unique = data[i].unique()
            types.append("categorical") if ((isinstance(unique[0], str)) or (len(unique) <= 20)) else types.append("numerical")
    return types

#Create Leaf
def createLeaf(data):

    #Return the distinct classes of the last column and their row count 
    uniqueClass, count = np.unique(data[:, -1], return_counts=True)

    #.argmax() returns the position of the maximum values along an axis (last column)
    return uniqueClass[count.argmax()]

#Main algorithm
def decisionTree(df, counter=0, minSample=2, depth=3):
    
    # data preparations
    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPES #Declare global variable
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = featureType(df)
        data = df.values
    else:
        data = df           
    
    #Base case because this function is a recursive function
    #Check if the target column only have one unique class AND 
    #Check the size of the dataset to make sure its smaller than min
    if (len(np.unique(data[:, -1])) == 1) or (len(data) < minSample) or (counter == depth): return createLeaf(data)
    else:
        counter += 1 #+1 counter because this is a recursive function
        column, value = bestSplit(data, potentialSplit(data)) #Finding split value and column
        dataBelow, dataAbove = split(data, column, value) #Return the dataBelow & dataAbove after the split
        
        if len(dataBelow) == 0 or len(dataAbove) == 0: return createLeaf(data) #Check for empty data
        
        #Making sub tree:
        #Make a dictionary & determine questions
        question = ("{} <= {}".format(COLUMN_HEADERS[column], value) if FEATURE_TYPES[column] == "numerical" else "{} = {}".format(COLUMN_HEADERS[column], value)) 
        subTree = {question: []}
        
        #Get answers (recurtion)
        yes = decisionTree(dataBelow, counter, minSample, depth)
        no = decisionTree(dataAbove, counter, minSample, depth)
        if yes == no: subTree = yes #If the answers are the same, then there is no point in asking the qestion
        else:
            subTree[question].append(yes)
            subTree[question].append(no)
        return subTree

#------------------------------------------------------------------------------------------#

#Make predictions

def predicting(predict, tree):
    
    #Check if it is the root
    if not isinstance(tree, dict): return tree
    
    question = list(tree.keys())[0] #Return the string within the dictionary
    #Split the question string into 3 elements (stored inside a list)
    feature, operator, value = question.split(" ") #''petalWidth <= 1' >>> ['petalWidth', '<=', '1']

    #Picking the yes / no after asking question
    answer = ((tree[question][0] if predict[feature] <= float(value) else tree[question][1]) if operator == "<=" else (tree[question][0] if str(predict[feature]) == value else tree[question][1]))

    #Base case because this function is a recursive function
    #Continue to ask question if it is not a classified data
    return (answer if not isinstance(answer, dict) else predicting(predict, answer))
    
# 3.2 All examples of a dataframe
def predictions(data, tree):
    
    #axis = 1: Make sure it apply to all the row
    #args=(tree,): Pass in tree as a tuple
    if len(data) != 0:
        predictions = data.apply(predicting, args=(tree,), axis=1)
    else:
        # "df.apply()"" with empty dataframe returns an empty dataframe,
        # but "predictions" should be a series instead
        predictions = pd.Series()
        
    return predictions

#------------------------------------------------------------------------------------------#

#Evaultaion

def accuracy(data, tree):

    #Boolean of if the classification is correct
    correct = predictions(data, tree) == data.label
    #Becuse true is 1 and false is 0
    #The mean of the class equals the accuracy
    return correct.mean()
