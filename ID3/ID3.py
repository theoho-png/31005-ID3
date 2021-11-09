import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import id3
import experiment as exp

#------------------------------------------------------------------------------------------#

#Select the dataset

#Iris dataset:
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
name = ["sepalLength", "sepalWiid3h", "petalLength", "petalWiid3h", "label"]

#Breast Cancer dataset:
#url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data'
#name = ["class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "breast", "breast-quad", "label"]

#------------------------------------------------------------------------------------------#

#An 80-20 split data

try:
    train, test = id3.splitDF(id3.load(url, name),0) #Load data, split it into test and train data.
    print("The base case of 80-20 split data achieve a " + str(np.round(id3.accuracy(test, id3.decisionTree(train)), 4)) + " accuracy") #Announce prediction accuracy
#except (IndexError, ValueError): print("Split percentage should be between 1 to 99.")
except Exception as e: print(e)

#------------------------------------------------------------------------------------------#

#Experiment Design and Evaluation

#Determind best % to split
def split():

    bestScore = 0 #Default value is 0
    scoreList = [] #Use for sns lineplotting

    #The % must be between 1 to 99
    #If the % is 0, there will be no training data
    #If the % is 100, there will be no testing data
    for i in range(1,99):

        #Load data, split it into test and train data through trial and error
        train, test = id3.splitDF(df, i)
        #Store its accuracy it can use as comparison later
        score = id3.accuracy(test, id3.decisionTree(train))

        #Store result, use for sns lineplotting
        scoreList.append(score)

        if bestScore < score: #Compare prediction accuracy
            bestScore = score
            split = i
    #For sns lineplot:
    global SPLIT
    SPLIT = pd.DataFrame({'accuracy' : scoreList, 'split' : range(1, 99)})

    return bestScore, split

#Handle missing value
def missingValue():
    df = exp.missingValue(exp.load(url, name)) #Process all the missing value and then load data
    train, test = id3.splitDF(df, 80) #Split it into test and train data.
    return id3.accuracy(test, id3.decisionTree(train)) #Prediction accuracy

#Determind best tree depth
def depth():

    bestScore = 0 #Default value is 0
    scoreList = [] #Use for sns lineplotting

    #The range is kept between 1 to 10 to reduce unnecessary testing
    # (minimise computing power, since its already time demanding) 
    for i in range(1,10):
        
        #Load data, split it into test and train data
        #80-20 is the default
        train, test = id3.splitDF(df, 80)
        #Store its accuracy it can use as comparison later
        score = id3.accuracy(test, id3.decisionTree(train, depth = i))

        #Store result, use for sns lineplotting
        scoreList.append(score)

        if bestScore < score: #Compare prediction accuracy
            bestScore = score
            depth = i
    
    #For sns lineplot:
    global DEPTH
    DEPTH = pd.DataFrame({'accuracy' : scoreList, 'depth' : range(1, 10)})
    
    return bestScore, depth

def bestSettings():
    df = exp.missingValue(exp.load(url, name)) #Process all the missing value and then load data
    bestScore = 0  #Default value is 0

    #The % must be between 1 to 99
    #If the % is 0, there will be no training data
    #If the % is 100, there will be no testing data
    for i in range(1,99):

        #The range is kept between 1 to 10 to reduce unnecessary testing
        # (minimise computing power, since its already time demanding) 
        for k in range(1,10):

            #Load data, split it into test and train data through trial and error
            train, test = id3.splitDF(df, i) 
            #Store its accuracy it can use as comparison later
            score = id3.accuracy(test, id3.decisionTree(train, depth = k))

            if bestScore < score:#Compare prediction accuracy
                bestScore = score
                split = i
                depth = k
                
    return bestScore, split, depth

#------------------------------------------------------------------------------------------#

#Evaluation Results

#Experiment: Data preprocessing

#Find best % for train and test data
try:

    bestScore, split = split() #Store all return value
    print("With a " + str(split) + "% split, it achieve a " + str(np.round(bestScore, 4)) + " accuracy") #Announce prediction accuracy
    
    #Draw sns plot for depth and accuracy
    sns.lineplot(data=SPLIT, x="split", y="accuracy")

    #Show sns plot
    #plt.show()

except Exception as e: print(e)

#Handle missing data
try: print("After handling the missing value, its achieve a " + str(np.round(missingValue(), 4)) + " accuracy") #Announce prediction accuracy
except Exception as e: print(e)

#Experiment: Algorithm

try:
    bestScore, depth = depth() #Store all return value
    print("With a " + str(depth) + " depth, it achieve a " + str(np.round(bestScore, 4)) + " accuracy") #Announce prediction accuracy
    
    #Draw sns plot for depth and accuracy
    sns.lineplot(data=DEPTH, x="depth", y="accuracy")
    
    #Show sns plot
    #plt.show() 

except Exception as e: print(e)


#Experiment - Final
#!!!Will take a lot of computing power!!!

try:

    bestScore, split, depth = bestSettings() #Store all return value
    print("With a " + str(split) + "% split and " + str(depth) + " depth, it achieve a " + str(np.round(bestScore, 4)) + " accuracy") #Announce prediction accuracy

except Exception as e: print(e)
