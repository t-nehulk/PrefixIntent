# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# IMPORTS
import pandas as pd
import numpy as np
from numpy import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

## ACCESS FILE 
table = pd.read_table("C:/Users/t-nehulk/.spyder-py3/TestDataset/UnifiedDataset.txt", sep='\t')
df = pd.DataFrame(table)
    
## MAKE FEATURES
columns = ['IfStartsWithNumber', 'QueryLength', 'NumberOfDigits', 'NumberOfSpaces', 'NumberOfCommas', 'IfEndsInSpace',  'NumberOfCapitalLetters']#'SizeOfBoundingBox', 'IfHas5DigitNumber']
features = pd.DataFrame(columns=columns, index=df.index)

## ifStartsWithNumber feature (column 0)
## iterate along each row
def ifStartsWithNumber():
    for i in range(0, len(df)):
        ## if the first char is a digit, set index to 1
        if (df.iloc[i][3][:1].isdigit()):
            features.loc[i,'IfStartsWithNumber'] = 1;
        ## else, set value to 0
        else:
            features.loc[i,'IfStartsWithNumber'] = 0;

## QueryLength feature (column 1)
def QueryLength():
    for i in range(0, len(df)):
        ## each row is query length
        features.loc[i,'QueryLength'] = len(df.iloc[i][3])

## NumberOfDigits feature (column 2)
def NumberOfDigits():
    for i in range(0, len(df)):
        query = df.iloc[i][3]
        ## count variable
        count = 0
        ## iterate along each char of query
        for j in range(0, len(query)):
            ## if current char is digit, increase count
            if(query[j].isdigit()):
                count=count+1
        features.loc[i,'NumberOfDigits'] = count

## NumberOfSpaces feature (column 3)
def NumberOfSpaces():
    for i in range(0, len(df)):
        query = df.iloc[i][3]
        ## count variable
        count = 0
        ## iterate along each char of query
        for j in range(0, len(query)):
            ## if current char is space, increase count
            if(query[j] == ' '):
                count=count+1
        features.loc[i,'NumberOfSpaces'] = count

## NumberOfCommas feature (column 4)
def NumberOfCommas():
    for i in range(0, len(df)):
        query = df.iloc[i][3]
        ## count variable
        count = 0
        ## iterate along each char of query
        for j in range(0, len(query)):
            ## if current char is comma, increase count
            if(query[j] == ','):
                count=count+1
        features.loc[i,'NumberOfCommas'] = count

## IfEndsInSpace feature (column 5)
def IfEndsInSpace():
    for i in range(0, len(df)):
        query = df.iloc[i][3]
        ## if the last char is a space, set index to 1
        if (query[len(query)-1] == ' '):
            features.loc[i,'IfEndsInSpace'] = 1;
        ## else, set value to 0
        else:
            features.loc[i,'IfEndsInSpace'] = 0;

## NumberOfCapitalLetters feature (column 6)
def NumberOfCapitalLetters():
    ## create list to store feature
    #NumberOfCapitalLetters = []
    for i in range(0, len(df)):        
        query = df.iloc[i][3]
        ## count variable
        count = 0
        ## iterate along each char of query
        for j in range(0, len(query)):
            ## if current char is capital letter, increase count
            if(query[j].isupper()):
                count=count+1
        features.loc[i,'NumberOfCapitalLetters'] = count;
        ## store the count 
        #NumberOfCapitalLetters.append(count)
    ## add NumberOfCapitalLetters feature to feature Dataframe
    #features['NumberOfCapitalLetters'] = NumberOfCapitalLetters 

## SizeOfBoundingBox feature (column 7)
def SizeOfBoundingBox():
    for i in range(0, len(df)):
         ## creates array of 4 edges of bounding box
         edges = df.iloc[i][8].split(',')
         ## calculates length of bounding box
         length = float(edges[0]) - float(edges[2])
         ## calculates width of bounding box
         width = float(edges[1]) - float(edges[3])
         ## calculates area by multiplying length and width
         area = length*width
         ## set value of row to area
         features.loc[i,'SizeOfBoundingBox'] = area;

## IfHas5DigitNumber feature (column 8)
def IfHas5DigitNumber():
    ## create list to store feature
    IfHas5DigitNumber = []
    for i in range(0, len(df)):
        query = df.iloc[i][3]
        ## count variable
        count = 0
        ## iterate along each char of query
        for j in range(0, len(query)):
            ## if current char is a digit, add to count variable
            if(query[j].isdigit()):
                count=count+1
            ## if there is a word with 5 or more numbers, set feature to 1 and end method
            if (count > 4):
                IfHas5DigitNumber.append(1);
                break
        ## if count never reaches 5 digits, set feature to 0
        if (count < 5):
            IfHas5DigitNumber.append(0);
    ## add IfHas5DigitNumber feature to feature Dataframe
    features['IfHas5DigitNumber'] = IfHas5DigitNumber  
                
#print(features)

## SPLIT TRAINING/TESTING DATA
#def TrainTestSplit():
#    X = features
#    y = df['Type']
#    ## 70/30 split 
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

## Bag of Words feature (separate functions for training and testing data, so done after split)
#def BagOfWords():
#    ## fill column with queries
#    features.iloc['Query'] = df.iloc['Query']
#    ## Vectorize data 
#    vect = CountVectorizer()
#    XTrainDTM = vect.fit_transform()
    
    
def ML():
    
    X = features
    y = df['Type']
    ## 70/30 split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    ## vectorize data
    #vect = CountVectorizer()
    #X_trainDTM = vect.fit_transform(X_train)
    #X_testDTM = vect.transform(X_test)
    
    ## PICK MODEL
    #model = LogisticRegression();
    #model = KNeighborsClassifier();
    #model = DecisionTreeClassifier();
    #model = SVC();
    model = RandomForestClassifier();
        
    ## FIT MODEL
    model.fit(X_train, y_train)
        
    ## PREDICT
    pred = model.predict(X_test)
        
    ## GET METRICS
    print('Accuracy')
    print(metrics.accuracy_score(y_test, pred))
    print(metrics.classification_report(y_test, pred))

def Main():
    ifStartsWithNumber()
    QueryLength()
    NumberOfDigits()
    NumberOfSpaces()
    NumberOfCommas()
    IfEndsInSpace()
    NumberOfCapitalLetters()
    #SizeOfBoundingBox()
    #IfHas5DigitNumber()
    #TrainTestSplit()
    ML()



