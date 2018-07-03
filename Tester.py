# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 14:41:51 2018

@author: t-nehulk
"""

# IMPORTS
import pandas as pd

import numpy as np
from numpy import *

import io

from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

## ACCESS FILE 
table = pd.read_table("C:/Users/t-nehulk/.spyder-py3/TestDataset/UnifiedDataset.txt", sep='\t')
df = pd.DataFrame(table)
    
## MAKE FEATURES
columns = ['ifStartsWithNumber', 'QueryLength', 'NumberOfDigits', 'NumberOfSpaces', 'NumberOfCommas', 'IfEndsInSpace',  'NumberOfCapitalLetters']#'SizeOfBoundingBox', 'IfHas5DigitNumber']
features = pd.DataFrame(columns=columns, index=df.index)

for i in range(0, len(df)):
    features.loc[i,'IfStartsWithNumber'] = 'beep'
