# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 16:33:50 2016

@author: hassan
"""

import csv
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid



# is vsm for training data_ 
 # takes time to fetch
mat_word_training = np.genfromtxt ('news-word_Training.csv', delimiter=",")
mat_word_test = np.genfromtxt ('news-word_Test.csv', delimiter=",") 
mat_label = np.genfromtxt ('label_training.csv', delimiter=",", skip_header=1) 

#ch2 = SelectKBest(chi2, k=opts.select_chi2)
#X_train = ch2.fit_transform(X_train, y_train)
#X_test = ch2.transform(X_test)


ch2 = SelectKBest(chi2, "all") #result= %58
#ch2 = SelectKBest(chi2) #10 features- result= %59
mat_new_features = ch2.fit_transform(mat_word_training, mat_label)
mat_selected_features = ch2.transform(mat_word_test)

#----------------------------------
#using classifier

classifier = KNeighborsClassifier(10)

classifier.fit(mat_new_features, mat_label)
predictions = classifier.predict(mat_selected_features)


#------------------------------------
#Write results to file
with open('result.csv', 'wb') as outcsv:
    writer = csv.writer(outcsv, delimiter=',', quoting = csv.QUOTE_NONE, quotechar='')
    writer.writerow(["id-news", "label"])    
    for ind in range(0,1459): 
        #doc_id = df_test["document_id"][ind]
        #doc_id =df_test["document_id"][ind]
        id = ind+1
        writer.writerow ([id,int(predictions[ind])])
 
 
