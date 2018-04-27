# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 22:35:51 2016

@author: hassan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer,  CountVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering
#from sklearn.feature_extraction import text 
import re # for regular expression
import nltk
from nltk.corpus import stopwords # Import the stop word list


np.set_printoptions(precision=4) 
df = pd.read_csv('Documents.csv', header=0)
#print(df.info())
print df.shape   #make sure that we read 
print df["SMS"][0] #check the first sms

#*********************************************
#remove punctuation and numbers,
def cleaning(sms):
    return re.sub("[^a-zA-Z]", " ", sms)     
        
df['cleansms']=df["SMS"].apply(cleaning) #add a column in df with clean sms
#*********************************************
#remove stop words
#nltk.download()  # Download text data sets, including stop words
df["cleansms"] = df["cleansms"].str.lower().str.split() #to lower case and split them into individual words (called "tokenization")
stop = set(stopwords.words("english")) # set is faster than list
df['cleansms'] = df['cleansms'].apply(lambda x: [item for item in x if item not in stop])
df['cleansms'] = df['cleansms'].apply(lambda x: " ".join(x) ) #join the words back into one sms


#**********************************************
#feature extraction using scikit-learn module. creat Bag of Words. calculate tf-idf
# does not good performance

#vectorizer = TfidfVectorizer(analyzer = "word", stop_words='english', min_df=1)
#vsm = vectorizer.fit_transform(df["cleansms"]) #creat vector space model
#vsm = vsm.toarray()



vectorizer = CountVectorizer (analyzer = "word")
vsm = vectorizer.fit_transform(df["cleansms"] )
vsm = vsm.toarray()
#print vsm.shape
#vocab = vectorizer.get_feature_names()
#print vocab
#
## Sum up the counts of each vocabulary word
#dist = np.sum(vsm, axis=0)
#for tag, count in zip(vocab, dist):
#    print count, tag
#*********************************************
#clustering - K-means

model = KMeans(max_iter =300 , n_clusters= 2)
#model = spectral_clustering(vsm , n_clusters=5,affinity='precomputed')
model.fit(vsm)
result = model.labels_

with open('result.csv', 'wb') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["SMS_id", "label"])    
    #wtr = csv.writer(open ('out.csv', 'w'), delimiter=',', lineterminator='\n')
    ind = 0
    for x in result : 
        ind =ind+1
        if x==0:
           writer.writerow ([ind,0])
        else:
            writer.writerow ([ind,1]) 
               
           
#the majority of sms should be pam.
print list(result).count(0) #0 for spam
print list(result).count(1) #1 for ham
#print list(result).count(2) #2 for spam

#*********************************************
#plt.figure(figsize=(14,7))
#colormap = np.array(['red', 'blue'])
#plt.subplot(1, 2)
#plt.scatter(c=colormap[model.labels_], s=40)
#plt.title('K Mean Classification')


