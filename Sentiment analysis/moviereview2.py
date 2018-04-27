# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 10:34:44 2016

@author: hassan
"""


import pandas as pd
import numpy as np
import csv
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from sklearn.ensemble import RandomForestClassifier
#from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer,  CountVectorizer
from bs4 import BeautifulSoup  #to remove HTML tags
import re # for regular expression
import nltk
from nltk.corpus import stopwords
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB


reviews = pd.read_csv('TrainDataset.tsv', sep='\t', quoting=csv.QUOTE_NONE,
                           names=["id", "label", "review"], header=0)    
#we can use quoting=3 as well

    

 #remove HTML tags   
#example1 = BeautifulSoup(reviews["review"][0])   
#print example1.get_text()


def review_to_words( raw_review ):
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text()
    #2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
   # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
   #set is much faster 
    stops = set(stopwords.words("english"))  
    
    
    #5. Remove stop words
    meaningful_words = [w for w in words if not w in stops] 
    #Join the words back into one string separated by space
    return( " ".join( meaningful_words )) 

#clean_review = review_to_words( reviews["review"][0] )
#print clean_review

num_reviews = reviews["review"].size
clean_train_reviews = []
for i in xrange( 0, num_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_train_reviews.append( review_to_words( reviews["review"][i] ) )


vectorizer = CountVectorizer (analyzer = "word")#, max_features = 15000)
bow = vectorizer.fit_transform(reviews['review'] )

test_reviews = pd.read_csv('TestDataset.tsv', sep='\t', quoting=csv.QUOTE_NONE,
                           names=["id", "review"],header=0)

num_reviews = len(test_reviews["review"])
clean_test_reviews = []

for i in xrange(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print "Review %d of %d\n" % (i+1, num_reviews)
    clean_review = review_to_words( test_reviews["review"][i] )
    clean_test_reviews.append( clean_review )

    
#we only call "transform", not "fit_transform" as we did for the training set.    
bow_test = vectorizer.transform(clean_test_reviews)


#feature selection
##feature selection

model = ExtraTreesClassifier()
model.fit(bow, reviews['label'])
sel_weight = model.feature_importances_
#s=np.sort(selected)
#choose 5000 max
bow = model.transform(bow)
bow_test = model.transform(bow_test)
#bow_test = bow_test.toarray()


#ind = np.argpartition(sel_weight, -5000)[-5000:]
##sel_weight[ind]
#
#bow_seleced_word = bow[:, ind]
##bow  = bow_seleced_word.toarray()
#
#model.fit(bow_seleced_word, reviews['label'])
#
#bow = model.transform(bow_seleced_word)
##bow = bow.toarray()


#----------------------------------------------------------
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit( bow, reviews["label"] )

#naive = MultinomialNB().fit(bow, reviews['label'])





result = forest.predict(bow_test)
#result = naive.predict(bow_test)
output = pd.DataFrame( data={"document_id":test_reviews["id"], "sentiment":result} )

output.to_csv( "result1.csv", index=False, header = ["document_id","sentiment" ],  quoting=3 )


##vsm = vsm.toarray()
#vocab = vectorizer.get_feature_names()
#
## Sum up the counts of each vocabulary word
#dist = np.sum(bow, axis=0)
#
## For each, print the vocabulary word and the number of times it 
## appears in the training set
#for tag, count in zip(vocab, dist):
#    print count, tag

    
    
    
    
#----------------------------------------------------------------------------
#neg=[]
#pos=[]
#l_reviews = list (reviews['label'])
#
##extract the positive and negative reviews
#for i in range(0,25000):
#    if l_reviews[i]  == 0:
#        neg.append(i)
#    else:
#        pos.append(i)        
#        
#def word_feats(words):
#    return dict([(word, True) for word in words])
#    
#negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
#posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]
#            
            
            
            
            
            