# -*- coding: utf-8 -*-
"""
Created on Fri Dec 02 16:43:39 2016

@author: hassan
"""

import pandas as pd
import numpy as np
import csv
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix




messages = pd.read_csv('SMSSpamCollection.txt', sep='\t', quoting=csv.QUOTE_NONE,
                           names=["label", "message"])
#print df

#for message_no, message in enumerate(messages[:10]):
#    print message_no, message



print messages.groupby('label').describe() #view aggregate statistics
#print messages.length.describe()

#def split_into_tokens(message):
#    message = unicode(message, 'utf8')  # convert bytes into proper unicode
#    return TextBlob(message).words
#
#messages.message.head().apply(split_into_tokens)
#normalize words into their base form (lemmas)
#def split_into_lemmas(message):
#    message = unicode(message, 'utf8').lower()
#    words = TextBlob(message).words
#    # for each word, take its "base form" = lemma 
#    return [word.lemma for word in words]
#
#messages.message.head().apply(split_into_lemmas)
#-------------------------------------------------------
#convert each message, represented as a list of tokens (lemmas) above, into a vector
#bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['message'])
bow_transformer = CountVectorizer(analyzer="word", stop_words= 'english', lowercase = 'true').fit(messages['message'])

print len(bow_transformer.vocabulary_)
#bag of words
messages_bow = bow_transformer.transform(messages['message'])
tfidf_transformer = TfidfTransformer().fit(messages_bow)

#the IDF  of word 'university'
print tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]
messages_tfidf = tfidf_transformer.transform(messages_bow)

#---------------------------------------------------------
#Naive Byse
# fit is used for traiing the model
spam_detector = MultinomialNB().fit(messages_tfidf, messages['label'])
all_predictions = spam_detector.predict(messages_tfidf)
print 'accuracy', accuracy_score(messages['label'], all_predictions)

#for check
message4 = messages['message'][3]
print message4
bow4 = bow_transformer.transform([message4])
tfidf4 = tfidf_transformer.transform(bow4)
print 'predicted:', spam_detector.predict(tfidf4)[0]
print 'expected:', messages.label[3]
#-------------------------------------------------------

with open('result1.csv', 'wb') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["SMS_id", "label"])    
    #wtr = csv.writer(open ('out.csv', 'w'), delimiter=',', lineterminator='\n')
    ind = 0
    for x in all_predictions : 
        ind =ind+1
        if x== 'ham':
           writer.writerow ([ind,1])
        else:
            writer.writerow ([ind,0]) 

