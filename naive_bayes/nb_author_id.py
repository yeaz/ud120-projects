#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################

# Classifier
clf = MultinomialNB()

# Train classifier
fitting_start_time = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-fitting_start_time,3),"s"

# Get prediction for test set
prediction_start_time = time()
pred = clf.predict(features_test)
print "predicting time:", round(time()-prediction_start_time,3),"s"

# Compute accuracy
accuracy = accuracy_score(labels_test, pred)
print accuracy


#########################################################


