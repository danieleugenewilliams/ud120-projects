#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn import svm
clf = svm.SVC(C=10000.0, kernel="rbf")

#Cutting out 99% of training data...
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100]

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
#pred = clf.predict(features_test)

# Predict 10th, 26th, 50th
#pred10 = clf.predict(features_test[10])
#pred26 = clf.predict(features_test[26])
#pred50 = clf.predict(features_test[50])

#print "10th:", pred10
#print "26th:", pred26
#print "50th:", pred50

# Predict which emails belong to Chris (1)
pred = clf.predict(features_test)
print "pred:", pred.tolist().count(1)

print "prediction time:", round(time()-t1, 3), "s"

#from sklearn.metrics import accuracy_score
#print accuracy_score(pred, labels_test)


#########################################################


