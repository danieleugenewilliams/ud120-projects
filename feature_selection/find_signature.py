#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)
from sklearn import tree
from sklearn.metrics import accuracy_score


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150]
labels_train   = labels_train[:150]



### your code goes here
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

print("training points count:", len(features_train.toarray()))
print("acc:", accuracy_score(pred, labels_test))

cnt = 0
for fi in clf.feature_importances_:
    
    if fi > 0.0:
        print("feature importance:", cnt, ": ", fi)
    cnt += 1

print("21323", vectorizer.get_feature_names()[21323])

"""
('training points count:', 150)
('acc:', 0.94766780432309439)
('feature importance:', 19671, ': ', 0.07495003331112593)
('feature importance:', 24321, ': ', 0.026315789473684213)
('feature importance:', 33201, ': ', 0.13402829486224865)
('feature importance:', 33614, ': ', 0.76470588235294124)
"""
