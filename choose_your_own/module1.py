#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.neighbors import KNeighborsClassifier
from time import time

from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
(features_train, features_test, labels_train, labels_test )= preprocess()
features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100]


clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(features_train, labels_train)
t0 = time()
pred=clf.predict(features_test)
print(clf.score(features_test, labels_test))
print ("training time:", round(time()-t0, 3), "s")

