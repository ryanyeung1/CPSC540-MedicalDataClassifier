import os
from pathlib import Path

import pandas as pd

from data_preprocessing import get_heart
from data_preprocessing import get_breastCancer
from data_preprocessing import get_liver
from data_preprocessing import get_hepatitis
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


def SVM(X_train, X_test, y_train, y_test):

    # format y datasets to allow for use in SVM func
    new_y_train= pd.DataFrame(data= y_train)
    new_y_test= pd.DataFrame(data= y_test)

    # train model
    clf = svm.SVC(kernel='linear', C = 1)
    clf.fit(X_train, new_y_train.values.ravel())

    # get y_predict and find accuracy
    y_predict = clf.predict(X_test)

    return y_predict, new_y_test

def RandomForests(X_train, X_test, y_train, y_test):

    # format y datasets to allow for use in SVM func
    new_y_train= pd.DataFrame(data= y_train)
    new_y_test= pd.DataFrame(data= y_test)

    # train model
    rf = RandomForestClassifier()
    rf.fit(X_train, new_y_train.values.ravel())

    # get y_predict and find accuracy
    y_predict = rf.predict(X_test)

    return y_predict, new_y_test
