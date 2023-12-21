import os
from pathlib import Path

import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def SVM(X_train, X_test, y_train, y_test):

    # format y datasets to allow for use in SVM func
    new_y_train= pd.DataFrame(data= y_train)
    new_y_test= pd.DataFrame(data= y_test)

    # train model
    parameters = {'C': [0.01, 0.1, 1, 10, 100]}
    svc = svm.SVC(kernel='rbf')
    clf = GridSearchCV(svc, parameters)
    clf.fit(X_train, new_y_train.values.ravel())

    # get y_predict and find accuracy
    y_predict = clf.predict(X_test)

    return y_predict, new_y_test


def RandomForests(X_train, X_test, y_train, y_test):

    # format y datasets to allow for use in SVM func
    new_y_train= pd.DataFrame(data= y_train)
    new_y_test= pd.DataFrame(data= y_test)

    # train model
    parameters = {'n_estimators': [10, 50, 100, 150, 200]}
    rf = RandomForestClassifier(random_state=123)
    clf = GridSearchCV(rf, parameters)
    clf.fit(X_train, new_y_train.values.ravel())

    # get y_predict and find accuracy
    y_predict = clf.predict(X_test)

    return y_predict, new_y_test
