import os
from pathlib import Path

import pandas as pd

from data_preprocessing import get_heart
from data_preprocessing import get_breastCancer
from data_preprocessing import get_liver
from data_preprocessing import get_hepatitis
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())


def SVM(X_train, X_test, y_train, y_test):

    # format y datasets to allow for use in SVM func
    new_y_train= pd.DataFrame(data= y_train)
    new_y_test= pd.DataFrame(data= y_test)

    # train model
    clf = svm.SVC(kernel='linear', C = 1)
    clf.fit(X_train, new_y_train.values.ravel())

    # get y_predict and find accuracy
    y_predict = clf.predict(X_test)
    compare = new_y_test.values.ravel() == y_predict
    accuracy = sum(compare)/len(compare)

    return y_predict, accuracy

def RandomForests(X_train, X_test, y_train, y_test):

    # format y datasets to allow for use in SVM func
    new_y_train= pd.DataFrame(data= y_train)
    new_y_test= pd.DataFrame(data= y_test)

    # train model
    rf = RandomForestClassifier()
    rf.fit(X_train, new_y_train.values.ravel())

    # get y_predict and find accuracy
    y_predict = rf.predict(X_test)
    compare = new_y_test.values.ravel() == y_predict
    accuracy = sum(compare)/len(compare)

    return y_predict, accuracy


def main():
    #SVM predictions:

    # Heart
    X_train, X_test, y_train, y_test = get_heart()
    print('\nHeart:')
    y_predict_HeartSVM, accuracy = SVM(X_train, X_test, y_train, y_test)
    print('SVM accuracy = ', accuracy)
    y_predict, accuracy = RandomForests(X_train, X_test, y_train, y_test)
    print('Random Forests accuracy = ', accuracy)

    # Breast Cancer
    X_train, X_test, y_train, y_test = get_breastCancer()
    print('\nBreast Cancer:')
    y_predict, accuracy = SVM(X_train, X_test, y_train, y_test)
    print('SVM accuracy = ', accuracy)
    y_predict, accuracy = RandomForests(X_train, X_test, y_train, y_test)
    print('Random Forests accuracy = ', accuracy)

    # Liver
    X_train, X_test, y_train, y_test = get_liver()
    print('\nLiver:')
    y_predict, accuracy = SVM(X_train, X_test, y_train, y_test)
    print('SVM accuracy = ', accuracy)
    y_predict, accuracy = RandomForests(X_train, X_test, y_train, y_test)
    print('Random Forests accuracy = ', accuracy)

    # Hepatitis
    X_train, X_test, y_train, y_test = get_hepatitis()
    print('\nHepatitis:')
    y_predict, accuracy = SVM(X_train, X_test, y_train, y_test)
    print('SVM accuracy = ', accuracy)
    y_predict, accuracy = RandomForests(X_train, X_test, y_train, y_test)
    print('Random Forests accuracy = ', accuracy)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
