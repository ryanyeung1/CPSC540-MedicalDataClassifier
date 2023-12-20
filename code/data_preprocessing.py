from ucimlrepo import fetch_ucirepo
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
import numpy as np


# Imports Heart Disease dataset from UCI repository, performs preprocessing and
# returns X_train, X_test, y_train, y_test
def get_heart():
    heart_disease = fetch_ucirepo(id=45)

    # data (as pandas dataframes)
    X = heart_disease.data.features
    y = heart_disease.data.targets

    # Drop examples with missing values
    y = y.drop(X[X.isna().any(axis=1)].index)
    X = X.dropna()
    X = X.to_numpy()
    y = y.to_numpy()

    # Encode labels such that 1 is 'positive' and 0 is 'negative'
    y[y > 0] = 1
    classes = [1, 0]

    #Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Robust scaling
    robustscalar = preprocessing.RobustScaler()
    X_train_scaled = robustscalar.fit_transform(X_train)
    X_test_scaled = robustscalar.transform(X_test)

    # Perform standardized scaling
    # scaler = preprocessing.StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, classes


# Imports Breast Cancer dataset from UCI repository, performs preprocessing and
# returns X_train, X_test, y_train, y_test
def get_breastCancer():
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

    # data (as pandas dataframes)
    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets

    X = X.to_numpy()
    y = y.to_numpy()

    # Encode labels such that 1 is 'positive' and 0 is 'negative'
    y = np.unique(y, return_inverse=True)[1]
    classes = [1, 0]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Robust scaling
    robustscalar = preprocessing.RobustScaler()
    X_train_scaled = robustscalar.fit_transform(X_train)
    X_test_scaled = robustscalar.transform(X_test)

    # Perform standardized scaling
    # scaler = preprocessing.StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, classes


# Imports Liver Disease dataset from UCI repository, performs preprocessing and
# returns X_train, X_test, y_train, y_test
def get_liver():
    liver_disorders = fetch_ucirepo(id=60)

    # data (as pandas dataframes)
    X = liver_disorders.data.features
    y = liver_disorders.data.targets

    X = X.to_numpy()
    y = y.to_numpy()
    # Encode labels such that < 3 drinks = 0 (negative) and >= 3 drinks = 1 (positive)
    y[y < 3] = 0
    y[y >= 3] = 1
    classes = [1, 0]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Robust scaling
    robustscalar = preprocessing.RobustScaler()
    X_train_scaled = robustscalar.fit_transform(X_train)
    X_test_scaled = robustscalar.transform(X_test)

    # Perform standardized scaling
    # scaler = preprocessing.StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, classes


# Imports Hepatitis dataset from UCI repository, performs preprocessing and
# returns X_train, X_test, y_train, y_test
def get_hepatitis():
    hepatitis = fetch_ucirepo(id=46)

    # data (as pandas dataframes)
    X = hepatitis.data.features
    y = hepatitis.data.targets

    # Remove age and sex features
    #  X = X.drop(['Age', 'Sex'], axis=1)

    # Drop examples with missing values
    # y = y.drop(X[X.isna().any(axis=1)].index)
    # X = X.dropna()

    X = X.to_numpy()
    y = y.to_numpy()

    # Encode labels such that 1 is 'positive' and 0 is 'negative'
    y[y == 2] = 0
    classes = [1, 0]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Imputing missing values with mean instead of removing
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    X_train = imp.fit_transform(X_train)
    X_test = imp.transform(X_test)

    # knnImp = KNNImputer(n_neighbors=1)
    # X_train = knnImp.fit_transform(X_train)
    # X_test = knnImp.transform(X_test)

    # Robust scaling
    robustscalar = preprocessing.RobustScaler()
    X_train_scaled = robustscalar.fit_transform(X_train)
    X_test_scaled = robustscalar.transform(X_test)

    # Perform standardized scaling
    # scaler = preprocessing.StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, classes
