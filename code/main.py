import os
from pathlib import Path

import numpy as np

from data_preprocessing import get_heart
from data_preprocessing import get_breastCancer
from data_preprocessing import get_liver
from data_preprocessing import get_hepatitis

from models import SVM
from models import RandomForests

from visualization import calculate_metrics
from visualization import plot_confusion_matrix
from visualization import visualize_metrics_bar_chart

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

def main():
    
    models = ['SVM', 'Random Forest']
    dataset_types = ['Heart Disease', 'Breast Cancer', 'Liver Disorder', 'Hepatitis']

    # Heart
    X_train, X_test, y_train, y_test, heart_classes = get_heart()
    y_pred_heart_svm, y_true_heart_svm = SVM(X_train, X_test, y_train, y_test)
    y_pred_heart_rf, y_true_heart_rf = RandomForests(X_train, X_test, y_train, y_test)
    #Plot confusion matrix
    y_true_heart = y_true_heart_rf
    y_preds_heart = [y_pred_heart_svm, y_pred_heart_rf]
    plot_confusion_matrix(y_true_heart, y_preds_heart, heart_classes, models, 'heart_disease')

    # Breast Cancer
    X_train, X_test, y_train, y_test, breast_classes = get_breastCancer()
    y_pred_breast_svm, y_true_breast_svm = SVM(X_train, X_test, y_train, y_test)
    y_pred_breast_rf, y_true_breast_rf = RandomForests(X_train, X_test, y_train, y_test)
    #Plot confusion matrix
    y_true_breast = y_true_breast_rf
    y_preds_breast = [y_pred_breast_svm, y_pred_breast_rf]
    plot_confusion_matrix(y_true_breast, y_preds_breast, breast_classes, models, 'breast_cancer')

    # Liver
    X_train, X_test, y_train, y_test, liver_classes = get_liver()
    y_pred_liver_svm, y_true_liver_svm = SVM(X_train, X_test, y_train, y_test)
    y_pred_liver_rf, y_true_liver_rf = RandomForests(X_train, X_test, y_train, y_test)
    #Plot confusion matrix
    y_true_liver = y_true_liver_rf
    y_preds_liver = [y_pred_liver_svm, y_pred_liver_rf]
    plot_confusion_matrix(y_true_liver, y_preds_liver, liver_classes, models, 'liver_disorders')

    # Hepatitis
    X_train, X_test, y_train, y_test, hep_classes = get_hepatitis()
    y_pred_hep_svm, y_true_hep_svm = SVM(X_train, X_test, y_train, y_test)
    y_pred_hep_rf, y_true_hep_rf = RandomForests(X_train, X_test, y_train, y_test)
    #Plot confusion matrix
    y_true_hep = y_true_hep_rf
    y_preds_hep = [y_pred_hep_svm, y_pred_hep_rf]
    plot_confusion_matrix(y_true_hep, y_preds_hep, hep_classes, models, 'hepatitis')

    #Obtain metric values
    metrics_values = np.array([
        calculate_metrics(y_true_heart_svm, y_pred_heart_svm),
        calculate_metrics(y_true_heart_rf, y_pred_heart_rf),
        calculate_metrics(y_true_breast_svm, y_pred_breast_svm),
        calculate_metrics(y_true_breast_rf, y_pred_breast_rf),
        calculate_metrics(y_true_liver_svm, y_pred_liver_svm),
        calculate_metrics(y_true_liver_rf, y_pred_liver_rf),
        calculate_metrics(y_true_hep_svm, y_pred_hep_svm),
        calculate_metrics(y_true_hep_rf, y_pred_hep_rf)
    ])
    print(metrics_values)
    # Reshape the metrics_values to have the shape (num_datasets, num_models, num_metrics)
    metrics_values = metrics_values.reshape((len(dataset_types), len(models), -1))
    # Visualize metrics
    visualize_metrics_bar_chart(metrics_values, models, dataset_types)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
