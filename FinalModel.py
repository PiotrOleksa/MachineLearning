import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced
from imblearn.pipeline import Pipeline

def loading_data():
    X = pd.read_csv('train_data.csv', header=None)
    y = pd.read_csv('train_data.csv', header=None, names=['y'])

    return X, y

def coding(y):
    encoder = lambda x: 1 if x == -1 else 0
    decoder = lambda x: -1 if x == 1 else 1
    y= y['y'].apply(encoder)

    return y

def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    return X_train, X_test, y_train, y_test

def pipeline(X_train, X_test, y_train, y_test):
    svc = SVC()
    sc = StandardScaler()
    kpca = KernelPCA(gamma=0.03, kernel='linear')
    pipe = Pipeline(steps=[('scaler', sc), ('decomposition', kpca), ('estimator', svc)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    conf = confusion_matrix(y_pred,y_test)
    report = classification_report_imbalanced(y_pred,y_test)

    return conf, report

if __name__ == '__main__':
    X, y = loading_data()
    y = coding(y)
    X_train, X_test, y_train, y_test = split(X, y)
    conf, report = pipeline(X_train, X_test, y_train, y_test)
    print('Confusion Matrix: ', conf)
    print('Classification Report Imbalanced: ', report)


