import pandas as pd

from sklearn.model_selection import train_test_split
from imblearn.metrics import classification_report_imbalanced
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

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

def dummy_class(X_train, y_train, X_test, y_test):
    dc = DummyClassifier(strategy='stratified')
    fit = dc.fit(X_train, y_train)
    y_pred = dc.predict(X_test)
    report = classification_report_imbalanced(y_test, y_pred)

    return report

def decision_tree(X_train, y_train, X_test, y_test):
    dtc = DecisionTreeClassifier()
    fit = dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    report = classification_report_imbalanced(y_test, y_pred)

    return report

if __name__ == '__main__':
    X, y = loading_data()
    y = coding(y)
    X_train, X_test, y_train, y_test = split(X, y)
    report_dc = dummy_class(X_train, y_train, X_test, y_test)
    report_dt = decision_tree(X_train, y_train, X_test, y_test)
    print('Dummy Class: ', report_dc)
    print('Decision Tree: ',  report_dt)

