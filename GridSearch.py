import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
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

def grid_search(X_train, y_train):
    pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('decomposition', PCA()), ('estimator', SVC())])
    gridsearch = GridSearchCV(pipe, verbose = 2, cv=4, n_jobs = -1, scoring = ['f1_weighted','recall_weighted','precision_weighted'],refit = False,)
    best = gridsearch.fit(X_train, y_train)

    return best

if __name__ == '__main__':
    X, y = loading_data()
    y = coding(y)
    X_train, X_test, y_train, y_test = split(X, y)
    best = grid_search(X_train, y_train)
    print('Grid Search Result: ', best)