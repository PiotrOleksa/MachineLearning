import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression, LinearRegression
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")


def loading_data():
    X = pd.read_csv(r'C:\Users\piotr\Jupyter_Notebooks\data\train_data.csv', header=None)
    y = pd.read_csv(r'C:\Users\piotr\Jupyter_Notebooks\data\train_labels.csv', header=None)

    return X, y

def coding(y):
    encoder = lambda x: 1 if x == -1 else 0
    decoder = lambda x: -1 if x == 1 else 1
    y= y['y'].apply(encoder)

    return y

def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    return X_train, X_test, y_train, y_test


def AUC(X,y):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
                                                        test_size=.25,
                                                        random_state=1234)
    classifiers = [LogisticRegression(random_state=1234), 
                   GaussianNB(),
                   SVC(probability=True),
                   KNeighborsClassifier(), 
                   DecisionTreeClassifier(random_state=1234),
                   RandomForestClassifier(random_state=1234)]
    
    # Define a result table as a DataFrame
    result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])
    
    # Train the models and record the results
    for cls in classifiers:
        model = cls.fit(X_train, y_train)
        yproba = model.predict_proba(X_test)[::,1]
        
        fpr, tpr, _ = roc_curve(y_test,  yproba)
        auc = roc_auc_score(y_test, yproba)
        
        result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                            'fpr':fpr, 
                                            'tpr':tpr, 
                                            'auc':auc}, ignore_index=True)
    
    # Set name of the classifiers as index labels
    result_table.set_index('classifiers', inplace=True)
    return result_table


def AUC_plot(result_table):
    fig = plt.figure(figsize=(8,6))
    
    for i in result_table.index:
        plt.plot(result_table.loc[i]['fpr'], 
                 result_table.loc[i]['tpr'], 
                 label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
        
    plt.plot([0,1], [0,1], color='orange', linestyle='--')
    
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)
    
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)
    
    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')
    
    plt.show()
    
    
def confusion_matrix_func():
    model = SVC()
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    matrix = confusion_matrix(y_test, predicted)
    print(matrix)

def c_report():
    model = SVC()
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    report = classification_report(y_test, predicted)
    print(report)
    
def mae():
    kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
    model = SVC()
    scoring = 'neg_mean_absolute_error'
    results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    print("MAE: %.3f (%.3f)" % (results.mean(), results.std()))

    
def r2():
    kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
    model = SVC()
    scoring = 'r2'
    results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    print("R^2: %.3f (%.3f)" % (results.mean(), results.std()))


if __name__ == '__main__':
    X, y = loading_data()
    X_train, X_test, y_train, y_test = split(X, y)
    result_table = AUC(X,y)
    AUC_plot(result_table)
    confusion_matrix_func()
    c_report()
    mae()
    r2()
    

