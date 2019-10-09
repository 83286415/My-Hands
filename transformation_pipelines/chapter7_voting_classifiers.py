# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# Chapter import
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# To plot pretty figures
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
WORKING_PATH = os.path.abspath(os.path.join(os.getcwd(), '..'))
ROOT_PATH = os.path.join(WORKING_PATH, 'Hands on SK and TS\\')
CHAPTER_ID = "ensemble_learning_and_random_forests"


def save_fig(fig_id, tight_layout=True):
    path = image_path(fig_id) + ".png"
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)  # cannot save file if path doesn't exist


def image_path(fig_id):
    return os.path.join(ROOT_PATH, "images", CHAPTER_ID, fig_id)


if __name__ == '__main__':

    # hard voting

    # moon data set
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # build models
    log_clf = LogisticRegression(random_state=42)
    rnd_clf = RandomForestClassifier(random_state=42)
    svm_clf = SVC(random_state=42)

    voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='hard')  # hard voting
    print(voting_clf.fit(X_train, y_train))
    '''
        VotingClassifier(estimators=[('lr', LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                                                intercept_scaling=1, max_iter=100, multi_class='ovr', 
                                                                n_jobs=1,penalty='l2', random_state=42, 
                                                                solver='liblinear', tol=0.0001,verbose=0, 
                                                                warm_start=False)), 
                                     ('rf', RandomFor...f', max_iter=-1, probability=False, random_state=42, 
                                        shrinking=True,tol=0.001, verbose=False))],
                                    flatten_transform=None, n_jobs=1, voting='hard', weights=None)
         '''  # RandomForests classifier details info needed later

    # show each classifier's accuarcy score
    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))  # voting classifier is the best one
    '''
        LogisticRegression 0.864
        RandomForestClassifier 0.872
        SVC 0.888
        VotingClassifier 0.896
    '''

    # soft voting

    # re-build models with probability added in SVC and soft voting in Voting classifier
    log_clf = LogisticRegression(random_state=42)
    rnd_clf = RandomForestClassifier(random_state=42)
    svm_clf = SVC(probability=True, random_state=42)  # probability added

    voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='soft')  # soft voting
    voting_clf.fit(X_train, y_train)

    # show each classifier's accuarcy score
    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
    '''
       LogisticRegression 0.864
        RandomForestClassifier 0.872
        SVC 0.888
        VotingClassifier 0.912 
    '''