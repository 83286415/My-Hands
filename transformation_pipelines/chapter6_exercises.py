# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# Chapter import
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.base import clone
from scipy.stats import mode

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
CHAPTER_ID = "decision_trees"


def save_fig(fig_id, tight_layout=True):
    path = image_path(fig_id) + ".png"
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)  # cannot save file if path doesn't exist


def image_path(fig_id):
    return os.path.join(ROOT_PATH, "images", CHAPTER_ID, fig_id)


if __name__ == '__main__':

    # Exercises

    # 7: dicision trees model with grid search cv

    # moon data
    Xm, ym = make_moons(n_samples=10000, noise=0.4, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(Xm, ym, test_size=0.2, random_state=42)

    # parameters and esitmator
    params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
    grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1, verbose=1)

    grid_search_cv.fit(X_train, y_train)
    print(grid_search_cv.best_estimator_)  # max_leaf_nodes=17  min_samples_split=2
    '''
    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=17,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='best')
    '''

    # test
    y_pred = grid_search_cv.predict(X_test)
    print(accuracy_score(y_test, y_pred))  # 0.8695

    # 8: grow a forest

    # sub-data set
    n_trees = 1000
    n_instances = 100

    mini_sets = []

    # split X_train into 1000 pieces and each piece has 100 samples reshuffled
    rs = ShuffleSplit(n_splits=n_trees, test_size=len(X_train) - n_instances, random_state=42)
    # refer to its definition and cloud note: sklearn
    for mini_train_index, mini_test_index in rs.split(X_train):
        X_mini_train = X_train[mini_train_index]
        y_mini_train = y_train[mini_train_index]
        mini_sets.append((X_mini_train, y_mini_train))

    # train tree models in forest
    forest = [clone(grid_search_cv.best_estimator_) for _ in range(n_trees)]  # copy: deep copy

    # fit and predict: train 1000 tree models with X_mini_train and predict
    accuracy_scores = []
    for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
        tree.fit(X_mini_train, y_mini_train)
        y_pred = tree.predict(X_test)

        accuracy_scores.append(accuracy_score(y_test, y_pred))
    print(np.mean(accuracy_scores))  # mean accuracy of all test data: 0.8054494999999999

    # get test predictions
    Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)  # empty: returns a new array shape like (1000, 100)
    for tree_index, tree in enumerate(forest):
        Y_pred[tree_index] = tree.predict(X_test)

    # get the most common prediction
    y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)  # return the most common array elements and their counts
    # refer to its definition and cloud note: scipy
    print(y_pred_majority_votes, n_votes)  # [[1 1 0 ... 0 0 0]] [[951 912 963 ... 919 994 602]]
    print(accuracy_score(y_test, y_pred_majority_votes.reshape([-1])))  # the forest accuracy: 0.872
