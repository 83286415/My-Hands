# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# Chapter import
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_mldata
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from deslib.des.knora_e import KNORAE

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


def plot_regression_predictions(tree_reg, X, y, axes=[0, 1, -0.2, 1], ylabel="$y$"):
    x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
    y_pred = tree_reg.predict(x1)
    plt.axis(axes)
    plt.xlabel("$x_1$", fontsize=18)
    if ylabel:
        plt.ylabel(ylabel, fontsize=18, rotation=0)
    plt.plot(X, y, "b.")  # plot X points in blue
    plt.plot(x1, y_pred, "r.-", linewidth=2, label=r"$\hat{y}$")  # red horizon lines: the average values


def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)


def plot_digit(data):
    image = data.reshape(28, 28)  # reshape (70000, 784) into (28, 28)
    plt.imshow(image, cmap=matplotlib.cm.hot, interpolation="nearest")  # cmap: color map;
    # interpolation: refer to cloud note matplotlib
    plt.axis("off")  # no axis


if __name__ == '__main__':

    # Exercise 8:

    # dataa set
    mnist = fetch_mldata('MNIST original')
    X, y = mnist['data'], mnist['target']

    train_size, validation_size, test_size = 40000, 10000, 10000
    rnd_indices = np.random.permutation(60000)  # shuffle data set and return th index

    # train, validation and test data set
    X_train = X[rnd_indices[:train_size]]
    y_train = y[rnd_indices[:train_size]]
    X_valid = X[rnd_indices[train_size:-test_size]]
    y_valid = y[rnd_indices[train_size:-test_size]]
    X_test = X[rnd_indices[-test_size:]]
    y_test = y[rnd_indices[-test_size:]]
    X_train_dsel, X_dsel, y_train_dsel, y_dsel = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

    # build models
    rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
    extra_clf = ExtraTreesClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
    svm_clf = SVC(probability=True, kernel="linear", C=float("inf"))

    # fit and predict
    rnd_clf.fit(X_train, y_train)
    extra_clf.fit(X_train, y_train)
    svm_clf.fit(X_train, y_train)

    hard_voting_clf = VotingClassifier(
        estimators=[('rf', rnd_clf), ('ex', ), ('svc', svm_clf)], voting='hard')  # hard voting

    soft_voting_clf = VotingClassifier(
        estimators=[('rf', rnd_clf), ('ex', ), ('svc', svm_clf)], voting='soft')  # soft voting

    # show each classifier's accuarcy score
    for clf in (rnd_clf, extra_clf, svm_clf, hard_voting_clf, soft_voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

    # Exercise: 9

    knorae = KNORAE(rnd_clf)
    knorae.fit(X_dsel, y_dsel)
    print(knorae.__class__.__name__, accuracy_score(y_test, y_pred))
