# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
from sklearn.svm import SVC
from sklearn import datasets
import matplotlib
import matplotlib.pyplot as plt

# Where to save the figures
WORKING_PATH = os.path.abspath(os.path.join(os.getcwd(), '..'))
ROOT_PATH = os.path.join(WORKING_PATH, 'Hands on SK and TS\\')
CHAPTER_ID = "support_vector_machines"

# to make this notebook's output stable across runs
np.random.seed(42)


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(ROOT_PATH, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # At the decision boundary, w0*x0 + w1*x1 + b = 0 (w0: length, w1: width)
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)


if __name__ == '__main__':

    # To plot pretty figures
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    # load iris data
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]  # petal length, petal width like [[length, width], [], [], ...]
    y = iris["target"]

    setosa_or_versicolor = (y == 0) | (y == 1)  # target 0 or 1
    X = X[setosa_or_versicolor]
    y = y[setosa_or_versicolor]

    # SVM Classifier model
    svm_clf = SVC(kernel="linear", C=float("inf"))
    svm_clf.fit(X, y)
    print(svm_clf.coef_)  # [[1.29411744 0.82352928]] two features so two columns: length, width
    print(svm_clf.intercept_)  # [-3.78823471]

    # define three lines
    x0 = np.linspace(0, 5.5, 200)
    pred_1 = 5 * x0 - 20  # green dot line
    pred_2 = x0 - 1.8  # purple line
    pred_3 = 0.1 * x0 + 0.5  # red line

    # plot two margin images
    plt.figure(figsize=(12, 2.7))

    plt.subplot(121)  # the left image: three color lines
    plt.plot(x0, pred_1, "g--", linewidth=2)
    plt.plot(x0, pred_2, "m-", linewidth=2)
    plt.plot(x0, pred_3, "r-", linewidth=2)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", label="Iris-Versicolor")  # blue square
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", label="Iris-Setosa")  # yellow circle
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="upper left", fontsize=14)
    plt.axis([0, 5.5, 0, 2])

    plt.subplot(122)  # the right image: margin
    plot_svc_decision_boundary(svm_clf, 0, 5.5)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs")
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo")
    plt.xlabel("Petal length", fontsize=14)
    plt.axis([0, 5.5, 0, 2])

    save_fig("large_margin_classification_plot")
    plt.show()