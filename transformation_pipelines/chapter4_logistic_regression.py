# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

# Where to save the figures
WORKING_PATH = os.path.abspath(os.path.join(os.getcwd(), '..'))
ROOT_PATH = os.path.join(WORKING_PATH, 'Hands on SK and TS\\')
CHAPTER_ID = "training_linear_models"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(ROOT_PATH, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


if __name__ == '__main__':

    # Estimating Probabilities

    # plot logistic function curve
    t = np.linspace(-10, 10, 100)  # 100 x points with the same interval in (-10, 10)
    sig = 1 / (1 + np.exp(-t))  # sigmoid
    plt.figure(figsize=(9, 3))
    plt.plot([-10, 10], [0, 0], "k-")  # (x range, y range, color and line)
    plt.plot([-10, 10], [0.5, 0.5], "k:")  # : point
    plt.plot([-10, 10], [1, 1], "k:")
    plt.plot([0, 0], [-1.1, 1.1], "k-")
    plt.plot(t, sig, "b-", linewidth=2, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")
    '''
        ``'b'``          blue
        ``'g'``          green
        ``'r'``          red
        ``'c'``          cyan
        ``'m'``          magenta
        ``'y'``          yellow
        ``'k'``          black
        ``'w'``          white
    '''
    plt.xlabel("t")
    plt.legend(loc="upper left", fontsize=20)
    plt.axis([-10, 10, -0.1, 1.1])
    save_fig("logistic_function_plot")
    # plt.show()

    # Decision Boundaries

    # classify three kinds of iris flowers
    iris = datasets.load_iris()
    print(list(iris.keys()))  # ['data', 'target', 'target_names', 'DESCR', 'feature_names']
    print(iris.DESCR)
    X = iris["data"][:, 3:]  # show petal width
    y = (iris["target"] == 2).astype(np.int)  # y is a list with: 1 if Iris-Virginica, else 0
    # target 1: Iris-Setosa     2: Iris-Versicolour     3: Iris-Virginica

    # define a logistic regression model
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X, y)
    '''LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=42, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)'''

    # make petal width data set using np
    X_new = np.linspace(0, 3, 1000).reshape(-1, 1)  # petal width in (0, 3). X_new is a list with petal width.
    # reshape(row, column) if we don't know the row #, make it -1. So (-1, 1) reshape it into one column and total rows.
    y_proba = log_reg.predict_proba(X_new)  # estimating probabilities.
    # y_proba is a list like [[Not Virginica Proba, Virginica Proba], [], [], ...] and the Proba is in (0, 1)
    decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]  # the first element in X_new list whose proba >=0.5 to target 2

    plt.figure(figsize=(10, 4))
    plt.plot(X[y == 0], y[y == 0], "bs")  # blue box: refer to _axes.py line 1485
    plt.plot(X[y == 1], y[y == 1], "g^")  # green triangle
    plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)  # black vertical dot line
    plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris-Virginica")
    plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris-Virginica")
    plt.text(decision_boundary + 0.02, 0.15, "Decision  boundary", fontsize=14, color="k", ha="center")
    plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')  # blue arrow
    plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='g', ec='g')  # green arrow
    plt.xlabel("Petal width (cm)", fontsize=14)
    plt.ylabel("Probability", fontsize=14)
    plt.legend(loc="center left", fontsize=14)
    plt.axis([0, 3, -0.02, 1.02])
    save_fig("logistic_regression_plot")
    plt.show()