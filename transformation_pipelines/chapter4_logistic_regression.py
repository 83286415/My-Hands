# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
    # plot a image to show classification of iris based only one feature: petal width

    # classify three kinds of iris flowers
    iris = datasets.load_iris()
    print(list(iris.keys()))  # ['data', 'target', 'target_names', 'DESCR', 'feature_names']
    print(iris.DESCR)
    X = iris["data"][:, 3:]  # show petal width. X is a petal width list like [[width 1], [width 2], ... ]
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
    # predict_proba() returns a list of probabilities; predict() returns a prediction result.

    decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]  # the first element in X_new list whose proba >=0.5 to target 2
    print('decision boundary: ', decision_boundary)  # decision boundary:  [1.61561562]
    print('predict 1.7, 1.5 as: ', log_reg.predict([[1.7], [1.5]]))  # predict 1.7, 1.5 as:  [1 0]

    plt.figure(figsize=(10, 4))
    # plot(x range, y range, pattern and color): x range could be filtered by y's value as below.
    plt.plot(X[y == 0], y[y == 0], "bs")  # blue box: not target 2. (refer to _axes.py line 1485)
    plt.plot(X[y == 1], y[y == 1], "g^")  # green triangle: target 2

    # decision boundary: the blue and green probability lines cross at the decision boundary.
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
    # plt.show()

    # Decision Boundaries
    # plot a image to show classification of iris based on two features: petal length and petal width

    # train data set with two features
    X = iris["data"][:, (2, 3)]  # petal length, petal width. X is a list like [[length, width], [], [], ...]
    y = (iris["target"] == 2).astype(np.int)  # y is a list with: 1 if Iris-Virginica, else 0

    # re-define a logistic regression model with L2 regularization intense C.
    log_reg = LogisticRegression(C=10 ** 10, random_state=42)  # different with the former log_reg
    # C: Inverse of regularization strength; smaller values specify stronger regularization.
    # So, the regularization intense of logistic model is not the alpha but C, and C is alpha's inverse.
    log_reg.fit(X, y)

    x0, x1 = np.meshgrid(
        np.linspace(2.9, 7, 500).reshape(-1, 1),
        np.linspace(0.8, 2.7, 200).reshape(-1, 1),
    )  # meshgrid: https://www.jb51.net/article/166710.htm  easy to generate grid matrix
    X_new = np.c_[x0.ravel(), x1.ravel()]  # ravel(): flatten a array

    y_proba = log_reg.predict_proba(X_new)

    plt.figure(figsize=(10, 4))
    plt.plot(X[y == 0, 0], X[y == 0, 1], "bs")
    plt.plot(X[y == 1, 0], X[y == 1, 1], "g^")

    zz = y_proba[:, 1].reshape(x0.shape)
    contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)  # contour: https://www.jianshu.com/p/487b211d3c37?from=timeline

    left_right = np.array([2.9, 7])
    print('coef_: ', log_reg.coef_)  # coef_:  [[ 5.7528683  10.44455633]] two features' coefficient
    print('intercept_: ', log_reg.intercept_)  # intercept_:  [-45.26062435] bias
    boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]
    # boundary = -(coef1*x + bias)/coef2

    plt.clabel(contour, inline=1, fontsize=12)
    plt.plot(left_right, boundary, "k--", linewidth=3)
    plt.text(3.5, 1.5, "Not Iris-Virginica", fontsize=14, color="b", ha="center")
    plt.text(6.5, 2.3, "Iris-Virginica", fontsize=14, color="g", ha="center")
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.axis([2.9, 7, 0.8, 2.7])
    save_fig("logistic_regression_contour_plot")
    # plt.show()

    # Soft-max Regression

    # Re-define X and y
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = iris["target"]  # y is not only target==2 any more. It's a multi-classification labels set, so all targets in it

    # define a soft max model with logistic model class
    # multi_class: multi classification     solver: refer to model's doc
    softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10, random_state=42)
    softmax_reg.fit(X, y)

    # make test data X_new
    x0, x1 = np.meshgrid(
        np.linspace(0, 8, 500).reshape(-1, 1),
        np.linspace(0, 3.5, 200).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_proba = softmax_reg.predict_proba(X_new)
    # return all samples' probabilities list like [[target 1 proba, target 2 proba, target 3 proba], [], [], ...]
    y_predict = softmax_reg.predict(X_new)  # return the prediction result list like [0, 1, 1, 2, 2, 1, 0, ...]

    zz1 = y_proba[:, 1].reshape(x0.shape)  # all samples' target 2 probabilities are reshaped into a 1 column array
    zz = y_predict.reshape(x0.shape)

    plt.figure(figsize=(10, 4))

    # plot markers
    plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris-Virginica")
    # (x range: target 2' petal length, y range: target 2's petal width, green triangle)
    plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris-Versicolor")  # blue square
    plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris-Setosa")  # yellow circle

    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])  # get colors: refer to cloud note: matplotlib
    # https://blog.csdn.net/zhaogeng111/article/details/78419015

    plt.contourf(x0, x1, zz, cmap=custom_cmap)  # plot and color the contour field
    contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)  # plot contour line
    plt.clabel(contour, inline=1, fontsize=12)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="center left", fontsize=14)
    plt.axis([0, 7, 0, 3.5])
    save_fig("softmax_regression_contour_plot")
    plt.show()