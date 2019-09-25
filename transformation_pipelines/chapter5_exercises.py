# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import time
import numpy as np
import os

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_mldata
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error

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
    plt.savefig(path, format='png', dpi=300)  # cannot save file if path doesn't exist


def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)


def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]  # svm_clf.coef_ [[1.29411744 0.82352928]] two features so two columns: length, width
    b = svm_clf.intercept_[0]

    # At the decision boundary: refer to cloud note's Decision Boundary
    #   w0*x0 + w1*x1 + b = 0 (w0: theta0, w1: theta1, b: theta2)
    #   theta: [theta0, theta1, theta2]
    #   w0: coef_[0][0],  w1: coef_[0][1],  b: intercept_[0]
    # => x1 = -w0/w1 * x0 - b/w1,  and x1 is the decision_boundary, so the formula is as below:
    x0 = np.linspace(xmin, xmax, 200)  # x axis range, that is petal length range
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]  # also refer to cloud note chapter 5 soft margin - decision boundary
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_  # return a array of two support vectors (two points)
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')  # plot the support vectors circled with pink
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)


if __name__ == '__main__':

    # Exercise 8 P165

    # data set
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = iris["target"]

    setosa_or_versicolor = (y == 0) | (y == 1)  # not y == 2 that is Iris-Virginica
    X = X[setosa_or_versicolor]
    y = y[setosa_or_versicolor]

    # build models
    C = 5
    alpha = 1 / (C * len(X))  # used to compute eta in SGD

    lin_clf = LinearSVC(loss="hinge", C=C, random_state=42)
    svm_clf = SVC(kernel="linear", C=C)
    sgd_clf = SGDClassifier(loss="hinge", learning_rate="constant", eta0=0.001, alpha=alpha,
                            max_iter=100000, random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lin_clf.fit(X_scaled, y)
    svm_clf.fit(X_scaled, y)
    sgd_clf.fit(X_scaled, y)

    print("LinearSVC:                   ", lin_clf.intercept_, lin_clf.coef_)  # [0.28481447] [[1.05541976 1.09851597]]
    print("SVC:                         ", svm_clf.intercept_, svm_clf.coef_)  # [0.31933577] [[1.1223101  1.02531081]]
    print("SGDClassifier(alpha={:.5f}):".format(sgd_clf.alpha), sgd_clf.intercept_, sgd_clf.coef_)
    # SGDClassifier(alpha=0.00200): [0.32] [[1.12293103 1.02620763]]

    # plot
    # Compute the slope and bias of each decision boundary
    w1 = -lin_clf.coef_[0, 0] / lin_clf.coef_[0, 1]
    b1 = -lin_clf.intercept_[0] / lin_clf.coef_[0, 1]
    w2 = -svm_clf.coef_[0, 0] / svm_clf.coef_[0, 1]
    b2 = -svm_clf.intercept_[0] / svm_clf.coef_[0, 1]
    w3 = -sgd_clf.coef_[0, 0] / sgd_clf.coef_[0, 1]
    b3 = -sgd_clf.intercept_[0] / sgd_clf.coef_[0, 1]

    # Transform the decision boundary lines back to the original scale and plot the decision boundary line like these:
    # x0 = np.linspace(xmin, xmax, 200)
    # decision_boundary = -w[0]/w[1] * x0 - b/w[1]
    # plt.plot(x0, decision_boundary, "k-", linewidth=2)
    line1 = scaler.inverse_transform([[-10, -10 * w1 + b1], [10, 10 * w1 + b1]])  # w1: -w[0]/w[1]   b1: -b/w[1]
    line2 = scaler.inverse_transform([[-10, -10 * w2 + b2], [10, 10 * w2 + b2]])  # x0: [-10, 10]
    line3 = scaler.inverse_transform([[-10, -10 * w3 + b3], [10, 10 * w3 + b3]])

    # Plot all three decision boundaries
    plt.figure(figsize=(11, 4))
    plt.plot(line1[:, 0], line1[:, 1], "k:", label="LinearSVC") # plot street center line (decision boundary line)
    plt.plot(line2[:, 0], line2[:, 1], "b--", linewidth=2, label="SVC")
    plt.plot(line3[:, 0], line3[:, 1], "r-", label="SGDClassifier")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs")  # label="Iris-Versicolor": blue square
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo")  # label="Iris-Setosa": yellow circle
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="upper center", fontsize=14)
    plt.axis([0, 5.5, 0, 2])
    # plt.show()

    # Exercise 9 P165

    # data set
    mnist = fetch_mldata("MNIST original")  # MNIST is a one vs rest problem, not a one vs one problem
    X = mnist["data"]
    y = mnist["target"]

    X_train = X[:60000]
    y_train = y[:60000]
    X_test = X[60000:]
    y_test = y[60000:]

    np.random.seed(42)
    rnd_idx = np.random.permutation(60000)  # re-shuffle
    X_train = X_train[rnd_idx]
    y_train = y_train[rnd_idx]

    # build model
    lin_clf = LinearSVC(random_state=42)

    scaler = StandardScaler()  # scale
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
    X_test_scaled = scaler.transform(X_test.astype(np.float32))

    # lin_clf.fit(X_train_scaled, y_train)  # comment it out to save time
    # y_pred = lin_clf.predict(X_train_scaled)
    # print('LinearSVC accuracy score: ', accuracy_score(y_train, y_pred))  # 0.9204 but not good enough

    # SVC model to deal with One vs Rest
    svm_clf = SVC(decision_function_shape="ovr")  # default is OvO, so set it OvR: one vs rest. MNIST is a OvR problem
    svm_clf.fit(X_train_scaled[:10000], y_train[:10000])
    y_pred = svm_clf.predict(X_train_scaled)
    print('SVC in OvR model accuracy score: ', accuracy_score(y_train, y_pred))  # 0.94615

    # Non-linear classification: Adding Similarity Features with RBF kernel skill to improve
    # grid search the best parameters: refer to chapter2.py
    param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}  # C:penalty  gamma:Kernel coefficient
    # reciprocal: A reciprocal continuous random variable.        uniform: A uniform continuous random variable
    rnd_search_cv = RandomizedSearchCV(svm_clf, param_distributions, n_iter=10, verbose=2)
    rnd_search_cv.fit(X_train_scaled[:1000], y_train[:1000])

    # choose the best estimator
    print('best estimator: ', rnd_search_cv.best_estimator_)
    print('best score: ', rnd_search_cv.best_score_)  # best score:  0.856
    rnd_search_cv.best_estimator_.fit(X_train_scaled, y_train)
    '''best estimator: 
    SVC(C=8.852316058423087, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=0.001766074650481071,
      kernel='rbf', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)
    '''

    # prediction and print accuracy
    y_pred = rnd_search_cv.best_estimator_.predict(X_train_scaled)
    print('best estimator accuracy score: ', accuracy_score(y_train, y_pred))  # 0.99965

    # test the model
    y_pred = rnd_search_cv.best_estimator_.predict(X_test_scaled)
    print('best estimator test accuracy score: ', accuracy_score(y_test, y_pred))  # 0.9709

    # Exercise 10 P166

    # data set
    housing = fetch_california_housing()
    X = housing["data"]
    y = housing["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # build model
    lin_svr = LinearSVR(random_state=42)
    lin_svr.fit(X_train_scaled, y_train)

    y_pred = lin_svr.predict(X_train_scaled)
    mse = mean_squared_error(y_train, y_pred)
    print('LinearSVR MSE: ', mse)  # 0.949968822217229 not good
    print('LinearSVR RMSE: ', np.sqrt(mse))

    # grid search the best estimator with SVR() model which can use kernel skill
    param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
    rnd_search_cv = RandomizedSearchCV(SVR(), param_distributions, n_iter=10, verbose=2, random_state=42)
    rnd_search_cv.fit(X_train_scaled, y_train)

    print('best estimator: ', rnd_search_cv.best_estimator_)
    '''SVR(C=4.745401188473625, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
      gamma=0.07969454818643928, kernel='rbf', max_iter=-1, shrinking=True,
      tol=0.001, verbose=False)
    '''

    y_pred = rnd_search_cv.best_estimator_.predict(X_train_scaled)
    mse = mean_squared_error(y_train, y_pred)
    print('best estimator SVR MSE: ', mse)
    print('best estimator SVR RMSE: ', np.sqrt(mse))  # 0.5727524770785356

    # test
    y_pred = rnd_search_cv.best_estimator_.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print('best estimator SVR test MSE: ', mse)
    print('best estimator SVR test RMSE: ', np.sqrt(mse))  # 0.592916838552874
