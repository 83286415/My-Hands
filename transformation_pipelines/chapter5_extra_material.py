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
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier

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


class MyLinearSVC(BaseEstimator):
    def __init__(self, C=1, eta0=1, eta_d=10000, n_epochs=1000, random_state=None):
        self.C = C  # penalty
        self.eta0 = eta0  # eta beginning value
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.eta_d = eta_d  # used to decrease eta

    def eta(self, epoch):
        return self.eta0 / (epoch + self.eta_d)  # update eta by reducing its value

    def fit(self, X, y):
        # Random initialization
        if self.random_state:
            np.random.seed(self.random_state)
        w = np.random.randn(X.shape[1], 1)  # n feature weights
        b = 0

        m = len(X)
        t = y * 2 - 1  # -1 if t==0, +1 if t==1
        X_t = X * t
        self.Js = []

        # Training
        for epoch in range(self.n_epochs):
            support_vectors_idx = (X_t.dot(w) + t * b < 1).ravel()
            X_t_sv = X_t[support_vectors_idx]
            t_sv = t[support_vectors_idx]

            J = 1 / 2 * np.sum(w * w) + self.C * (np.sum(1 - X_t_sv.dot(w)) - b * np.sum(t_sv))
            self.Js.append(J)  # training object

            w_gradient_vector = w - self.C * np.sum(X_t_sv, axis=0).reshape(-1, 1)
            b_derivative = -C * np.sum(t_sv)

            w = w - self.eta(epoch) * w_gradient_vector
            b = b - self.eta(epoch) * b_derivative

        self.intercept_ = np.array([b])
        self.coef_ = np.array([w])
        support_vectors_idx = (X_t.dot(w) + t * b < 1).ravel()
        self.support_vectors_ = X[support_vectors_idx]
        return self

    def decision_function(self, X):
        return X.dot(self.coef_[0]) + self.intercept_[0]

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(np.float64)


if __name__ == '__main__':

    # Training time

    # plot 1000 moons data points
    X, y = make_moons(n_samples=1000, noise=0.4, random_state=42)
    plot_dataset(X, y, [-2.5, 4, -2, 3])
    # plt.show()

    # plot training time
    tol = 0.1  # the bias tolerance to stop training
    tols = []
    times = []  # training time list
    for i in range(10):
        svm_clf = SVC(kernel="poly", gamma=3, C=10, tol=tol, verbose=1)
        t1 = time.time()
        svm_clf.fit(X, y)
        t2 = time.time()
        times.append(t2 - t1)
        tols.append(tol)
        print(i, tol, t2 - t1)
        tol /= 10  # reduce tol
    plt.figure(figsize=(8, 5))
    plt.semilogx(tols, times)  # the same as plot but x axis is the logarithm of the original x values
    # plt.show()

    # Linear SVM classifier implementation using Batch Gradient Descent

    # Training set
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = (iris["target"] == 2).astype(np.float64).reshape(-1, 1)  # Iris-Virginica

    # Linear SVC by BGD
    C = 2
    svm_clf = MyLinearSVC(C=C, eta0=10, eta_d=1000, n_epochs=60000, random_state=2)
    svm_clf.fit(X, y)
    svm_clf.predict(np.array([[5, 2], [4, 1]]))  # array([[1.], [0.]]) that is True and False

    # plot epochs curve
    plt.figure(figsize=(10, 8))
    plt.plot(range(svm_clf.n_epochs), svm_clf.Js)  # x axis: each epoch     y: training object
    plt.axis([0, svm_clf.n_epochs, 0, 100])

    print(svm_clf.intercept_, svm_clf.coef_)  # [-15.56780998] [[[2.28129013] [2.71597487]]]

    # SVC() used as a comparison
    svm_clf2 = SVC(kernel="linear", C=C)
    svm_clf2.fit(X, y.ravel())
    print(svm_clf2.intercept_, svm_clf2.coef_)  # [-15.51721253] [[2.27128546 2.71287145]]

    # SGD used as another comparison
    sgd_clf = SGDClassifier(loss="hinge", alpha=0.017, max_iter=50, random_state=42)
    sgd_clf.fit(X, y.ravel())
    print(sgd_clf.intercept_, sgd_clf.coef_)  # [-14.062485] [[2.24179316 1.79750198]]

    m = len(X)
    t = y * 2 - 1  # -1 if t==0, +1 if t==1
    X_b = np.c_[np.ones((m, 1)), X]  # Add bias input x0=1
    X_b_t = X_b * t
    sgd_theta = np.r_[sgd_clf.intercept_[0], sgd_clf.coef_[0]]
    print(sgd_theta)  # [-14.062485     2.24179316   1.79750198]
    support_vectors_idx = (X_b_t.dot(sgd_theta) < 1).ravel()
    sgd_clf.support_vectors_ = X[support_vectors_idx]  # need this to plot
    sgd_clf.C = C

    # plot my LinearSVC and SVC decision boundary images
    # my LinearSVC()
    yr = y.ravel()
    plt.figure(figsize=(18, 5))
    plt.subplot(131)
    plt.plot(X[:, 0][yr == 1], X[:, 1][yr == 1], "g^", label="Iris-Virginica")
    plt.plot(X[:, 0][yr == 0], X[:, 1][yr == 0], "bs", label="Not Iris-Virginica")
    plot_svc_decision_boundary(svm_clf, 4, 6)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.title("MyLinearSVC", fontsize=14)
    plt.axis([4, 6, 0.8, 2.8])

    # SVC()
    plt.subplot(132)
    plt.plot(X[:, 0][yr == 1], X[:, 1][yr == 1], "g^")
    plt.plot(X[:, 0][yr == 0], X[:, 1][yr == 0], "bs")
    plot_svc_decision_boundary(svm_clf2, 4, 6)
    plt.xlabel("Petal length", fontsize=14)
    plt.title("SVC", fontsize=14)
    plt.axis([4, 6, 0.8, 2.8])

    # SGD()
    plt.subplot(133)
    plt.plot(X[:, 0][yr == 1], X[:, 1][yr == 1], "g^")
    plt.plot(X[:, 0][yr == 0], X[:, 1][yr == 0], "bs")
    plot_svc_decision_boundary(sgd_clf, 4, 6)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.title("SGDClassifier", fontsize=14)
    plt.axis([4, 6, 0.8, 2.8])
    plt.show()