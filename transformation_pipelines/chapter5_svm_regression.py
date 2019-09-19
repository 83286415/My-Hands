# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR
'''
    Linear Support Vector Regression(LinearSVR)

    Similar to SVR with parameter kernel='linear', but implemented in terms of
    liblinear rather than libsvm, so it has more flexibility in the choice of
    penalties and loss functions and should scale better to large numbers of
    samples.
    This class supports both dense and sparse input.
'''
from sklearn.svm import SVR
'''
    Epsilon-Support Vector Regression(SVR)

    The free parameters in the model are C and epsilon. Kernel tricks in.
    The implementation is based on libsvm.
'''

# Where to save the figures
WORKING_PATH = os.path.abspath(os.path.join(os.getcwd(), '..'))
ROOT_PATH = os.path.join(WORKING_PATH, 'Hands on SK and TS\\')
CHAPTER_ID = "support_vector_machines"

# to make this notebook's output stable across runs
np.random.seed(42)


def find_support_vectors(svm_reg, X, y):
    y_pred = svm_reg.predict(X)
    off_margin = (np.abs(y - y_pred) >= svm_reg.epsilon)  # off_margin is a list filled with bool like [True, False]
    return np.argwhere(off_margin)  # np.argwhere: returns a list of index of non-zero (not False) value


def plot_svm_regression(svm_reg, X, y, axes):
    x1s = np.linspace(axes[0], axes[1], 100).reshape(100, 1)  # test data
    y_pred = svm_reg.predict(x1s)
    plt.plot(x1s, y_pred, "k-", linewidth=2, label=r"$\hat{y}$")  # plot the predicted line
    plt.plot(x1s, y_pred + svm_reg.epsilon, "k--")  # higher the street dot line
    plt.plot(x1s, y_pred - svm_reg.epsilon, "k--")  # lower it
    plt.scatter(X[svm_reg.support_], y[svm_reg.support_], s=180, facecolors='#FFAAAA')  # circle the points outside
    # s: the size(thick or thin) of the circle on the points outside street
    plt.plot(X, y, "bo")  # blue circle
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.legend(loc="upper left", fontsize=18)
    plt.axis(axes)


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(ROOT_PATH, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)  # cannot save file if path doesn't exist


if __name__ == '__main__':

    # SVM Regression with LinearSVR(epsilon): faster but no kernel tricks

    # prepare data set
    m = 50  # sample count
    X = 2 * np.random.rand(m, 1)
    y = (4 + 3 * X + np.random.randn(m, 1)).ravel()

    # define a LinearSVR regress-er instance
    svm_reg = LinearSVR(epsilon=1.5, random_state=42)  # epsilon decides width of a street: bigger epsilon, wider street
    '''  LinearSVR(C=1.0, dual=True, epsilon=1.5, fit_intercept=True,
         intercept_scaling=1.0, loss='epsilon_insensitive', max_iter=1000,
         random_state=42, tol=0.0001, verbose=0)
    '''  # C: penalty.
    svm_reg.fit(X, y)

    # compare different epsilons in LinearSVR(): bigger epsilon, wider street
    svm_reg1 = LinearSVR(epsilon=1.5, random_state=42)
    svm_reg2 = LinearSVR(epsilon=0.5, random_state=42)
    svm_reg1.fit(X, y)
    svm_reg2.fit(X, y)

    # support_: points outside the street
    svm_reg1.support_ = find_support_vectors(svm_reg1, X, y)  # [[ 7], [14], [25], ... ]
    svm_reg2.support_ = find_support_vectors(svm_reg2, X, y)
    # support_ is not a attribute of LinearSVR. It's added here.

    # a sample of reg1: a green point on the predicted line and used to points the epsilon
    eps_x1 = 1
    eps_y_pred = svm_reg1.predict([[eps_x1]])

    # plot
    plt.figure(figsize=(9, 4))
    plt.subplot(121)
    plot_svm_regression(svm_reg1, X, y, [0, 2, 3, 11])  # support_ has been defined above
    plt.title(r"$\epsilon = {}$".format(svm_reg1.epsilon), fontsize=18)
    plt.ylabel(r"$y$", fontsize=18, rotation=0)
    # plt.plot([eps_x1, eps_x1], [eps_y_pred, eps_y_pred - svm_reg1.epsilon], "k-", linewidth=2)
    plt.plot([eps_x1, eps_x1], [eps_y_pred, eps_y_pred], "g^")
    plt.annotate(
        '', xy=(eps_x1, eps_y_pred), xycoords='data',
        xytext=(eps_x1, eps_y_pred - svm_reg1.epsilon),
        textcoords='data', arrowprops={'arrowstyle': '<->', 'linewidth': 1.5}
    )  # no text added into image but bio-direction arrow added
    plt.text(0.91, 5.6, r"$\epsilon$", fontsize=20)
    plt.subplot(122)
    plot_svm_regression(svm_reg2, X, y, [0, 2, 3, 11])
    plt.title(r"$\epsilon = {}$".format(svm_reg2.epsilon), fontsize=18)
    save_fig("svm_regression_plot")
    # plt.show()

    # SVM Regression with SVR(kernel, degree, C, epsilon)

    # prepare data set
    m = 100  # sample count
    X = 2 * np.random.rand(m, 1) - 1
    y = (0.2 + 0.1 * X + 0.5 * X ** 2 + np.random.randn(m, 1) / 10).ravel()

    # define a SVR regress-er instance: kernel tricks make a 2 degrees curves instead of LinearSVR straight line
    svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
    '''
        SVR(C=100, cache_size=200, coef0=0.0, degree=2, epsilon=0.1, gamma='auto',
        kernel='poly', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    '''  # C: penalty  # epsilon: width of street
    svm_poly_reg.fit(X, y)

    # different penalties C: smaller C, more flatten curve
    svm_poly_reg1 = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
    svm_poly_reg2 = SVR(kernel="poly", degree=2, C=0.01, epsilon=0.1)
    svm_poly_reg1.fit(X, y)
    svm_poly_reg2.fit(X, y)

    # plot
    plt.figure(figsize=(9, 4))
    plt.subplot(121)
    plot_svm_regression(svm_poly_reg1, X, y, [-1, 1, 0, 1])
    plt.title(r"$degree={}, C={}, \epsilon = {}$".format(svm_poly_reg1.degree, svm_poly_reg1.C, svm_poly_reg1.epsilon),
              fontsize=18)  # SVR attributes supported
    plt.ylabel(r"$y$", fontsize=18, rotation=0)
    plt.subplot(122)
    plot_svm_regression(svm_poly_reg2, X, y, [-1, 1, 0, 1])
    plt.title(r"$degree={}, C={}, \epsilon = {}$".format(svm_poly_reg2.degree, svm_poly_reg2.C, svm_poly_reg2.epsilon),
              fontsize=18)
    save_fig("svm_with_polynomial_kernel_plot")
    plt.show()