# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LinearRegression


# to make this notebook's output stable across runs
np.random.seed(42)

# Ignore useless warnings (see SciPy issue #5998)
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

# To plot pretty figures
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

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

    # Linear regression using the Normal Equation

    # prepare data
    X = 2 * np.random.rand(100, 1)  # random.rand(100, 1) returns a "100 row 1 column" values in [0, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)  # randn(100, 1) returns a "100 row 1 column" values in normal distribution
    # X is getting bigger then y is getting bigger too.

    plt.plot(X, y, "b.")  # blue dot
    plt.xlabel("$x_1$", fontsize=18)  # x_1: X with 1 as its subscript;  $: shown in latex mathematical formula
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([0, 2, 0, 15])
    save_fig("generated_data_plot")
    # plt.show()

    # compute best theta(lin_reg.intercept_, lin_reg.coef_ refer to chapter2)
    X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance for theta 0;   ones: a 100 row 1 column of 1 array
    # np.ones((2, 1)) output: array([[1.], [1.]]);
    # X_b = [[1, x0], [1, x1], [1, x2] ... [1, x99]]T
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)  # .T: T transform
    # X_b.T.dot(X_b) == dot.(X_b.T, X_b);  dot: dot product. eg. a·b=a1b1+a2b2+……+anbn
    # theta_best = inv(X_b.T·X_b)·X_b.T·y  # normal equation
    # print(theta_best)  # output: [[4.21509616], [2.77011339]]. actually it's lin_reg.intercept_, lin_reg.coef_

    # make predict with best theta
    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
    y_predict = X_new_b.dot(theta_best)  # make prediction with theta_best
    # print(y_predict)  # output: [[4.21509616] [9.75532293]]

    plt.plot(X_new, y_predict, "r-", linewidth=2, label="Predictions")
    plt.plot(X, y, "b.")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.legend(loc="upper left", fontsize=14)
    plt.axis([0, 2, 0, 15])
    save_fig("linear_model_predictions")
    # plt.show()  # X_new and y_predict is red line almost in the middle of these blue dots

    # the process of computing best theta and make prediction could be replaced by lin_reg as below:
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)  # no need to add ones array to X array
    # print(lin_reg.intercept_, lin_reg.coef_)  # output: [4.21509616] [[2.77011339]]  coef_ refer to chapter2
    y_predict = lin_reg.predict(X=X_new)
    # print(y_predict)  # output: [[4.21509616] [9.75532293]]  the same result as above y_predict

    theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)  # note: X_b input, not X
    # theta_best_svd: lin_reg.intercept_, lin_reg.coef_. the meaning of these refers to chapter2
    # residuals:
    # rank: rank of X_b matrix input
    # s:
    # value less than rcond will be 0;
    # print(theta_best_svd, residuals, rank, s)
    # output: [[4.21509616] [2.77011339]],  [80.6584564],  2,  [14.37020392  4.11961067]

    theta_best_svd = np.linalg.pinv(X_b).dot(y)  # pinv(X_b) == inv(X_b.T.dot(X_b)).dot(X_b.T)
    # print(theta_best_svd)  # output: [[4.21509616] [2.77011339]]

    # Batch Gradient Descent
    eta = 0.1  # learning rate η
    n_iterations = 1000
    m = 100
    theta = np.random.randn(2, 1)  # theta θ = a random two row one column vector
    for iteration in range(n_iterations):
        gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)  #
        theta = theta - eta * gradients
    print(theta)