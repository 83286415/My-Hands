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


def plot_gradient_descent(theta, eta, theta_path=None):
        m = len(X_b)  # count of samples in X_b
        plt.plot(X, y, "b.")
        n_iterations = 1000
        for iteration in range(n_iterations):
            if iteration < 10:
                y_predict = X_new_b.dot(theta)
                style = "b-" if iteration > 0 else "r--"
                plt.plot(X_new, y_predict, style)  # plot the linear model
            gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)  # gradient vector of cost in gradient descent
            theta = theta - eta * gradients  # eta:learning rate    theta: cost vector
            if theta_path is not None:
                theta_path.append(theta)
        plt.xlabel("$x_1$", fontsize=18)
        plt.axis([0, 2, 0, 15])
        plt.title(r"$\eta = {}$".format(eta), fontsize=16)


def learning_schedule(t):  # simulated annealing: reduce eta in each iteration
        return t0 / (t + t1)


if __name__ == '__main__':

    # Linear regression:

    # 1. Normal Equation
    print('\nNormal Equation')

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

    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)  # theta formula refer to cloud note: normal equation
    # np.lialg: numpy's Linear Algebra
    # inv: inverse matrix
    # .T: T transform
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
    print('LinearRegression: ', lin_reg.intercept_, lin_reg.coef_)
    # output: [4.21509616] [[2.77011339]]  coef_ refer to chapter2 and below:
    # y = ax + b : intercept_ is b and coef_ is a as well as the theta.
    # y = 3x + 4 according to the output of lig_reg and now this result is also almost the same as y's formula at data
    # prepare line 44: y = 4 + 3 * X + np.random.randn(100, 1)

    y_predict = lin_reg.predict(X_new)
    # print(y_predict)  # output: [[4.21509616] [9.75532293]]  the same result as above y_predict

    # other methods to get theta:
    theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)  # lstsq: refer to cloud  normal equation
    # note: X_b input, not X
    # theta_best_svd: lin_reg.intercept_, lin_reg.coef_. the meaning of these refers to chapter2
    # residuals:
    # rank: rank of X_b matrix input. that is ju zhen de zhi.
    # s: Singular values of `a`
    # value less than rcond will be 0;
    print('np.linalg.lstsq: ', theta_best_svd, residuals, rank, s)
    # output: [[4.21509616] [2.77011339]],  [80.6584564],  2,  [14.37020392  4.11961067]

    theta_best_svd = np.linalg.pinv(X_b).dot(y)  # pinv(X_b) == inv(X_b.T.dot(X_b)).dot(X_b.T) That is AXA=A, XAX=X.
    print('np.linalg.pinv: ', theta_best_svd)  # output: [[4.21509616] [2.77011339]]

    # 2. Batch Gradient Descent
    print('\nBatch Gradient Descent')

    eta = 0.1  # learning rate η
    n_iterations = 1000
    m = 100  # the count of the data in data set. We set 100 rows in X, so here we set m = 100
    theta = np.random.randn(2, 1)  # theta θ = a random two row one column vector as a random start of gradient descent
    for iteration in range(n_iterations):
        gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)  # gradient vector of the cost
        theta = theta - eta * gradients  # eta: learning rate η
    print('Batch Gradient Descent in iterations: ', theta)  # [[4.21509616], [2.77011339]]
    # how to set a better eta and iterations value: refer to book P116 and cloud note Batch Gradient Descent

    theta_path_bgd = []  # theta list in all iterations
    np.random.seed(42)
    theta = np.random.randn(2, 1)  # random initialization

    plt.figure(figsize=(10, 4))
    plt.subplot(131)
    plot_gradient_descent(theta, eta=0.02)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.subplot(132)
    plot_gradient_descent(theta, eta=0.1, theta_path=theta_path_bgd)
    plt.subplot(133)
    plot_gradient_descent(theta, eta=0.5)

    save_fig("gradient_descent_plot")
    # plt.show()  # comment it out to let code continue

    # 3. Stochastic Gradient Descent - SGD
    print('\nStochastic Gradient Descent')

    theta_path_sgd = []  # theta list in all iterations
    m = len(X_b)  # count of samples in X_b, m = 100
    np.random.seed(42)

    n_epochs = 50
    t0, t1 = 5, 50  # learning schedule hyper-parameters

    theta = np.random.randn(2, 1)  # random initialization
    plt.figure(figsize=(8, 8))

    for epoch in range(n_epochs):  # 50 iterations to compute the theta value
        for i in range(m):    # loops in all samples in data set
            if epoch == 0 and i < 20:
                y_predict = X_new_b.dot(theta)
                style = "b-" if i > 0 else "r--"  # red line: the first data sample; blue line: else
                plt.plot(X_new, y_predict, style)  # plot the linear model of the first 20 vectors of (X_new, y_predict)
            random_index = np.random.randint(m)
            xi = X_b[random_index:random_index + 1]
            yi = y[random_index:random_index + 1]  # pick up a random sample to compute theta
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            eta = learning_schedule(epoch * m + i)  # reduce eta for each sample
            theta = theta - eta * gradients
            theta_path_sgd.append(theta)

    plt.plot(X, y, "b.")  # blue point
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([0, 2, 0, 15])
    save_fig("sgd_plot")
    # plt.show()

    print('Stochastic Gradient Descent in iterations: ', theta)
    # the last theta: [[4.21076011], [2.74856079]] is close to the min value[[4.21509616] [2.77011339]] but not it.

    from sklearn.linear_model import SGDRegressor  # SGD: Stochastic Gradient Descent

    sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1, random_state=42)
    sgd_reg.fit(X, y.ravel())  # ravel: https://blog.csdn.net/lanchunhui/article/details/50354978
    # SGDRegressor(alpha=0.0001, average=False, epsilon=0.1, eta0=0.1,
    #              fit_intercept=True, l1_ratio=0.15, learning_rate='invscaling',
    #              loss='squared_loss', max_iter=50, n_iter=None, penalty=None,
    #              power_t=0.25, random_state=42, shuffle=True, tol=None, verbose=0,
    #              warm_start=False)

    print('SGDRegressor: ', sgd_reg.intercept_, sgd_reg.coef_)  # [4.16782089] [2.72603052] also close but not the theta

