# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


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


def plot_model(model_class, polynomial, alphas, **model_kargs):
    for alpha, style in zip(alphas, ("b-", "g--", "r:")):
        model = model_class(alpha, **model_kargs) if alpha > 0 else LinearRegression()
        if polynomial:
            model = Pipeline([
                    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),  # 10 degree polynomial
                    ("std_scaler", StandardScaler()),
                    ("regul_reg", model),
                ])  # StandardScaler() must be between PolynomialFeatures() and Ridge()
        model.fit(X, y)
        y_new_regul = model.predict(X_new)
        lw = 2 if alpha > 0 else 1
        plt.plot(X_new, y_new_regul, style, linewidth=lw, label=r"$\alpha = {}$".format(alpha))
    plt.plot(X, y, "b.", linewidth=3)
    plt.legend(loc="upper left", fontsize=15)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 3, 0, 4])


if __name__ == '__main__':

    # Ridge Regression

    np.random.seed(42)
    m = 20
    X = 3 * np.random.rand(m, 1)  # 20 row 1 column, each data in (0, 3)
    y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5  # linear model
    X_new = np.linspace(0, 3, 100).reshape(100, 1)  # 100 points in (0, 3) with same interval

    plt.figure(figsize=(8,4))
    plt.subplot(121)  # add the first image in a frame with 1 row 2 columns. the last 1 is the first image.
    plot_model(Ridge, polynomial=False, alphas=(0, 10, 100), random_state=42)  # ridge model
    # different alphas make different prediction curves. refer to cloud note.
    # The bigger alpha is, the curve is more flatten

    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.subplot(122)  # add the second image
    plot_model(Ridge, polynomial=True, alphas=(0, 10**-5, 1), random_state=42)

    save_fig("ridge_regression_plot")
    # plt.show()

    ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
    #           cholesky:  uses the standard scipy.linalg.solve function to obtain a closed-form solution.
    ridge_reg.fit(X, y)
    print(ridge_reg.predict([[1.5]]))  # [[1.55071465]]
    print('RidgeRegression_cholesky: ', ridge_reg.intercept_, ridge_reg.coef_)  # [1.00650911] [[0.36280369]]

    sgd_reg = SGDRegressor(max_iter=5, penalty="l2", random_state=42)  # penalty L2 refer to cloud note "L2"
    sgd_reg.fit(X, y.ravel())
    print(sgd_reg.predict([[1.5]]))  # [1.13500145]
    print('SGD Regressor with L2 penalty: ', sgd_reg.intercept_, sgd_reg.coef_)  # [0.35165674] [0.52222981]

    ridge_reg = Ridge(alpha=1, solver="sag", random_state=42)  # Stochastic Average Gradient descent
    #           sag: uses a Stochastic Average Gradient descent, and 'saga' uses
    #           its improved, unbiased version named SAGA. Both methods also use an
    #           iterative procedure, and are often faster than other solvers when
    #           both n_samples and n_features are large. Note that 'sag' and
    #           'saga' fast convergence is only guaranteed on features with
    #           approximately the same scale. You can preprocess the data with a
    #           scaler from sklearn.preprocessing.
    ridge_reg.fit(X, y)
    print(ridge_reg.predict([[1.5]]))  # [[1.5507201]]
    print('RidgeRegression_sag: ', ridge_reg.intercept_, ridge_reg.coef_)  # [1.00645006] [[0.3628467]]

    # Lasso Regression

    # Compared with Ridge Regression' plot above. Refer to cloud note.
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plot_model(Lasso, polynomial=False, alphas=(0, 0.1, 1), random_state=42)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.subplot(122)
    plot_model(Lasso, polynomial=True, alphas=(0, 10 ** -7, 1), tol=1, random_state=42)  # tol: tolerance,refer to Lasso

    save_fig("lasso_regression_plot")
    # plt.show()

    lasso_reg = Lasso(alpha=0.1)  # alpha: regularization intensity
    lasso_reg.fit(X, y)
    print('Lasso Regression', lasso_reg.predict([[1.5]]))  # Lasso Regression [1.53788174]

    sgd_reg = SGDRegressor(max_iter=5, penalty="l1", random_state=42)  # penalty L1 refer to cloud note "L1"
    sgd_reg.fit(X, y.ravel())
    print(sgd_reg.predict([[1.5]]))  # [1.13498188]
    print('SGD Regressor with L1 penalty: ', sgd_reg.intercept_, sgd_reg.coef_)  # [0.35166208] [0.5222132]

    # Elastic Net
    # How to choose Ridge regression, Lasso regression or Elastic Net? refer to cloud note in Elastic Net or book P132.

    # Elastic Net: combine Ridge with Lasso regression with a ratio r in loss.
    elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)  # l1_ratio refer to cloud note: ElasticNet's r.
    elastic_net.fit(X, y)
    print('Elastic Net: ', elastic_net.predict([[1.5]]))  # [1.54333232]

    # Early Stopping

    np.random.seed(42)
    m = 100  # count of samples
    X = 6 * np.random.rand(m, 1) - 3  # X in (-3, 3)
    y = 2 + X + 0.5 * X ** 2 + np.random.randn(m, 1)

    X_train, X_val, y_train, y_val = train_test_split(X[:50], y[:50].ravel(), test_size=0.5, random_state=10)  # half

    poly_scaler = Pipeline([
        ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),  # high degree polynomial
        ("std_scaler", StandardScaler()),
    ])  # PolynomialFeatures + StandardScaler

    # fit, fit_transform, transform refer to cloud note Chapter 2 SKlearn and its pipeline
    X_train_poly_scaled = poly_scaler.fit_transform(X_train)  # fit and transform on train data set with pipeline
    X_val_poly_scaled = poly_scaler.transform(X_val)    # transform on validation data set with pipeline

    sgd_reg = SGDRegressor(max_iter=1,
                           penalty=None,
                           eta0=0.0005,
                           warm_start=True,
                           learning_rate="constant",
                           random_state=42)  # SDG

    n_epochs = 500
    train_errors, val_errors = [], []
    for epoch in range(n_epochs):
        sgd_reg.fit(X_train_poly_scaled, y_train)
        y_train_predict = sgd_reg.predict(X_train_poly_scaled)
        y_val_predict = sgd_reg.predict(X_val_poly_scaled)
        train_errors.append(mean_squared_error(y_train, y_train_predict))  # MSE
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    best_epoch = int(np.argmin(val_errors))  # argmin: returns the indices of the minimum values along an axis.
    best_val_rmse = np.sqrt(val_errors[best_epoch])  # find the lowest RMSE

    plt.figure(figsize=(8, 4))
    # annotate: add annotation(comments) in the image.
    plt.annotate('Best model',
                 xy=(best_epoch, best_val_rmse),
                 xytext=(best_epoch, best_val_rmse + 1),
                 ha="center",
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=16,
                 )  # https://blog.csdn.net/you_are_my_dream/article/details/53454549

    best_val_rmse -= 0.03  # just to make the graph look better
    plt.plot([0, n_epochs], [best_val_rmse, best_val_rmse], "k:", linewidth=2)
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation set")  # RMSE
    plt.plot(np.sqrt(train_errors), "r--", linewidth=2, label="Training set")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)
    save_fig("early_stopping_plot")
    plt.show()

    # re-define a SGD model to reproduce "Early Stopping"
    sgd_reg = SGDRegressor(max_iter=1, warm_start=True, penalty=None,
                           learning_rate="constant", eta0=0.0005, random_state=42)
    # wart_start: continue mode, not restart fit from the beginning if interrupted

    minimum_val_error = float("inf")  # infinity, float("inf") is the biggest float and float("-inf") is the smallest
    best_epoch = None
    best_model = None
    for epoch in range(1000):
        sgd_reg.fit(X_train_poly_scaled, y_train)  # continues where it left off for warm_start=True
        y_val_predict = sgd_reg.predict(X_val_poly_scaled)
        val_error = mean_squared_error(y_val, y_val_predict)  # MSE
        if val_error < minimum_val_error:
            minimum_val_error = val_error
            best_epoch = epoch
            best_model = clone(sgd_reg)
    print('Early Stopping best epoch and model: ', best_epoch, best_model)
    # best epoch 239
    # SGDRegressor(alpha=0.0001, average=False, epsilon=0.1, eta0=0.0005,
    #        fit_intercept=True, l1_ratio=0.15, learning_rate='constant',
    #        loss='squared_loss', max_iter=1, n_iter=None, penalty=None,
    #        power_t=0.25, random_state=42, shuffle=True, tol=None, verbose=0,
    #        warm_start=True)
