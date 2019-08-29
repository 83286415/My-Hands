# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge


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

    # ridge regression

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
    plt.show()

    ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
    #           cholesky:  uses the standard scipy.linalg.solve function to obtain a closed-form solution.
    ridge_reg.fit(X, y)
    print(ridge_reg.predict([[1.5]]))  # [[1.55071465]]
    print('RidgeRegression_cholesky: ', ridge_reg.intercept_, ridge_reg.coef_)  # [1.00650911] [[0.36280369]]

    sgd_reg = SGDRegressor(max_iter=5, penalty="l2", random_state=42)  # penalty L2 refer to cloud note "L2"
    sgd_reg.fit(X, y.ravel())
    print(sgd_reg.predict([[1.5]]))  # [1.13500145]
    print('SGDRegressor: ', sgd_reg.intercept_, sgd_reg.coef_)  # [0.35165674] [0.52222981]

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

    # lasso regression

    #