# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import os
import numpy as np
import numpy.random as rnd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


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
    np.random.seed(42)  # to make this notebook's output stable across runs

    m = 100  # the count of samples, the lens of data set X
    X = 6 * np.random.rand(m, 1) - 3  # X in (-3, 3)
    y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)  # y = a*x*x+b*x+c   y in (0, 10.5)

    # plot the polynomial points image
    plt.plot(X, y, "b.")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([-3, 3, 0, 10])
    save_fig("quadratic_data_plot")
    # plt.show()  # points image

    # polynomial regression model y = a*x*x+b*x+c
    poly_features = PolynomialFeatures(degree=2, include_bias=False)  # degree=2: x*x   include_bias: add c to linear
    X_poly = poly_features.fit_transform(X)
    print('X[0]: ', X[0], '     X_poly: ', X_poly[0])
    # X[0]: [-0.75275929]  X_poly: [-0.75275929  0.56664654]  0.5666=0.7527*0.7527 is a weight added to the data -0.7527

    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)
    print('intercept: ', lin_reg.intercept_, '  theta: ', lin_reg.coef_)
    # intercept:  [1.78134581]  theta:  [[0.93366893 0.56456263]]

    # X_new for testing the lin_reg model
    X_new = np.linspace(-3, 3, 100).reshape(100, 1)  # linespace() returns a points array with same interval
    print(X_new)  # [[-3.        ] [-2.93939394] [-2.87878788] [-2.81818182] ... ]
    X_new_poly = poly_features.transform(X_new)  # should use the poly_features to transform X_new
    print(X_new_poly)
    # [[-3.00000000e+00  9.00000000e+00] [-2.93939394e+00  8.64003673e+00] [-2.87878788e+00  8.28741965e+00] ... ]
    y_new = lin_reg.predict(X_new_poly)
    plt.plot(X, y, "b.")  # blue point as a comparison
    plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.legend(loc="upper left", fontsize=14)
    plt.axis([-3, 3, 0, 10])
    save_fig("quadratic_predictions_plot")
    # plt.show()

    # Learning Curves

    for style, width, degree in (("g-", 1, 300), ("b--", 2, 2), ("r-+", 2, 1)):
        polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
        std_scaler = StandardScaler()
        lin_reg = LinearRegression()
        polynomial_regression = Pipeline([
            ("poly_features", polybig_features),
            ("std_scaler", std_scaler),
            ("lin_reg", lin_reg),
        ])
        polynomial_regression.fit(X, y)
        y_newbig = polynomial_regression.predict(X_new)
        plt.plot(X_new, y_newbig, style, label=str(degree), linewidth=width)

    plt.plot(X, y, "b.", linewidth=3)
    plt.legend(loc="upper left")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([-3, 3, 0, 10])
    save_fig("high_degree_polynomials_plot")
    plt.show()