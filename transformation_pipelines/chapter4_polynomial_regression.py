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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


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


def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []  # mse list of train and validation data at each size
    for m in range(1, len(X_train)):  # train size as x axis
        model.fit(X_train[:m], y_train[:m])  # fit at each size of train data
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))  # mse returns loss
        val_errors.append(mean_squared_error(y_val, y_val_predict))
        # print(val_errors)  # [5.702729504261572, 5.047167241360556, 4.471884824260103, ...]

    plt.figure(figsize=(7, 4))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")  # plot rmse as y and x is 0, 1, 2, 3...
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")  # plot y using x as index array 0..N-1
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)


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
    # print(X_new)  # [[-3.        ] [-2.93939394] [-2.87878788] [-2.81818182] ... ]
    X_new_poly = poly_features.transform(X_new)  # should use the poly_features to transform X_new
    # print(X_new_poly)
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

    # plot 300 degrees, 2 degrees, 1 degree image of polynomial regression
    plt.figure(figsize=(10, 4))
    for style, width, degree in (("g-", 1, 300), ("b--", 2, 2), ("r-+", 2, 1)):
        polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
        std_scaler = StandardScaler()
        lin_reg = LinearRegression()
        polynomial_regression = Pipeline([
            ("poly_features", polybig_features),
            ("std_scaler", std_scaler),
            ("lin_reg", lin_reg),
        ])  # pipeline refer to chapter2 py
        polynomial_regression.fit(X, y)
        y_newbig = polynomial_regression.predict(X_new)
        plt.plot(X_new, y_newbig, style, label=str(degree), linewidth=width)

    plt.plot(X, y, "b.", linewidth=3)
    plt.legend(loc="upper left")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([-3, 3, 0, 10])
    save_fig("high_degree_polynomials_plot")
    # plt.show()

    # plot a linear regression model
    lin_reg = LinearRegression()

    plot_learning_curves(lin_reg, X, y)  # refer to def
    plt.axis([0, 80, 0, 3])  # [xmin, xmax, ymin, ymax]
    save_fig("underfitting_learning_curves_plot")
    # plt.show()

    # plot a polynomial regression model
    polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
        ("lin_reg", LinearRegression()),
    ])  # 10 degree polynomial and followed by a linear regression model in a pipeline

    plot_learning_curves(polynomial_regression, X, y)
    plt.axis([0, 80, 0, 3])
    save_fig("learning_curves_plot")  # polynomial model is better than linear on this data set
    plt.show()