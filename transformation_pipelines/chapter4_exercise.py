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


def to_one_hot(y):
    n_classes = y.max() + 1  # max(): return the max value of the array y
    m = len(y)  # row count
    Y_one_hot = np.zeros((m, n_classes))  # (m, max+1) zero array
    Y_one_hot[np.arange(m), y] = 1  # set 1
    return Y_one_hot


def softmax(logits):
    exps = np.exp(logits)
    exp_sums = np.sum(exps, axis=1, keepdims=True)
    return exps / exp_sums


if __name__ == '__main__':

    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = iris["target"]  # all labels are targets

    X_with_bias = np.c_[np.ones([len(X), 1]), X]  # add 1 bias to X

    np.random.seed(2042)

    # split ratio
    test_ratio = 0.2
    validation_ratio = 0.2

    # data set size definition
    total_size = len(X_with_bias)
    test_size = int(total_size * test_ratio)
    validation_size = int(total_size * validation_ratio)
    train_size = total_size - test_size - validation_size

    rnd_indices = np.random.permutation(total_size)  # shuffle data set and return th index

    # train, validation and test data set
    X_train = X_with_bias[rnd_indices[:train_size]]
    y_train = y[rnd_indices[:train_size]]
    X_valid = X_with_bias[rnd_indices[train_size:-test_size]]
    y_valid = y[rnd_indices[train_size:-test_size]]
    X_test = X_with_bias[rnd_indices[-test_size:]]
    y_test = y[rnd_indices[-test_size:]]

    # one hot code
    print(y_train[:10])  # [0 1 2 1 1 0 1 1 1 0]
    print(to_one_hot(y_train[:10]))
    ''''[[1. 0. 0.]
         [0. 1. 0.]
         [0. 0. 1.]
         [0. 1. 0.]
         [0. 1. 0.]
         [1. 0. 0.]
         [0. 1. 0.]
         [0. 1. 0.]
         [0. 1. 0.]
         [1. 0. 0.]]'''
    Y_train_one_hot = to_one_hot(y_train)
    Y_valid_one_hot = to_one_hot(y_valid)
    Y_test_one_hot = to_one_hot(y_test)

    # define theta array like (n_inputs, n_outputs)
    n_inputs = X_train.shape[1]  # == 3 (2 features plus the bias term);        shape: (90, 3)
    n_outputs = len(np.unique(y_train))  # == 3 (3 iris classes)

    eta = 0.1  # learning rate
    n_iterations = 5001
    m = len(X_train)
    epsilon = 1e-7
    alpha = 0.1  # regularization hyper parameter
    best_loss = np.infty

    Theta = np.random.randn(n_inputs, n_outputs)

    for iteration in range(n_iterations):
        logits = X_train.dot(Theta)
        Y_proba = softmax(logits)
        xentropy_loss = -np.mean(np.sum(Y_train_one_hot * np.log(Y_proba + epsilon), axis=1))
        l2_loss = 1 / 2 * np.sum(np.square(Theta[1:]))      # L2 penalty
        loss = xentropy_loss + alpha * l2_loss
        error = Y_proba - Y_train_one_hot
        gradients = 1 / m * X_train.T.dot(error) + np.r_[np.zeros([1, n_outputs]), alpha * Theta[1:]]
        Theta = Theta - eta * gradients

        logits = X_valid.dot(Theta)
        Y_proba = softmax(logits)
        xentropy_loss = -np.mean(np.sum(Y_valid_one_hot * np.log(Y_proba + epsilon), axis=1))
        l2_loss = 1 / 2 * np.sum(np.square(Theta[1:]))
        loss = xentropy_loss + alpha * l2_loss
        if iteration % 500 == 0:
            print(iteration, loss)  # print loss at each 500 iteration loop
        if loss < best_loss:
            best_loss = loss
        else:                       # loss >= best loss
            print(iteration - 1, best_loss)
            print(iteration, loss, "early stopping!")
            break
            '''
                0 3.775806436742873
                500 0.5640708554829896
                1000 0.5411729190098661
                1500 0.5349311427185633
                2000 0.533062762855566
                2500 0.5326241775798368
                2673 0.5326050502092685
                2674 0.5326050503516297 
                early stopping!
            '''

    # validation accuracy
    logits = X_valid.dot(Theta)
    Y_proba = softmax(logits)
    y_predict = np.argmax(Y_proba, axis=1)

    accuracy_score = np.mean(y_predict == y_valid)
    print(accuracy_score)  # 1.0

    # test accuracy
    logits = X_test.dot(Theta)
    Y_proba = softmax(logits)
    y_predict = np.argmax(Y_proba, axis=1)

    accuracy_score = np.mean(y_predict == y_test)
    print(accuracy_score)  # 0.9333333333333333

    # plot: some functions refer to chapter4_logistic_regression.py
    x0, x1 = np.meshgrid(
        np.linspace(0, 8, 500).reshape(-1, 1),
        np.linspace(0, 3.5, 200).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    X_new_with_bias = np.c_[np.ones([len(X_new), 1]), X_new]

    logits = X_new_with_bias.dot(Theta)
    Y_proba = softmax(logits)
    y_predict = np.argmax(Y_proba, axis=1)

    zz1 = Y_proba[:, 1].reshape(x0.shape)
    zz = y_predict.reshape(x0.shape)

    plt.figure(figsize=(10, 4))
    plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris-Virginica")
    plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris-Versicolor")
    plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris-Setosa")

    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])

    plt.contourf(x0, x1, zz, cmap=custom_cmap)
    contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
    plt.clabel(contour, inline=1, fontsize=12)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="upper left", fontsize=14)
    plt.axis([0, 7, 0, 3.5])
    plt.show()
