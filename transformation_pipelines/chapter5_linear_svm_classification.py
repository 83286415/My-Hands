# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
'''
    LinearSVC: 
    Linear Support Vector Classification.
    Similar to SVC with parameter kernel='linear', but implemented in terms of
    liblinear rather than libsvm, so it has more flexibility in the choice of
    penalties and loss functions and should scale better to large numbers of
    samples. But cannot find support vectors automatically.
'''

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

    # plot decision boundary

    # plot pretty figures
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    # load iris data
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]  # petal length, petal width like [[length, width], [], [], ...]
    y = iris["target"]

    setosa_or_versicolor = (y == 0) | (y == 1)  # target 0 or 1
    X = X[setosa_or_versicolor]
    y = y[setosa_or_versicolor]

    # SVM Classifier model
    svm_clf = SVC(kernel="linear", C=float("inf"))
    svm_clf.fit(X, y)
    print(svm_clf.coef_)  # [[1.29411744 0.82352928]] two features so two columns: length, width
    print(svm_clf.intercept_)  # [-3.78823471]

    # define three lines
    x0 = np.linspace(0, 5.5, 200)
    pred_1 = 5 * x0 - 20  # green dot line
    pred_2 = x0 - 1.8  # purple line
    pred_3 = 0.1 * x0 + 0.5  # red line

    # plot two margin images
    plt.figure(figsize=(12, 2.7))

    plt.subplot(121)  # the left image: three color lines
    plt.plot(x0, pred_1, "g--", linewidth=2)  # plot(x range, y range, color and pattern, etc)
    plt.plot(x0, pred_2, "m-", linewidth=2)
    plt.plot(x0, pred_3, "r-", linewidth=2)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", label="Iris-Versicolor")  # blue square
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", label="Iris-Setosa")  # yellow circle
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="upper left", fontsize=14)
    plt.axis([0, 5.5, 0, 2])

    plt.subplot(122)  # the right image: margin
    plot_svc_decision_boundary(svm_clf, 0, 5.5)  # petal length range: (0, 5.5)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs")
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo")
    plt.xlabel("Petal length", fontsize=14)
    plt.axis([0, 5.5, 0, 2])

    save_fig("large_margin_classification_plot")
    # plt.show()

    # sensitivity to features scales

    # define a new SVC model without scaling train data
    Xs = np.array([[1, 50], [5, 20], [3, 80], [5, 60]]).astype(np.float64)
    ys = np.array([0, 0, 1, 1])
    svm_clf = SVC(kernel="linear", C=100)  # smaller C than above SVC(C=float("inf")), so bigger margin
    svm_clf.fit(Xs, ys)

    plt.figure(figsize=(12, 3.2))
    plt.subplot(121)  # plot the margin without scaling
    plt.plot(Xs[:, 0][ys == 1], Xs[:, 1][ys == 1], "bo")
    plt.plot(Xs[:, 0][ys == 0], Xs[:, 1][ys == 0], "ms")
    plot_svc_decision_boundary(svm_clf, 0, 6)  # x range (1, 5) so min is 0 and max is 6
    plt.xlabel("$x_0$", fontsize=20)
    plt.ylabel("$x_1$  ", fontsize=20, rotation=0)
    plt.title("Unscaled", fontsize=16)
    plt.axis([0, 6, 0, 90])

    # scale train data and re-fit
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(Xs)  # fit_transform must follows scale()
    svm_clf.fit(X_scaled, ys)  # re-fit train data scaled in SVC model

    plt.subplot(122)  # the image is much better to show the margin which is with scaled train data
    plt.plot(X_scaled[:, 0][ys == 1], X_scaled[:, 1][ys == 1], "bo")
    plt.plot(X_scaled[:, 0][ys == 0], X_scaled[:, 1][ys == 0], "ms")
    plot_svc_decision_boundary(svm_clf, -2, 2)  # try to define x range begins with (-2, 2)
    plt.xlabel("$x_0$", fontsize=20)
    plt.title("Scaled", fontsize=16)
    plt.axis([-2, 2, -2, 2])

    save_fig("sensitivity_to_feature_scales_plot")
    # plt.show()

    # margin violations - this code is in book chapter 5 P148

    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = (iris["target"] == 2).astype(np.float64)  # binary classification - target 2: Iris-Virginica

    # join Scale and LinearSVC into a pipeline to ignore the Scale's transform_fit
    svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)),
    ])  # loss = hinge: https://blog.csdn.net/fendegao/article/details/79968994
    # here LinearSVC() not SVC above. LinearSVC cannot find support vectors automatically but SVC can.
    '''
    Pipeline(memory=None,
     steps=[('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), 
            ('linear_svc', LinearSVC(C=1, class_weight=None, dual=True, fit_intercept=True,
            intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr',
            penalty='l2', random_state=42, tol=0.0001, verbose=0))])
    '''
    svm_clf.fit(X, y)
    print(svm_clf.predict([[5.5, 1.7]]))  # [1.]    so length 5.5 and width 1.7 is target 2

    # make two pipelines with different C in SVC models
    scaler = StandardScaler()
    svm_clf1 = LinearSVC(C=1, loss="hinge", random_state=42)
    svm_clf2 = LinearSVC(C=100, loss="hinge", random_state=42)  # C is bigger but margin is smaller

    scaled_svm_clf1 = Pipeline([
        ("scaler", scaler),
        ("linear_svc", svm_clf1),
    ])
    scaled_svm_clf2 = Pipeline([
        ("scaler", scaler),
        ("linear_svc", svm_clf2),
    ])
    '''
    Pipeline(memory=None,
     steps=[('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), 
            ('linear_svc', LinearSVC(C=100, class_weight=None, dual=True, fit_intercept=True,
            intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr',
            penalty='l2', random_state=42, tol=0.0001, verbose=0))])
    '''
    scaled_svm_clf1.fit(X, y)
    scaled_svm_clf2.fit(X, y)

    # Convert to unscaled parameters: cancel scale_ to w(coef) and b(intercept)
    b1 = svm_clf1.decision_function([-scaler.mean_ / scaler.scale_])
    b2 = svm_clf2.decision_function([-scaler.mean_ / scaler.scale_])
    w1 = svm_clf1.coef_[0] / scaler.scale_
    w2 = svm_clf2.coef_[0] / scaler.scale_
    svm_clf1.intercept_ = np.array([b1])
    svm_clf2.intercept_ = np.array([b2])
    svm_clf1.coef_ = np.array([w1])
    svm_clf2.coef_ = np.array([w2])

    # Find support vectors (LinearSVC does not do this automatically)
    # i don't know how to find the support vectors manually...
    # So use SVC() instead of LinearSVC() to find support vectors automatically.
    t = y * 2 - 1
    support_vectors_idx1 = (t * (X.dot(w1) + b1) < 1).ravel()  # ravel: flatten
    support_vectors_idx2 = (t * (X.dot(w2) + b2) < 1).ravel()  # not only support vectors and violations involved
    svm_clf1.support_vectors_ = X[support_vectors_idx1]
    svm_clf2.support_vectors_ = X[support_vectors_idx2]
    print(svm_clf1.support_vectors_)

    plt.figure(figsize=(24, 9))
    plt.subplot(121)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^", label="Iris-Virginica")
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs", label="Iris-Versicolor")
    plot_svc_decision_boundary(svm_clf1, 8, 12)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="upper left", fontsize=14)
    plt.title("$C = {}$".format(svm_clf1.C), fontsize=16)
    plt.axis([4, 6, 0.8, 2.8])

    plt.subplot(122)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
    plot_svc_decision_boundary(svm_clf2, 8, 12)
    plt.xlabel("Petal length", fontsize=14)
    plt.title("$C = {}$".format(svm_clf2.C), fontsize=16)
    plt.axis([4, 6, 0.8, 2.8])

    save_fig("regularization_plot")
    plt.show()
