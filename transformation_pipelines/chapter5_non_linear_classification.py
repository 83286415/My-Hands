# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_moons
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


def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)


def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)  # the center of the street
    y_decision = clf.decision_function(X).reshape(x0.shape)  # decision_function: return a score of distance to street
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)  # plot the hyperplane: the center line of the street
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)  # plot and color the contour field
    # draw contour lines and color contours field


def gaussian_rbf(x, landmark, gamma):
    return np.exp(-gamma * np.linalg.norm(x - landmark, axis=1)**2)  # refer to book P151 and cloud note - math concept


if __name__ == '__main__':

    # Polynomial Kernel

    # plot moons data set
    X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    # plt.show()

    # plot contour lines and the classification line
    polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),  # 3 degrees
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))
    ])  # LinearSVC(): one-vs-rest in multi-classification
    '''
    Pipeline(memory=None,
     steps=[('poly_features', PolynomialFeatures(degree=3, include_bias=True, interaction_only=False)), 
     ('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), 
     ('svm_clf', LinearSVC(C=10, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr',
     penalty='l2', random_state=42, tol=0.0001, verbose=0))])
     '''
    polynomial_svm_clf.fit(X, y)
    plt.figure(figsize=(8, 5))
    plot_predictions(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])  # plot contour lines and the classification line
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])  # plot the data point in blue and green

    save_fig("moons_polynomial_svc_plot")
    # plt.show()

    # kernel skill with SVC: bigger degrees and coef0 -> over fit   smaller degrees and coef0 -> under fit
    poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))     # 3 degrees coef0=1, coef0: kernel parameter
    ])
    poly_kernel_svm_clf.fit(X, y)
    '''Pipeline(memory=None,
     steps=[('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), 
     ('svm_clf', SVC(C=5, cache_size=200, class_weight=None, coef0=1,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='poly',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False))])
      '''
    poly100_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=10, coef0=100, C=5))    # 10 degrees coef0=100
    ])
    poly100_kernel_svm_clf.fit(X, y)
    '''Pipeline(memory=None,
     steps=[('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), 
     ('svm_clf', SVC(C=5, cache_size=200, class_weight=None, coef0=100,
      decision_function_shape='ovr', degree=10, gamma='auto', kernel='poly',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False))])'''

    plt.figure(figsize=(11, 4))

    plt.subplot(121)
    plot_predictions(poly_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    plt.title(r"$d=3, r=1, C=5$", fontsize=18)

    plt.subplot(122)
    plot_predictions(poly100_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    plt.title(r"$d=10, r=100, C=5$", fontsize=18)

    save_fig("moons_kernelized_polynomial_svc_plot")
    # plt.show()

    # Adding Similarity Features

    # only X1D cannot be classified by linear model but X2D can
    X1D = np.linspace(-4, 4, 9).reshape(-1, 1)  # reshape(-1, 1): only column count is decided to be 1
    X2D = np.c_[X1D, X1D ** 2]
    '''X2D:
        [[-4. 16.]
         [-3.  9.]
         [-2.  4.]
         [-1.  1.]
         [ 0.  0.]
         [ 1.  1.]
         [ 2.  4.]
         [ 3.  9.]
         [ 4. 16.]]
    '''
    gamma = 0.3  # used in RBF

    # used in left image: add new features to x1s and make it into x2s, x3s, two new data sets
    x1s = np.linspace(-4.5, 4.5, 200).reshape(-1, 1)  # used to plot dot line
    x2s = gaussian_rbf(x1s, -2, gamma)  # landmark = -2     # for green - - line
    x3s = gaussian_rbf(x1s, 1, gamma)                       # for blue ... line

    # used in right image: add two new similar features to X1D by rbf
    XK = np.c_[gaussian_rbf(X1D, -2, gamma), gaussian_rbf(X1D, 1, gamma)]  # join two rbf array with landmarks -2, 1
    yk = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])

    plt.figure(figsize=(11, 4))

    plt.subplot(121)
    plt.grid(True, which='both')  # plot with grid
    plt.axhline(y=0, color='k')  # Add a horizontal line across the axis. # k: black
    plt.scatter(x=[-2, 1], y=[0, 0], s=150, alpha=0.5, c="red")  # plot the red circle on those two points
    plt.plot(X1D[:, 0][yk == 0], np.zeros(4), "bs")  # plot blue square box on points along x axis
    plt.plot(X1D[:, 0][yk == 1], np.zeros(5), "g^")  # plot green triangle
    plt.plot(x1s, x2s, "g--")  # plot gaussian rbf line which is like a hill
    plt.plot(x1s, x3s, "b:")
    plt.gca().get_yaxis().set_ticks([0, 0.25, 0.5, 0.75, 1])  # gca(): get the current axis.
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"Similarity", fontsize=14)
    plt.annotate(r'$\mathbf{x}$',  # add X test into image
                 xy=(X1D[3, 0], 0),  # the arrow points to [-1, 0] from annotate X
                 xytext=(-0.5, 0.20),   # test X location
                 ha="center",
                 arrowprops=dict(facecolor='black', shrink=0.1),
                 fontsize=18,
                 )
    plt.text(-2, 0.9, "$x_2$", ha="center", fontsize=20)  # add X2 text
    plt.text(1, 0.9, "$x_3$", ha="center", fontsize=20)
    plt.axis([-4.5, 4.5, -0.1, 1.1])

    plt.subplot(122)
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')  # Add a vertical line across the axis x==0. # k: black

    '''XK:
    [[3.01194212e-01 5.53084370e-04]
     [7.40818221e-01 8.22974705e-03]
     [1.00000000e+00 6.72055127e-02]
     [7.40818221e-01 3.01194212e-01]
     [3.01194212e-01 7.40818221e-01]
     [6.72055127e-02 1.00000000e+00]
     [8.22974705e-03 7.40818221e-01]
     [5.53084370e-04 3.01194212e-01]
     [2.03995034e-05 6.72055127e-02]]'''
    plt.plot(XK[:, 0][yk == 0], XK[:, 1][yk == 0], "bs")
    plt.plot(XK[:, 0][yk == 1], XK[:, 1][yk == 1], "g^")
    plt.xlabel(r"$x_2$", fontsize=20)
    plt.ylabel(r"$x_3$  ", fontsize=20, rotation=0)
    plt.annotate(r'$\phi\left(\mathbf{x}\right)$',
                 xy=(XK[3, 0], XK[3, 1]),
                 xytext=(0.65, 0.50),
                 ha="center",
                 arrowprops=dict(facecolor='black', shrink=0.1),
                 fontsize=18,
                 )
    plt.plot([-0.1, 1.1], [0.57, -0.1], "r--", linewidth=3)
    plt.axis([-0.1, 1.1, -0.1, 1.1])

    plt.subplots_adjust(right=0.2)  # adjust the distance between subplots

    save_fig("kernel_method_plot")
    plt.show()