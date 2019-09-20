# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

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


def plot_2D_decision_function(w, b, ylabel=True, x1_lim=[-3, 3]):
    x1 = np.linspace(x1_lim[0], x1_lim[1], 200)
    y = w * x1 + b
    m = 1 / w

    plt.plot(x1, y)
    plt.plot(x1_lim, [1, 1], "k:")
    plt.plot(x1_lim, [-1, -1], "k:")
    plt.axhline(y=0, color='k')  # Add a horizontal line across the axis.
    plt.axvline(x=0, color='k')  # vertical line
    plt.plot([m, m], [0, 1], "k--")
    plt.plot([-m, -m], [0, -1], "k--")
    plt.plot([-m, m], [0, 0], "k-o", linewidth=3)
    plt.axis(x1_lim + [-2, 2])
    plt.xlabel(r"$x_1$", fontsize=16)
    if ylabel:
        plt.ylabel(r"$w_1 x_1$  ", rotation=0, fontsize=16)
    plt.title(r"$w_1 = {}$".format(w), fontsize=16)


def plot_3D_decision_function(ax, w, b, x1_lim=[4, 6], x2_lim=[0.8, 2.8]):  # http://c.biancheng.net/view/2720.html
    x1_in_bounds = (X[:, 0] > x1_lim[0]) & (X[:, 0] < x1_lim[1])  # the index of x in (4, 6)
    X_crop = X[x1_in_bounds]
    y_crop = y[x1_in_bounds]
    x1s = np.linspace(x1_lim[0], x1_lim[1], 20)
    x2s = np.linspace(x2_lim[0], x2_lim[1], 20)
    x1, x2 = np.meshgrid(x1s, x2s)
    xs = np.c_[x1.ravel(), x2.ravel()]
    df = (xs.dot(w) + b).reshape(x1.shape)
    m = 1 / np.linalg.norm(w)  # m = wT
    boundary_x2s = -x1s*(w[0]/w[1])-b/w[1]
    margin_x2s_1 = -x1s*(w[0]/w[1])-(b-1)/w[1]
    margin_x2s_2 = -x1s*(w[0]/w[1])-(b+1)/w[1]
    ax.plot_surface(x1s, x2, np.zeros_like(x1),
                    color="b", alpha=0.2, cstride=100, rstride=100)  # plot_surface(x,y,z)
    ax.plot(x1s, boundary_x2s, 0, "k-", linewidth=2, label=r"$h=0$")
    ax.plot(x1s, margin_x2s_1, 0, "k--", linewidth=2, label=r"$h=\pm 1$")
    ax.plot(x1s, margin_x2s_2, 0, "k--", linewidth=2)
    ax.plot(X_crop[:, 0][y_crop==1], X_crop[:, 1][y_crop==1], 0, "g^")
    ax.plot_wireframe(x1, x2, df, alpha=0.3, color="k")  # plot frame
    ax.plot(X_crop[:, 0][y_crop==0], X_crop[:, 1][y_crop==0], 0, "bs")
    ax.axis(x1_lim + x2_lim)
    ax.text(4.5, 2.5, 3.8, "Decision function $h$", fontsize=15)
    ax.set_xlabel(r"Petal length", fontsize=15)
    ax.set_ylabel(r"Petal width", fontsize=15)
    ax.set_zlabel(r"$h = \mathbf{w}^T \mathbf{x} + b$", fontsize=18)
    ax.legend(loc="upper left", fontsize=16)


if __name__ == '__main__':

    # Decision Function and Predictions

    # data X and y
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = (iris["target"] == 2).astype(np.float64)  # Iris-Virginica

    # make two pipelines with different C in SVC models
    scaler = StandardScaler()
    svm_clf2 = LinearSVC(C=100, loss="hinge", random_state=42)  # C is bigger but margin is smaller
    scaled_svm_clf2 = Pipeline([
        ("scaler", scaler),
        ("linear_svc", svm_clf2),
    ])
    scaled_svm_clf2.fit(X, y)

    # plot 3D decision function
    fig = plt.figure(figsize=(11, 6))
    ax1 = fig.add_subplot(111, projection='3d')  # add a 3D layer
    plot_3D_decision_function(ax1, w=svm_clf2.coef_[0], b=svm_clf2.intercept_[0])

    save_fig("iris_3D_plot")
    # plt.show()

    # Small weight vector results in a large margin

    # plot: different w make different margins
    plt.figure(figsize=(12, 3.2))
    plt.subplot(121)
    plot_2D_decision_function(1, 0)
    plt.subplot(122)
    plot_2D_decision_function(0.5, 0, ylabel=False)
    save_fig("small_w_large_margin_plot")
    # plt.show()

    # Hinge loss

    # plot:     t: x axis     h: y axis
    t = np.linspace(-2, 4, 200)
    h = np.where(1 - t < 0, 0, 1 - t)  # np.where(condition, x, y); condition satisfied: return x; or return y

    plt.figure(figsize=(5, 2.8))
    plt.plot(t, h, "b-", linewidth=2, label="$max(0, 1 - t)$")
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.yticks(np.arange(-1, 2.5, 1))  # refer to cloud note matplotlib: each scale in y axis from (-1, 2.5)
    plt.xlabel("$t$", fontsize=16)
    plt.axis([-2, 4, -1, 2.5])
    plt.legend(loc="upper right", fontsize=16)
    save_fig("hinge_plot")
    plt.show()