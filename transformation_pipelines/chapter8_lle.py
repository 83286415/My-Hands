# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import warnings
import time

# Chapter import
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.metrics import mean_squared_error
from sklearn.manifold import LocallyLinearEmbedding
from mpl_toolkits.mplot3d import Axes3D


# To plot pretty figures
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


# Where to save the figures
WORKING_PATH = os.path.abspath(os.path.join(os.getcwd(), '..'))
ROOT_PATH = os.path.join(WORKING_PATH, 'Hands on SK and TS\\')
CHAPTER_ID = "dimensionality_reduction"


def save_fig(fig_id, tight_layout=True):
    path = image_path(fig_id) + ".png"
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)  # cannot save file if path doesn't exist


def image_path(fig_id):
    return os.path.join(ROOT_PATH, "images", CHAPTER_ID, fig_id)


if __name__ == '__main__':

    # swiss roll data set
    X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=41)

    # build model
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)  # LLE: k=10
    X_reduced = lle.fit_transform(X)

    # plot
    fig = plt.figure(figsize=(12, 6))

    # plot unrolled swiss in 2D
    plt.subplot(121)
    plt.title("Unrolled swiss roll using LLE", fontsize=14)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
    plt.xlabel("$z_1$", fontsize=18)
    plt.ylabel("$z_2$", fontsize=18)
    plt.axis([-0.065, 0.055, -0.1, 0.12])
    plt.grid(True)

    # plot rolled swiss in 3D
    plt.subplot(122)
    plt.title("Rolled swiss roll using LLE", fontsize=14)
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.hot)
    # ax.view_init(10, -70)
    ax.set_xlabel("$x_1$", fontsize=18)
    ax.set_ylabel("$x_2$", fontsize=18)
    ax.set_zlabel("$x_3$", fontsize=18)
    plt.grid(True)

    save_fig("lle_unrolling_plot")
    plt.show()