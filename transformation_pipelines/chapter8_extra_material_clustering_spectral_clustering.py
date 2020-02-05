# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# Chapter import
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_moons

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


def plot_spectral_clustering(sc, X, size, alpha, show_xlabels=True, show_ylabels=True):
    plt.scatter(X[:, 0], X[:, 1], marker='o', s=size, c='gray', cmap="Paired", alpha=alpha)
    plt.scatter(X[:, 0], X[:, 1], marker='o', s=30, c='w')
    plt.scatter(X[:, 0], X[:, 1], marker='.', s=10, c=sc.labels_, cmap="Paired")

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom='off')
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft='off')
    plt.title("RBF gamma={}".format(sc.gamma), fontsize=14)


if __name__ == '__main__':

    # refer to cloud note spectral clustering

    # data set
    X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)

    # build models
    sc1 = SpectralClustering(n_clusters=2, gamma=100, random_state=42)
    sc1.fit(X)
    sc2 = SpectralClustering(n_clusters=2, gamma=1, random_state=42)
    sc2.fit(X)

    print(np.percentile(sc1.affinity_matrix_, 95))  # 0.04251990648936265
    print(np.percentile(sc2.affinity_matrix_, 95))  # 0.9689155435458034

    # plot
    plt.figure(figsize=(9, 3.2))

    plt.subplot(121)
    plot_spectral_clustering(sc1, X, size=500, alpha=0.1)

    plt.subplot(122)
    plot_spectral_clustering(sc2, X, size=4000, alpha=0.01, show_ylabels=False)

    plt.show()