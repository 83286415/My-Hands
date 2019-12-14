# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import warnings
import time

# Chapter import
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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

    # Load MNIST
    mnist = fetch_mldata('MNIST original')
    X = mnist["data"]
    y = mnist["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1)

    # MDS: Multidimensional Scaling
    mds = MDS(n_components=2, random_state=42)
    X_reduced_mds = mds.fit_transform(X_train)  # it takes a very long time to fit...

    # ISOmap
    isomap = Isomap(n_components=2)
    X_reduced_isomap = isomap.fit_transform(X_train)

    # TSNE
    tsne = TSNE(n_components=2, random_state=42)
    X_reduced_tsne = tsne.fit_transform(X_train)

    # LDA: Linear Discriminant Analysis
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X_train, y_train)
    X_reduced_lda = lda.transform(X_train)

    # plot
    titles = ["MDS", "Isomap", "t-SNE", "LDA"]

    plt.figure(figsize=(14, 4))

    for subplot, title, X_reduced in zip((141, 142, 143, 144), titles,
                                         (X_reduced_mds, X_reduced_isomap, X_reduced_tsne, X_reduced_lda)):
        plt.subplot(subplot)
        plt.title(title, fontsize=14)
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
        plt.xlabel("$z_1$", fontsize=18)
        if subplot == 131:
            plt.ylabel("$z_2$", fontsize=18, rotation=0)
        plt.grid(True)

    save_fig("other_dim_reduction_plot")
    plt.show()