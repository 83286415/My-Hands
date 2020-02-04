# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# Chapter import
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier

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


def plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):
    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True
    anomalies_mask = dbscan.labels_ == -1  # abnormal points
    non_core_mask = ~(core_mask | anomalies_mask)  # ~:not  |:or

    cores = dbscan.components_
    anomalies = X[anomalies_mask]
    non_cores = X[non_core_mask]

    plt.scatter(cores[:, 0], cores[:, 1],
                c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
    plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20, c=dbscan.labels_[core_mask])
    plt.scatter(anomalies[:, 0], anomalies[:, 1],
                c="r", marker="x", s=100)  # abnormal points in x
    plt.scatter(non_cores[:, 0], non_cores[:, 1], c=dbscan.labels_[non_core_mask], marker=".")
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom='off')
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft='off')
    plt.title("eps={:.2f}, min_samples={}".format(dbscan.eps, dbscan.min_samples), fontsize=14)


def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)


def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=50, linewidths=50,
                color=cross_color, zorder=11, alpha=1)


def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom='off')
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft='off')


if __name__ == '__main__':

    # refer to cloud note DBSCAN

    # data set
    X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)

    dbscan = DBSCAN(eps=0.05, min_samples=5)  # r = 0.05  min points in r = 5
    dbscan.fit(X)

    # result
    print(dbscan.labels_[:10])  # [ 0  2 -1 -1  1  0  0  0  2  5] the  first ten samples labels assigned
    print(len(dbscan.core_sample_indices_))  # 808
    print(dbscan.core_sample_indices_[:10])  # [ 0  4  5  6  7  8 10 11 12 13]
    print(dbscan.components_[:3])  # Copy of each core sample found by training.
    '''[[-0.02137124  0.40618608]
        [-0.84192557  0.53058695]
        [ 0.58930337 -0.32137599]]'''
    print(np.unique(dbscan.labels_))  # [-1  0  1  2  3  4  5  6]

    dbscan2 = DBSCAN(eps=0.2)  # smaller r
    dbscan2.fit(X)

    # plot
    plt.figure(figsize=(9, 3.2))

    plt.subplot(121)
    plot_dbscan(dbscan, X, size=100)

    plt.subplot(122)
    plot_dbscan(dbscan2, X, size=600, show_ylabels=False)

    save_fig("dbscan_diagram")
    # plt.show()

    # knn with dbscan's core samples
    dbscan = dbscan2
    knn = KNeighborsClassifier(n_neighbors=50)
    knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])  # fit knn with dbscan's core samples

    X_new = np.array([[-0.5, 0], [0, 0.5], [1, -0.1], [2, 1]])
    print(knn.predict(X_new))
    print(knn.predict_proba(X_new))

    # plot
    plt.figure(figsize=(6, 3))
    plot_decision_boundaries(knn, X, show_centroids=False)
    plt.scatter(X_new[:, 0], X_new[:, 1], c="b", marker="+", s=200, zorder=10)
    save_fig("cluster_classification_diagram")
    plt.show()

    y_dist, y_pred_idx = knn.kneighbors(X_new, n_neighbors=1) # return r(distance) and index
    y_pred = dbscan.labels_[dbscan.core_sample_indices_][y_pred_idx]
    print(y_pred)
    y_pred[y_dist > 0.2] = -1  # exclude points whose r > 0.2
    print(y_pred.ravel())  # array([-1,  0,  1, -1])