# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import warnings
import time

# Chapter import
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.pipeline import Pipeline
from sklearn.manifold import MDS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


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


def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)


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


def plot_clusterer_comparison(clusterer1, clusterer2, X, title1=None, title2=None):
    clusterer1.fit(X)
    clusterer2.fit(X)

    plt.figure(figsize=(10, 3.2))

    plt.subplot(121)
    plot_decision_boundaries(clusterer1, X)
    if title1:
        plt.title(title1, fontsize=14)

    plt.subplot(122)
    plot_decision_boundaries(clusterer2, X, show_ylabels=False)
    if title2:
        plt.title(title2, fontsize=14)




if __name__ == '__main__':

    # 1. Introduction

    # data set
    data = load_iris()
    X = data.data
    y = data.target
    print(data.target_names)  # ['setosa' 'versicolor' 'virginica']

    # plot: classification vs clustering
    plt.figure(figsize=(9, 3.5))

    plt.subplot(121)  # classification
    plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris-Setosa")
    # setosa, X: [sepal length, sepal width, petal length, petal width]
    plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris-Versicolor")
    plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris-Virginica")
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(fontsize=12)

    plt.subplot(122)  # it's not clustering, just showing all data points
    plt.scatter(X[:, 0], X[:, 1], c="k", marker=".")
    plt.xlabel("Petal length", fontsize=14)
    plt.tick_params(labelleft='off')

    save_fig("classification_vs_clustering_diagram")
    # plt.show()

    # 2. Gaussian mixture model

    y_pred = GaussianMixture(n_components=3, random_state=42).fit(X).predict(X)  # that is k=3, 3 clusters
    mapping = np.array([2, 0, 1])
    y_pred = np.array([mapping[cluster_id] for cluster_id in y_pred])  # change it into array
    print(np.sum(y==y_pred))  # 145
    print('Gaussian mixture model accuracy: ', np.sum(y_pred==y) / len(y_pred))  # 0.9666666666666667

    # plot
    plt.figure(figsize=(5, 3.5))
    plt.plot(X[y_pred == 0, 0], X[y_pred == 0, 1], "yo", label="Cluster 1")
    plt.plot(X[y_pred == 1, 0], X[y_pred == 1, 1], "bs", label="Cluster 2")
    plt.plot(X[y_pred == 2, 0], X[y_pred == 2, 1], "g^", label="Cluster 3")
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="upper right", fontsize=12)
    # plt.show()

    # 3. K-Means

    # data set
    blob_centers = np.array(
        [[0.2, 2.3],
         [-1.5, 2.3],
         [-2.8, 1.8],
         [-2.8, 2.8],
         [-2.8, 1.3]])   # 5 clusters' center point
    blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])  # bias away from center points. So 2 sparse clusters and 3 intense

    X, y = make_blobs(n_samples=2000, centers=blob_centers,
                      cluster_std=blob_std, random_state=7)

    # plot bolbs
    plt.figure(figsize=(8, 4))
    plot_clusters(X)
    save_fig("blobs_diagram")  # 2 sparse clusters and 3 intense
    # plt.show()

    # KMeans model
    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_pred = kmeans.fit_predict(X)
    print(y_pred is kmeans.labels_)  # True. So the y_pred is the labels
    print(kmeans.cluster_centers_)
    '''array([[-2.80389616,  1.80117999],
       [ 0.20876306,  2.25551336],
       [-2.79290307,  2.79641063],
       [-1.46679593,  2.28585348],
       [-2.80037642,  1.30082566]])'''

    # predict X new
    X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
    y_new_pred = kmeans.predict(X_new)
    print(y_new_pred)  # [1 1 2 2] the clusters labels which X_new points belongs to.

    # 4. Decision Boundaries

    # plot blobs decision boundaries
    plt.figure(figsize=(8, 4))
    plot_decision_boundaries(kmeans, X)
    save_fig("voronoi_diagram")
    # plt.show()

    # soft clustering
    y_soft_clustering = kmeans.transform(X_new)
    print(y_soft_clustering)  # return the distance array of each X instance away from all clusters' center
    '''array([[2.81093633, 0.32995317, 2.9042344 , 1.49439034, 2.88633901],
       [5.80730058, 2.80290755, 5.84739223, 4.4759332 , 5.84236351],
       [1.21475352, 3.29399768, 0.29040966, 1.69136631, 1.71086031],
       [0.72581411, 3.21806371, 0.36159148, 1.54808703, 1.21567622]])'''

    # 5. K-Means Algorithm

    # K-Means Algorithm refer to cloud note
    # build models
    kmeans_iter1 = KMeans(n_clusters=5, init="random", n_init=1, algorithm="full", max_iter=1, random_state=1)
    kmeans_iter2 = KMeans(n_clusters=5, init="random", n_init=1, algorithm="full", max_iter=2, random_state=1)
    kmeans_iter3 = KMeans(n_clusters=5, init="random", n_init=1, algorithm="full", max_iter=3, random_state=1)
    kmeans_iter1.fit(X)
    kmeans_iter2.fit(X)
    kmeans_iter3.fit(X)

    # plot
    plt.figure(figsize=(10, 8))

    plt.subplot(321)
    plot_data(X)
    plot_centroids(kmeans_iter1.cluster_centers_, circle_color='r', cross_color='w')
    plt.ylabel("$x_2$", fontsize=14, rotation=0)
    plt.tick_params(labelbottom='off')
    plt.title("Update the centroids (initially randomly)", fontsize=14)

    plt.subplot(322)
    plot_decision_boundaries(kmeans_iter1, X, show_xlabels=False, show_ylabels=False)
    plt.title("Label the instances", fontsize=14)

    plt.subplot(323)
    plot_decision_boundaries(kmeans_iter1, X, show_centroids=False, show_xlabels=False)
    plot_centroids(kmeans_iter2.cluster_centers_)

    plt.subplot(324)
    plot_decision_boundaries(kmeans_iter2, X, show_xlabels=False, show_ylabels=False)

    plt.subplot(325)
    plot_decision_boundaries(kmeans_iter2, X, show_centroids=False)
    plot_centroids(kmeans_iter3.cluster_centers_)

    plt.subplot(326)
    plot_decision_boundaries(kmeans_iter3, X, show_ylabels=False)

    save_fig("kmeans_algorithm_diagram")
    # plt.show()

    # 6. K-Means Variability

    # plot
    kmeans_rnd_init1 = KMeans(n_clusters=5, init="random", n_init=1, algorithm="full", random_state=11)
    kmeans_rnd_init2 = KMeans(n_clusters=5, init="random", n_init=1, algorithm="full", random_state=19)

    plot_clusterer_comparison(kmeans_rnd_init1, kmeans_rnd_init2, X,
                              "Solution 1", "Solution 2 (with a different random init)")

    save_fig("kmeans_variability_diagram")
    plt.show()

    # 7. Inertia

    print(kmeans.inertia_)  # refer to cloud note Inertia Metric  # 211.5985372581684
    print(kmeans.score(X))  # -211.59853725816856  # score: the higher, the better

    # 8. Multiple Initializations


