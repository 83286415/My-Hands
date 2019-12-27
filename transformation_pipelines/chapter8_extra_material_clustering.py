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
from sklearn.cluster import MiniBatchKMeans
from timeit import timeit


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


def load_next_batch(batch_size):
    return X[np.random.choice(len(X), batch_size, replace=False)]


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
    # different random seeds, different cluster centers.

    save_fig("kmeans_variability_diagram")
    # plt.show()

    # 7. Inertia

    print(kmeans.inertia_)  # refer to cloud note Inertia Metric  # 211.5985372581684
    print(kmeans.score(X))  # -211.59853725816856  # score: the higher, the better

    # 8. Multiple Initializations

    # 10 initializations model: run 10 times initializations to improve Inertia value, reducing it.
    kmeans_rnd_10_inits = KMeans(n_clusters=5, init="random", n_init=10, algorithm="full", random_state=11)
    kmeans_rnd_10_inits.fit(X)

    # plot
    plt.figure(figsize=(8, 4))
    plot_decision_boundaries(kmeans_rnd_10_inits, X)  # much better boundaries
    # plt.show()

    # 9 KMeans++

    # regular kmeans:
    kmeans_random = KMeans(n_clusters=5, init="random", n_init=1, algorithm="full", random_state=19)
    kmeans_random.fit(X)
    # k-means++
    kmeans_plus = KMeans(n_clusters=5, init="k-means++", n_init=1, algorithm="full", random_state=None)
    kmeans_plus.fit(X)
    # good init kmeans: specify the cluster centers for a good init
    good_init = np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])
    kmeans_good = KMeans(n_clusters=5, init=good_init, n_init=1, random_state=None)  # random_state=None
    kmeans_good.fit(X)
    print('kmeans_random inertia: ', kmeans_random.inertia_)  # 237.46249169442845
    print('kmeans_plus inertia: ', kmeans_plus.inertia_)  # 211.59853725816822  the best kmeans
    print('kmeans_good inertia: ', kmeans_good.inertia_)  # 211.5985372581684

    # 10. Mini-Batch K-Means

    # build an example model
    minibatch_kmeans = MiniBatchKMeans(n_clusters=5, random_state=42)
    minibatch_kmeans.fit(X)
    '''MiniBatchKMeans(batch_size=100, compute_labels=True, init='k-means++',
        init_size=None, max_iter=100, max_no_improvement=10, n_clusters=5,
        n_init=3, random_state=42, reassignment_ratio=0.01, tol=0.0,
        verbose=0)'''
    print('minibatch_kmeans: ', minibatch_kmeans.inertia_)  # 211.93186531476775 good

    # memmap; in case of data set doesn't fit memory
    filename = "my_mnist.data"
    m, n = 50000, 28 * 28
    X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m, n))  # memmap refer to chapter8_pca.py

    # build model
    minibatch_kmeans = MiniBatchKMeans(n_clusters=10, batch_size=10, random_state=42)
    minibatch_kmeans.fit(X_mm)

    np.random.seed(42)
    k = 5  # clusters count
    n_init = 10
    n_iterations = 100
    batch_size = 100
    init_size = 500  # more data for K-Means++ initialization
    evaluate_on_last_n_iters = 10
    best_kmeans = None

    # get the best kmeans score
    for init in range(n_init):
        minibatch_kmeans = MiniBatchKMeans(n_clusters=k, init_size=init_size)
        X_init = load_next_batch(init_size)
        minibatch_kmeans.partial_fit(X_init)

        minibatch_kmeans.sum_inertia_ = 0
        for iteration in range(n_iterations):
            X_batch = load_next_batch(batch_size)
            minibatch_kmeans.partial_fit(X_batch)
            if iteration >= n_iterations - evaluate_on_last_n_iters:
                minibatch_kmeans.sum_inertia_ += minibatch_kmeans.inertia_

        if (best_kmeans is None or
                minibatch_kmeans.sum_inertia_ < best_kmeans.sum_inertia_):
            best_kmeans = minibatch_kmeans

    print('best socre of mini batch kmeans: ', best_kmeans.score(X))  # -211.70999744411483

    # get the regular kmeans and mini batch kmeans fit time and inertia
    times = np.empty((100, 2))
    inertias = np.empty((100, 2))
    for k in range(1, 101):
        kmeans = KMeans(n_clusters=k, random_state=42)
        minibatch_kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
        print("\r{}/{}".format(k, 100), end="")
        times[k - 1, 0] = timeit("kmeans.fit(X)", number=10, globals=globals())
        times[k - 1, 1] = timeit("minibatch_kmeans.fit(X)", number=10, globals=globals())
        inertias[k - 1, 0] = kmeans.inertia_
        inertias[k - 1, 1] = minibatch_kmeans.inertia_

    # plot
    plt.figure(figsize=(10, 4))

    plt.subplot(121)
    plt.plot(range(1, 101), inertias[:, 0], "r--", label="K-Means")
    plt.plot(range(1, 101), inertias[:, 1], "b.-", label="Mini-batch K-Means")
    plt.xlabel("$k$", fontsize=16)
    # plt.ylabel("Inertia", fontsize=14)
    plt.title("Inertia", fontsize=14)
    plt.legend(fontsize=14)
    plt.axis([1, 100, 0, 100])

    plt.subplot(122)
    plt.plot(range(1, 101), times[:, 0], "r--", label="K-Means")
    plt.plot(range(1, 101), times[:, 1], "b.-", label="Mini-batch K-Means")
    plt.xlabel("$k$", fontsize=16)
    # plt.ylabel("Training time (seconds)", fontsize=14)
    plt.title("Training time (seconds)", fontsize=14)
    plt.axis([1, 100, 0, 6])
    # plt.legend(fontsize=14)

    save_fig("minibatch_kmeans_vs_kmeans")
    # plt.show()

    # 11. Finding the optimal number of clusters

    # get the inertias list while cluster count k is increasing
    kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X) for k in range(1, 10)]  # model list
    inertias = [model.inertia_ for model in kmeans_per_k]  # inertial list

    # plot the elbow and the best k value is around the elbow
    plt.figure(figsize=(8, 3.5))
    plt.plot(range(1, 10), inertias, "bo-")
    plt.xlabel("$k$", fontsize=14)
    plt.ylabel("Inertia", fontsize=14)
    plt.annotate('Elbow',
                 xy=(4, inertias[3]),
                 xytext=(0.55, 0.55),
                 textcoords='figure fraction',
                 fontsize=16,
                 arrowprops=dict(facecolor='black', shrink=0.1)
                 )
    plt.axis([1, 8.5, 0, 1300])
    save_fig("inertia_vs_k_diagram")
    plt.show()