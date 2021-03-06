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
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from timeit import timeit
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.image import imread


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
    if 0:  # comment it out for costing to much time on training
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
    # plt.show()

    # plot silhouette_score. silhouette_score refer to cloud note
    print(silhouette_score(X, kmeans.labels_))  # the bigger the better
    silhouette_scores = [silhouette_score(X, model.labels_)
                         for model in kmeans_per_k[1:]]  # the mean Silhouette Coefficient of all samples

    plt.figure(figsize=(8, 3))
    plt.plot(range(2, 10), silhouette_scores, "bo-")
    plt.xlabel("$k$", fontsize=14)
    plt.ylabel("Silhouette score", fontsize=14)
    plt.axis([1.8, 8.5, 0.55, 0.7])
    save_fig("silhouette_score_vs_k_diagram")
    # plt.show()

    # silhouette diagram
    plt.figure(figsize=(11, 9))

    for k in (3, 4, 5, 6):  # loop in clusters' #
        plt.subplot(2, 2, k - 2)

        y_pred = kmeans_per_k[k - 1].labels_
        silhouette_coefficients = silhouette_samples(X, y_pred)  # returns Silhouette Coefficient for each samples

        padding = len(X) // 30
        pos = padding
        ticks = []
        for i in range(k):
            coeffs = silhouette_coefficients[y_pred == i]  # each cluster's silhouette_coefficients of each model
            coeffs.sort()

            # color = matplotlib.cm.hot
            plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs, alpha=0.7)  # color bars of each cluster
            ticks.append(pos + len(coeffs) // 2)  # x limit
            pos += len(coeffs) + padding  # y limit

        plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
        plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
        if k in (3, 5):
            plt.ylabel("Cluster")

        if k in (5, 6):
            plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.xlabel("Silhouette Coefficient")
        else:
            plt.tick_params(labelbottom='off')

        # the red vertical line of the mean Silhouette Coefficient of all samples
        plt.axvline(x=silhouette_scores[k - 2], color="red", linestyle="--")
        plt.title("$k={}$".format(k), fontsize=16)  # k on top of the each image

    save_fig("silhouette_analysis_diagram")
    # plt.show()

    # 12. Limits of K-Means     todo: COP-KMeans?

    # data set
    X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
    X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
    X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
    X2 = X2 + [6, -8]
    X = np.r_[X1, X2]
    y = np.r_[y1, y2]

    plot_clusters(X)

    # build models
    kmeans_good = KMeans(n_clusters=3, init=np.array([[-1.5, 2.5], [0.5, 0], [4, 0]]), n_init=1, random_state=42)
    kmeans_bad = KMeans(n_clusters=3, random_state=42)
    kmeans_good.fit(X)
    kmeans_bad.fit(X)

    # plot
    plt.figure(figsize=(10, 3.2))

    plt.subplot(121)
    plot_decision_boundaries(kmeans_good, X)
    plt.title("Inertia = {:.1f}".format(kmeans_good.inertia_), fontsize=14)

    plt.subplot(122)
    plot_decision_boundaries(kmeans_bad, X, show_ylabels=False)
    plt.title("Inertia = {:.1f}".format(kmeans_bad.inertia_), fontsize=14)
    # TODO: No COP-KMeans related doc. i don't know why good model has a higher inertia value.

    save_fig("bad_kmeans_diagram")
    # plt.show()

    # 13. Using clustering for image segmentation

    # data set
    input_image_path = image_path('flower.png')
    image = imread(input_image_path)
    print(image.shape)  # (312, 613, 4)

    # try to build model for testing
    X = image.reshape(-1, 3)  # change image (312, 613, 4) array into a (x, 3) shape array. x == 255083, 3: 3 channels
    print(X.shape)  # (255008, 3)
    kmeans = KMeans(n_clusters=8, random_state=42).fit(X)

    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    # labels_ is a ndarray not a index. so cluster_centers_[kmeans.labels_] is not to get the value from ndarray but put
    # the labels_ values into cluster_centers_ array.

    # kmeans.cluster_centers_:float ndarray with shape (k, n_features) Centroids found at the last iteration of k-means.
    # kmeans.labels_:   integer ndarray with shape (n_samples,) label[i] is the code or index of the centroid the
    #                   i'th observation is closest to.
    clusters = np.asarray(kmeans.cluster_centers_, dtype=np.uint8)
    print('cluster_centers_ shape: ', clusters.shape)  # (8, 3), 8: clusters count; 3: n_features
    labels = np.asarray(kmeans.labels_, dtype=np.uint8)
    print('labels_ shape: ', labels.shape)  # (255008,)
    print('segmented_img shape: ', segmented_img.shape)  # (255008, 3) refer to the # above

    segmented_img = segmented_img.reshape(image.shape)

    # here is the real image segmentation code below:
    segmented_imgs = []
    n_colors = (10, 8, 6, 4, 2)  # also cluster count in model
    for n_clusters in n_colors:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)  # re-build model with different cluster count
        segmented_img = kmeans.cluster_centers_[kmeans.labels_]
        segmented_imgs.append(segmented_img.reshape(image.shape))

    # plot
    plt.figure(figsize=(10, 5))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    plt.subplot(231)
    plt.imshow(image)
    plt.title("Original image")
    plt.axis('off')

    for idx, n_clusters in enumerate(n_colors):
        plt.subplot(232 + idx)
        plt.imshow(segmented_imgs[idx])
        plt.title("{} colors".format(n_clusters))
        plt.axis('off')

    # plt.show()
    save_fig("image_segmentation_diagram", tight_layout=False)

    # 14 Using Clustering for Preprocessing

    # data set
    X_digits, y_digits = load_digits(return_X_y=True)  # load MNIST like data set: 1797 samples total
    X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)

    # build a log model
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, y_train)
    print('log_reg score: ', log_reg.score(X_test, y_test))  # 0.9666666666666667 it's a base line

    # build pipeline: K-means used as pre-processing step
    pipeline = Pipeline([
        ("kmeans", KMeans(n_clusters=50, random_state=42)),
        ("log_reg", LogisticRegression(random_state=42)),
    ])  # 50 clusters
    pipeline.fit(X_train, y_train)
    print('pipeline score: ', pipeline.score(X_test, y_test))  # 0.9822222222222222 better than base line

    # find a better parameter of K-means by GridSearchCV
    param_grid = dict(kmeans__n_clusters=range(2, 100))
    print(param_grid)  # {'kmeans__n_clusters': range(2, 100)}, model name in pipe line + __ + param name as dict's key
    grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
    grid_clf.fit(X_train, y_train)
    print('best param: ', grid_clf.best_params_)  # {'kmeans__n_clusters': 90}
    print('score of pipeline with best param: ', grid_clf.score(X_test, y_test))  # 0.9844444444444445 with param == 90

    # 15. Clustering for Semi-supervised Learning
    n_labeled = 50

    # build a log model
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])  # only 50 labeled train data: 50 random data
    print('random 50 label data score: ', log_reg.score(X_test, y_test))  # 0.8266666666666667

    # method 1:
    k = 50
    kmeans = KMeans(n_clusters=k, random_state=42)
    X_digits_dist = kmeans.fit_transform(X_train)  # make all train data into 50 clusters
    representative_digit_idx = np.argmin(X_digits_dist, axis=0)  # find the most close value to each cluster's center
    X_representative_digits = X_train[representative_digit_idx]  # the 50 representative digits array

    # plot
    plt.figure(figsize=(8, 2))
    for index, X_representative_digit in enumerate(X_representative_digits):
        plt.subplot(k // 10, 10, index + 1)
        plt.imshow(X_representative_digit.reshape(8, 8), cmap="binary", interpolation="bilinear")
        plt.axis('off')

    save_fig("representative_images_diagram", tight_layout=False)
    # plt.show()

    # define the label manually to representative digits array
    y_representative_digits = np.array([
        4, 8, 0, 6, 8, 3, 7, 7, 9, 2,
        5, 5, 8, 5, 2, 1, 2, 9, 6, 1,
        1, 6, 9, 0, 8, 3, 0, 7, 4, 1,
        6, 5, 2, 4, 1, 8, 6, 3, 9, 2,
        4, 2, 9, 4, 7, 6, 2, 3, 1, 1])  # it's not 50 random data but 50 clusters center data. So it means better score.

    # try test data on this 50 representative data trained model
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_representative_digits, y_representative_digits)
    print('representative 50 label score: ', log_reg.score(X_test, y_test))  # 0.9244444444444444 better than random one

    # method 2:
    # make a zero y array
    y_train_propagated = np.empty(len(X_train), dtype=np.int32)
    for i in range(k):
        y_train_propagated[kmeans.labels_ == i] = y_representative_digits[i]  # give representative labels to new y
        # give the cluster labels to all the other instances in the same cluster

    # fit a new model with new y
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, y_train_propagated)
    print('same label in one cluster score: ', log_reg.score(X_test, y_test))

    # method 3:
    # Giving the cluster label to ALL other instances in the same cluster is NOT a good idea. So give it to 20% cloest.
    percentile_closest = 20

    X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]  # np and labels pointers
    # return: [30.3917992  20.3734662  15.08582969 ... 19.36276495 19.5626378 18.23619458]
    for i in range(k):  # k: 50
        in_cluster = (kmeans.labels_ == i)  # return a [False False False ... False False False]
        cluster_dist = X_cluster_dist[in_cluster]  # get the digits in cluster
        cutoff_distance = np.percentile(cluster_dist, percentile_closest)  # get the 20% distance to cluster center
        above_cutoff = (X_cluster_dist > cutoff_distance)  # out of the cluster
        X_cluster_dist[in_cluster & above_cutoff] = -1

    # make data set
    partially_propagated = (X_cluster_dist != -1)
    X_train_partially_propagated = X_train[partially_propagated]
    y_train_partially_propagated = y_train_propagated[partially_propagated]

    # train a new model
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
    print('20% label score: ', log_reg.score(X_test, y_test))  # 0.9422222222222222 the best score

    print(np.mean(y_train_partially_propagated == y_train[partially_propagated]))  # 0.9896907216494846
    # the best score due to the propagated labels are actually pretty good: their accuracy is very close to 99%
