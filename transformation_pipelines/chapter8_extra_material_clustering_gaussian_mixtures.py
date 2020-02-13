# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import warnings
import time

# Chapter import
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from matplotlib.colors import LogNorm
from sklearn.mixture import BayesianGaussianMixture
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


def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=50, linewidths=50,
                color=cross_color, zorder=11, alpha=1)


def plot_gaussian_mixture(clusterer, X, resolution=1000, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = -clusterer.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z,
                 norm=LogNorm(vmin=1.0, vmax=30.0),
                 levels=np.logspace(0, 2, 12))
    plt.contour(xx, yy, Z,
                norm=LogNorm(vmin=1.0, vmax=30.0),
                levels=np.logspace(0, 2, 12),
                linewidths=1, colors='k')

    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z,
                linewidths=2, colors='r', linestyles='dashed')

    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    plot_centroids(clusterer.means_, clusterer.weights_)

    plt.xlabel("$x_1$", fontsize=14)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft='off')


def compare_gaussian_mixtures(gm1, gm2, X):
    plt.figure(figsize=(9, 4))

    plt.subplot(121)
    plot_gaussian_mixture(gm1, X)
    plt.title('covariance_type="{}"'.format(gm1.covariance_type), fontsize=14)

    plt.subplot(122)
    plot_gaussian_mixture(gm2, X, show_ylabels=False)
    plt.title('covariance_type="{}"'.format(gm2.covariance_type), fontsize=14)


def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)


if __name__ == '__main__':

    # Gaussian Mixtures

    # data set
    X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
    X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))  # dot returns array not a single value because the [[],[]]
    X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
    X2 = X2 + [6, -8]
    X = np.r_[X1, X2]
    y = np.r_[y1, y2]

    # build model
    gm = GaussianMixture(n_components=3, n_init=10, random_state=42)  # 3 clusters; n_init: train from 10 clusters
    # parameter: covariance_type = full(default), tied, spherical, diag. detailed info refer to EM part in cloud note.
    # usage of this covariance_type, refer to "# compare" below.

    gm.fit(X)
    print('weight: ', gm.weights_)  # The weights of each mixture cluster
    print('means: ', gm.means_)  # the mean of each mixture cluster, not the center (refer to EM in cloud note)
    print('covariances_: ', gm.covariances_)  # The covariance of each cluster  (refer to EM in cloud note)
    print('converged or not: ', gm.converged_)  # True
    print('iteration #: ', gm.n_iter_)  # How many iterations did it take

    print('predict X: ', gm.predict(X))  # array([2, 2, 1, ..., 0, 0, 0])
    print('predict the probability: ', gm.predict_proba(X))
    '''
    array([[2.32389467e-02, 6.77397850e-07, 9.76760376e-01],
       [1.64685609e-02, 6.75361303e-04, 9.82856078e-01],
       [2.01535333e-06, 9.99923053e-01, 7.49319577e-05],
       ...,
       [9.99999571e-01, 2.13946075e-26, 4.28788333e-07],
       [1.00000000e+00, 1.46454409e-41, 5.12459171e-16],
       [1.00000000e+00, 8.02006365e-41, 2.27626238e-15]])'''

    # new sample
    X_new, y_new = gm.sample(6)  # Generate random samples from the fitted Gaussian distribution
    print('X new: ', X_new)
    '''  [[ 2.95400315  2.63680992]
         [-1.16654575  1.62792705]
         [-1.39477712 -1.48511338]
         [ 0.27221525  0.690366  ]
         [ 0.54095936  0.48591934]
         [ 0.38064009 -0.56240465]]'''
    print('y new: ', y_new)  # [0 1 2 2 2 2]
    score_samples = gm.score_samples(X_new)  # Compute the weighted log probabilities for each sample
    print('score samples: ', score_samples)  # [-4.80269976 -2.01524058 -3.51660935 -2.22880963 -2.18454233 -3.82874339]

    # plot
    plt.figure(figsize=(8, 4))
    plot_gaussian_mixture(gm, X)
    save_fig("gaussian_mixtures_diagram")
    # plt.show()

    # compare GaussianMixture's covariance_type ,which can refer to EM part in cloud note.
    gm_full = GaussianMixture(n_components=3, n_init=10, covariance_type="full", random_state=42)  # full (default type)
    gm_tied = GaussianMixture(n_components=3, n_init=10, covariance_type="tied", random_state=42)  # tied
    gm_spherical = GaussianMixture(n_components=3, n_init=10, covariance_type="spherical", random_state=42)  # spherical
    gm_diag = GaussianMixture(n_components=3, n_init=10, covariance_type="diag", random_state=42)  # diag
    gm_full.fit(X)
    gm_tied.fit(X)
    gm_spherical.fit(X)
    gm_diag.fit(X)

    compare_gaussian_mixtures(gm_tied, gm_spherical, X)
    save_fig("covariance_type_diagram_of_tied_and_spherical")
    # plt.show()

    compare_gaussian_mixtures(gm_full, gm_diag, X)
    save_fig("covariance_type_diagram_of_full_and_diag")
    # plt.tight_layout()
    # plt.show()

    # Variational Bayesian Gaussian Mixtures

    # build model
    bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)  # components=10 must > clusters' count
    bgm.fit(X)
    print(np.round(bgm.weights_, 2))  # np.round() refer to cloud note
    # array([0.4 , 0.21, 0.4 , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]) So only 3 clusters in.

    # plot
    plt.figure(figsize=(8, 5))
    plot_gaussian_mixture(bgm, X)
    # plt.show()

    # build new models with different parameters
    bgm_low = BayesianGaussianMixture(n_components=10, max_iter=1000, n_init=1,
                                      weight_concentration_prior=0.01, random_state=42)  # 0.01
    bgm_high = BayesianGaussianMixture(n_components=10, max_iter=1000, n_init=1,
                                       weight_concentration_prior=10000, random_state=42)  # 10000, more clusters
    '''
    The higher concentration puts more mass in the center and will lead to more components being active, while a lower 
    concentration parameter will lead to more mass at the edge of the simplex.
    '''

    nn = 200
    bgm_low.fit(X[:nn])
    bgm_high.fit(X[:nn])

    print(np.round(bgm_low.weights_, 2))  # [0.56 0.44 0.   0.   0.   0.   0.   0.   0.   0.  ]  , 2 clusters
    print(np.round(bgm_high.weights_, 2))  # [0.21 0.42 0.01 0.01 0.01 0.01 0.01 0.12 0.22 0.01]  , 4 clusters

    # plot
    plt.figure(figsize=(9, 4))

    plt.subplot(121)
    plot_gaussian_mixture(bgm_low, X[:nn])
    plt.title("weight_concentration_prior = 0.01", fontsize=14)

    plt.subplot(122)
    plot_gaussian_mixture(bgm_high, X[:nn], show_ylabels=False)
    plt.title("weight_concentration_prior = 10000", fontsize=14)

    save_fig("mixture_concentration_prior_diagram")
    # plt.show()

    # practise
    X_moons, y_moons = make_moons(n_samples=1000, noise=0.05, random_state=42)

    bgm = BayesianGaussianMixture(n_components=12, max_iter=1000, n_init=1,
                                  weight_concentration_prior_type='dirichlet_distribution',
                                  weight_concentration_prior=0.0001, random_state=42)
    bgm.fit(X_moons)

    plt.figure(figsize=(9, 3.2))

    plt.subplot(121)
    plot_data(X_moons)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)

    plt.subplot(122)
    plot_gaussian_mixture(bgm, X_moons, show_ylabels=False)

    save_fig("moons_vs_bgm_diagram")
    plt.show()