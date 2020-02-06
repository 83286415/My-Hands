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

    # data set
    X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
    X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))  # dot returns array not a single value because the [[],[]]
    X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
    X2 = X2 + [6, -8]
    X = np.r_[X1, X2]
    y = np.r_[y1, y2]

    # build model
    gm = GaussianMixture(n_components=3, n_init=10, random_state=42)  # 3 clusters; n_init: train from 10 clusters
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