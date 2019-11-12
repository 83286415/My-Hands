# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import warnings

# Chapter import
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor
from deslib.des.knora_e import KNORAE
from sklearn.decomposition import PCA

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

    # Principal Components

    # data set
    np.random.seed(4)
    m = 60
    w1, w2 = 0.1, 0.3
    noise = 0.1

    angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
    X = np.empty((m, 3))  # return a empty array with shape (60, 3)
    X[:, 0] = np.cos(angles) + np.sin(angles) / 2 + noise * np.random.randn(m) / 2
    X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
    X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

    # find PC with numpy
    X_centered = X - X.mean(axis=0)  # data set centralization
    U, s, Vt = np.linalg.svd(X_centered)  # get the PC with svd()
    c1 = Vt.T[:, 0]
    c2 = Vt.T[:, 1]

    # Projecting to the Super Plane

    # S is the singular value matrix of X. refer to svd() in cloud note.
    m, n = X.shape
    S = np.zeros(X_centered.shape)
    S[:n, :n] = np.diag(s)

    if np.allclose(X_centered, U.dot(S).dot(Vt)):  # return True if X_centered == U.dot(S).dot(Vt)
        # compute the Wd matrix
        W2 = Vt.T[:, :2]  # Wd and d==2, the first 2 columns of Vt matrix

        # compute the Xd-proj, that is the X data set projected on super plane d
        X2D = X_centered.dot(W2)
        X2D_using_svd = X2D
        print(X2D_using_svd[:5])  # this is the Xd-proj
        # [[-1.26203346 -0.42067648]
        #  [ 0.08001485  0.35272239]
        #  [-1.17545763 -0.36085729]
        #  [-0.89305601  0.30862856]
        #  [-0.73016287  0.25404049]]

    # PCA using SK-learn

    # # compute the Xd-proj
    pca = PCA(n_components=2)
    X2D = pca.fit_transform(X)  # pca makes the X centered automatically
    print(X2D[:5])
    # [[ 1.26203346  0.42067648]
    #  [-0.08001485 -0.35272239]
    #  [ 1.17545763  0.36085729]
    #  [ 0.89305601 -0.30862856]
    #  [ 0.73016287 -0.25404049]]

    if np.allclose(X2D, -X2D_using_svd):  # different +/-, but it doesn't matter at all!
        print('SK-learn and SVD get the same projection')
        print(pca.components_)  # components_: show the pc(axis array) on the super plane
        # [[-0.93636116 -0.29854881 -0.18465208]
        #  [ 0.34027485 -0.90119108 -0.2684542 ]]

    # Recover the 3D points projected on the plane (PCA 2D subspace) with PCA
    X3D_inv = pca.inverse_transform(X2D)
    print('Are the recovered 3D points exactly equal to the original 3D points? ', np.allclose(X3D_inv, X))  # False
    # 3D points recovered are not exactly eaqual to the original 3D points for some info loss in projection step
    # so compute the MS:
    print(np.mean(np.sum(np.square(X3D_inv - X), axis=1)))  # 0.010170337792848549 a tiny mean error

    # Recover the 3D points projected on the plane (PCA 2D subspace) with SVD
    X3D_inv_using_svd = X2D_using_svd.dot(Vt[:2, :])
    print('It should be equal to the original 3D points: ', np.allclose(X3D_inv_using_svd, X3D_inv - pca.mean_))  # True

    # Explained Variance Ratio

    print(pca.explained_variance_ratio_)  # [0.84248607 0.14631839] 84.2% variance on the first axis, only 1.2% on 3rd

    # Chooosing the Right Number of Dimensions

    pca_1st = PCA(n_components=0.84)  # keep the first dimension (component, feature, axis) whose variance ratio=0.8425
    X2D_1st = pca_1st.fit_transform(X)
    print(X2D_1st[:5])  # Compared with the X2D above, only the first dimension is kept
    # [[ 1.26203346]
    #  [-0.08001485]
    #  [ 1.17545763]
    #  [ 0.89305601]
    #  [ 0.73016287]]

    # PCA Compression
    # MNIST compression