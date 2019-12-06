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
from sklearn.decomposition import IncrementalPCA

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


def plot_digits(instances, images_per_row=5, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=matplotlib.cm.binary, **options)
    plt.axis("off")


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

    # load data set
    mnist = fetch_mldata('MNIST original')
    X = mnist["data"]
    y = mnist["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # PCA compression
    pca = PCA(n_components=0.95)
    pca.fit(X_train)
    cumsum = np.cumsum(pca.explained_variance_ratio_)  # cumsum: the sum of the explained variance ratio of first axises
    d = np.argmax(cumsum >= 0.95) + 1  # +1: because the index starts from 0
    print(d)  # 154: only 154 dimensions are kept from 28*28 dimensions in original data
    print(pca.components_)  # a 154 features list: 154 features are kept, others are abandoned.
    print(np.sum(pca.explained_variance_ratio_))  # 0.9504463030200186

    # PCA Inverse transform
    pca = PCA(n_components=154)
    X_reduced = pca.fit_transform(X_train)
    X_recovered = pca.inverse_transform(X_reduced)

    # plot
    plt.figure(figsize=(7, 4))
    plt.subplot(121)
    plot_digits(X_train[::2100])  # pick one number's image from all images with the step: 2100 images
    plt.title("Original", fontsize=16)
    plt.subplot(122)
    plot_digits(X_recovered[::2100])
    plt.title("Compressed", fontsize=16)

    save_fig("mnist_compression_plot")

    # PCA Incremental

    # partial fit each sub MNIST data array to save memory
    n_batches = 100
    inc_pca = IncrementalPCA(n_components=154)  # train a incremental PCA model
    for X_batch in np.array_split(X_train, n_batches):  # save memory but partial fit n_batches times: maybe longer time
        print(".", end="")  # not shown in the book
        inc_pca.partial_fit(X_batch)  # not fit the whole train data set

    # transform the whole data set with the model above
    X_reduced_inc_pca = inc_pca.transform(X_train)
    X_recovered_inc_pca = inc_pca.inverse_transform(X_reduced_inc_pca)

    # verify
    print(np.allclose(pca.mean_, inc_pca.mean_))  # True: the mean values of two models are equal
    print(np.allclose(X_reduced_inc_pca, X_reduced))  # False: the results are different - inc PCA not perfect

    # plot
    plt.figure(figsize=(7, 4))
    plt.subplot(121)
    plot_digits(X_train[::2100])
    plt.subplot(122)
    plot_digits(X_recovered_inc_pca[::2100])
    plt.tight_layout()

    save_fig("mnist_incremental_pca_plot")

    # Using memmap() to read array file on disk

    # save X_train to disk
    filename = os.path.join(ROOT_PATH, 'Model saved', "my_mnist.data")
    m, n = X_train.shape

    X_mm = np.memmap(filename, dtype='float32', mode='write', shape=(m, n))  # filename: must be a path
    X_mm[:] = X_train
    del X_mm  # deleting the X_mm object, that is the memmap() class object,  will save X_mm to disk

    # read memory file from disk
    X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m, n))  # make a read only model

    batch_size = m // n_batches
    inc_pca_mem = IncrementalPCA(n_components=154, batch_size=batch_size)
    inc_pca_mem.fit(X_mm)
    # IncrementalPCA(batch_size=525, copy=True, n_components=154, whiten=False)
    X_reduced_inc_pca_mem = inc_pca_mem.transform(X_train)
    X_recovered_inc_pca = inc_pca.inverse_transform(X_reduced_inc_pca_mem)

    # verify
    print(np.allclose(pca.mean_, inc_pca_mem.mean_))  # True: the mean values of two models are equal
    print(np.allclose(X_reduced_inc_pca_mem, X_reduced))  # False: the results are different - inc PCA not perfect

    # plot
    plt.figure(figsize=(7, 4))
    plt.subplot(121)
    plot_digits(X_train[::2100])
    plt.subplot(122)
    plot_digits(X_recovered_inc_pca[::2100])
    plt.tight_layout()

    save_fig("mnist_incremental_pca_mem_plot")

    # Randomized PCA
