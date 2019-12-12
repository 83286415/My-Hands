# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import warnings
import time

# Chapter import
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.metrics import mean_squared_error


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

    # Kernel PCA

    # Data set
    X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)  # X: [n_samples, x, y, z]

    # build models: linear kernel, rbf kernel and sigmoid kernel PCA models
    lin_pca = KernelPCA(n_components=2, kernel="linear", fit_inverse_transform=True)
    rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
    sig_pca = KernelPCA(n_components=2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)
    # fit_inverse_transform: Learn the inverse transform for non-precomputed kernels.

    # plot
    y = t > 6.9

    plt.figure(figsize=(11, 4))
    for subplot, pca, title in ((131, lin_pca, "Linear kernel"), (132, rbf_pca, "RBF kernel, $\gamma=0.04$"),
                                (133, sig_pca, "Sigmoid kernel, $\gamma=10^{-3}, r=1$")):
        X_reduced = pca.fit_transform(X)
        if subplot == 132:
            X_reduced_rbf = X_reduced

        plt.subplot(subplot)
        plt.title(title, fontsize=14)
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
        plt.xlabel("$z_1$", fontsize=18)
        if subplot == 131:
            plt.ylabel("$z_2$", fontsize=18, rotation=0)
        plt.grid(True)

    save_fig("kernel_pca_plot")
    # plt.show()

    # Choose a Kernel PCA and adjust its super-parameters

    # make a pipeline classifier
    clf = Pipeline([
        ("kpca", KernelPCA(n_components=2)),
        ("log_reg", LogisticRegression())
    ])

    # Grid SearchCV
    param_grid = [{
        "kpca__gamma": np.linspace(0.03, 0.05, 10),
        "kpca__kernel": ["rbf", "sigmoid"]
    }]

    grid_search = GridSearchCV(clf, param_grid, cv=3)  # classifier input
    grid_search.fit(X, y)
    '''GridSearchCV(cv=3, error_score='raise',
       estimator=Pipeline(memory=None,
     steps=[('kpca', KernelPCA(alpha=1.0, coef0=1, copy_X=True, degree=3, eigen_solver='auto',
     fit_inverse_transform=False, gamma=None, kernel='linear',
     kernel_params=None, max_iter=None, n_components=2, n_jobs=1,
     random_state=None, remove_zero_eig=False, tol=0)), ('log_reg', LogisticRegre...ty='l2', random_state=None, 
     solver='liblinear', tol=0.0001, verbose=0, warm_start=False))]),
       fit_params=None, iid=True, n_jobs=1,
       param_grid=[{'kpca__gamma': array([0.03   , 0.03222, 0.03444, 0.03667, 0.03889, 0.04111, 0.04333,
       0.04556, 0.04778, 0.05   ]), 'kpca__kernel': ['rbf', 'sigmoid']}],
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring=None, verbose=0)'''
    print(grid_search.best_params_)  # {'kpca__gamma': 0.043333333333333335, 'kpca__kernel': 'rbf'}

    rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.0433,
                        fit_inverse_transform=True)  # best_params_
    X_reduced = rbf_pca.fit_transform(X)
    X_preimage = rbf_pca.inverse_transform(X_reduced)
    print(mean_squared_error(X, X_preimage))  # 32.78630879576616