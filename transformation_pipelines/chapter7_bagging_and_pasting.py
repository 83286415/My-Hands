# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# Chapter import
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error

# To plot pretty figures
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
WORKING_PATH = os.path.abspath(os.path.join(os.getcwd(), '..'))
ROOT_PATH = os.path.join(WORKING_PATH, 'Hands on SK and TS\\')
CHAPTER_ID = "ensembles"


def save_fig(fig_id, tight_layout=True):
    path = image_path(fig_id) + ".png"
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)  # cannot save file if path doesn't exist


def image_path(fig_id):
    return os.path.join(ROOT_PATH, "images", CHAPTER_ID, fig_id)


def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)


if __name__ == '__main__':

    # Bagging ensembles

    # moon data set
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # build bagging classifier with decision trees classifier
    bag_clf = BaggingClassifier(DecisionTreeClassifier(random_state=42), n_estimators=500,
                                max_samples=100, bootstrap=True, n_jobs=-1, random_state=42)
    # n_estimators: 500 trees
    # max_sample: 100 samples in one tree
    # bootstrap = True: bagging     = False: pasting
    # n_jobs: CPU cores working on this models      n_jobs = -1: all idle cores will work on this model

    bag_clf.fit(X_train, y_train)
    y_pred = bag_clf.predict(X_test)
    print('bagging accuracy score: ', accuracy_score(y_test, y_pred))  # 0.904 better than single decision tree below
    print('bagging RMSE: ', np.sqrt(mean_squared_error(y_test, y_pred)))  # 0.30983866769659335

    # build a decision tree model as contrast
    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train)
    y_pred_tree = tree_clf.predict(X_test)
    print('decision tree accuracy score: ', accuracy_score(y_test, y_pred_tree))  # 0.856
    print('tree RMSE: ', np.sqrt(mean_squared_error(y_test, y_pred_tree)))  # 0.3794733192202055  the worst

    # build pasting model
    past_clf = BaggingClassifier(DecisionTreeClassifier(random_state=42), n_estimators=500,
                                 max_samples=100, bootstrap=False, n_jobs=-1, random_state=42)
    past_clf.fit(X_train, y_train)
    y_pred_past = past_clf.predict(X_test)
    print('pasting accuracy score: ', accuracy_score(y_test, y_pred_past))  # 0.912 the best one of all!
    print('pasting RMSE: ', np.sqrt(mean_squared_error(y_test, y_pred_past)))  # 0.2966479394838265

    # plot
    plt.figure(figsize=(17, 4))
    plt.subplot(131)  # tree
    plot_decision_boundary(tree_clf, X, y)
    plt.title("Decision Tree", fontsize=14)
    plt.subplot(132)  # bagging
    plot_decision_boundary(bag_clf, X, y)
    plt.title("Decision Trees with Bagging", fontsize=14)
    plt.subplot(133)  # pasting
    plot_decision_boundary(past_clf, X, y)
    plt.title("Decision Trees with Pasting", fontsize=14)
    save_fig("decision_tree_without_and_with_bagging_and_pasting_plot")
    plt.show()
