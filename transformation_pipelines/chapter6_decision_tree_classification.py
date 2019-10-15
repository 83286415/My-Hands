# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# Chapter import
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
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
CHAPTER_ID = "decision_trees"


def save_fig(fig_id, tight_layout=True):
    path = image_path(fig_id) + ".png"
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)  # cannot save file if path doesn't exist


def image_path(fig_id):
    return os.path.join(ROOT_PATH, "images", CHAPTER_ID, fig_id)


def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris=True, legend=False, plot_training=True):
    # mutable parameters warning: https://blog.csdn.net/tcx1992/article/details/81312446
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if not iris:
        custom_cmap2 = ListedColormap(['#7d7d58', '#4c4c7f', '#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris-Setosa")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris-Versicolor")
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris-Virginica")
        plt.axis(axes)
    if iris:
        plt.xlabel("Petal length", fontsize=14)
        plt.ylabel("Petal width", fontsize=14)
    else:
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    if legend:
        plt.legend(loc="lower right", fontsize=14)


if __name__ == '__main__':

    # Training, Visualizing and Making Predictions

    # data set
    iris = load_iris()
    X = iris.data[:, 2:]  # petal length and width;    X.shape: (150, 2)
    y = iris.target  # all three kinds of iris

    # build model
    tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    tree_clf.fit(X, y)
    '''
    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='best')
    '''
    print(tree_clf.feature_importances_)

    # export_graphviz: to generate a dot file to visualize the decision tree model trained
    export_graphviz(
        tree_clf,
        out_file=image_path("iris_tree.dot"),
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,       # When set to ``True``, draw node boxes with rounded corners
        filled=True         # When set to ``True``, paint nodes to indicate majority class
    )

    # plot
    plt.figure(figsize=(8, 4))
    plot_decision_boundary(tree_clf, X, y)
    plt.plot([2.45, 2.45], [0, 3], "k-", linewidth=2)
    plt.plot([2.45, 7.5], [1.75, 1.75], "k--", linewidth=2)
    plt.plot([4.95, 4.95], [0, 1.75], "k:", linewidth=2)
    plt.plot([4.85, 4.85], [1.75, 3], "k:", linewidth=2)
    plt.text(1.40, 1.0, "Depth=0", fontsize=15)
    plt.text(3.2, 1.80, "Depth=1", fontsize=13)
    plt.text(4.05, 0.5, "(Depth=2)", fontsize=11)

    save_fig("decision_tree_decision_boundaries_plot")
    # plt.show()

    # Estimating Class Probabilities

    # make prediction and probability
    one_sample_predicted_proba = tree_clf.predict_proba([[5, 1.5]])
    one_sample_predicted = tree_clf.predict([[5, 1.5]])
    print('The probability of one sample with 5cm petal length and 1.5cm petal width: ', one_sample_predicted_proba)
    # [[0.         0.90740741 0.09259259]]
    print('The prediction of one sample with 5cm petal length and 1.5cm petal width: ', one_sample_predicted)  # [1]

    # Sensitivity to training set details

    # moon data
    Xm, ym = make_moons(n_samples=100, noise=0.25, random_state=53)

    # build model
    deep_tree_clf1 = DecisionTreeClassifier(random_state=42)
    deep_tree_clf2 = DecisionTreeClassifier(min_samples_leaf=4, random_state=42)  # add restriction
    deep_tree_clf1.fit(Xm, ym)
    deep_tree_clf2.fit(Xm, ym)

    # plot
    plt.figure(figsize=(11, 4))
    plt.subplot(121)
    plot_decision_boundary(deep_tree_clf1, Xm, ym, axes=[-1.5, 2.5, -1, 1.5], iris=False)
    plt.title("No restrictions", fontsize=16)
    plt.subplot(122)
    plot_decision_boundary(deep_tree_clf2, Xm, ym, axes=[-1.5, 2.5, -1, 1.5], iris=False)
    plt.title("min_samples_leaf = {}".format(deep_tree_clf2.min_samples_leaf), fontsize=14)

    save_fig("min_samples_leaf_plot")
    # plt.show()

    # Another case

    np.random.seed(6)
    Xs = np.random.rand(100, 2) - 0.5  # shape: (100, 2), values in [0, 1)-0.5
    ys = (Xs[:, 0] > 0).astype(np.float32) * 2  # (Xs[:, 0] > 0) returns True or False, if True, True*2 == 2

    angle = np.pi / 4  # pi/4
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    # [[0.70710678 - 0.70710678] [0.70710678  0.70710678]]
    Xsr = Xs.dot(rotation_matrix)

    tree_clf_s = DecisionTreeClassifier(random_state=42)
    tree_clf_s.fit(Xs, ys)
    tree_clf_sr = DecisionTreeClassifier(random_state=42)
    tree_clf_sr.fit(Xsr, ys)

    plt.figure(figsize=(11, 4))
    plt.subplot(121)
    plot_decision_boundary(tree_clf_s, Xs, ys, axes=[-0.7, 0.7, -0.7, 0.7], iris=False)
    plt.subplot(122)
    plot_decision_boundary(tree_clf_sr, Xsr, ys, axes=[-0.7, 0.7, -0.7, 0.7], iris=False)

    save_fig("sensitivity_to_rotation_plot")
    plt.show()