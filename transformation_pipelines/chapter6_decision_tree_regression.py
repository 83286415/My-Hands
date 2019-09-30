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
from sklearn.tree import DecisionTreeRegressor

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


def plot_regression_predictions(tree_reg, X, y, axes=[0, 1, -0.2, 1], ylabel="$y$"):
    x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
    y_pred = tree_reg.predict(x1)
    plt.axis(axes)
    plt.xlabel("$x_1$", fontsize=18)
    if ylabel:
        plt.ylabel(ylabel, fontsize=18, rotation=0)
    plt.plot(X, y, "b.")  # plot X points in blue
    plt.plot(x1, y_pred, "r.-", linewidth=2, label=r"$\hat{y}$")  # red horizon lines: the average values


if __name__ == '__main__':

    # Regression

    # Quadratic training set + noise
    np.random.seed(42)
    m = 200
    X = np.random.rand(m, 1)
    y = 4 * (X - 0.5) ** 2
    y = y + np.random.randn(m, 1) / 10

    # build models
    tree_reg1 = DecisionTreeRegressor(random_state=42, max_depth=2)  # Regressor not Classifier
    tree_reg2 = DecisionTreeRegressor(random_state=42, max_depth=3)
    tree_reg1.fit(X, y)
    tree_reg2.fit(X, y)
    '''
    DecisionTreeRegressor(criterion='mse', max_depth=2, max_features=None,
           max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           presort=False, random_state=42, splitter='best')
    '''

    # plot
    plt.figure(figsize=(11, 4))
    plt.subplot(121)
    plot_regression_predictions(tree_reg1, X, y)
    for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
        plt.plot([split, split], [-0.2, 1], style, linewidth=2)  # plot vertical black lines for each depth layer
    plt.text(0.21, 0.65, "Depth=0", fontsize=15)
    plt.text(0.01, 0.2, "Depth=1", fontsize=13)
    plt.text(0.65, 0.8, "Depth=1", fontsize=13)
    plt.legend(loc="upper center", fontsize=18)  # the location of parameter label in plot()
    plt.title("max_depth=2", fontsize=14)

    plt.subplot(122)
    plot_regression_predictions(tree_reg2, X, y, ylabel=None)
    for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
        plt.plot([split, split], [-0.2, 1], style, linewidth=2)
    for split in (0.0458, 0.1298, 0.2873, 0.9040):
        plt.plot([split, split], [-0.2, 1], "k:", linewidth=1)
    plt.text(0.3, 0.5, "Depth=2", fontsize=13)
    plt.title("max_depth=3", fontsize=14)

    save_fig("tree_regression_plot")
    # plt.show()

    # export_graphviz: to generate a dot file to visualize the decision tree model trained
    export_graphviz(
        tree_reg1,
        out_file=image_path("regression_tree.dot"),
        feature_names=["x1"],
        rounded=True,
        filled=True
    )

    # Restrictions on decision tree regression model

    # build models with/without restriction
    tree_reg1 = DecisionTreeRegressor(random_state=42)
    tree_reg2 = DecisionTreeRegressor(random_state=42, min_samples_leaf=10)
    tree_reg1.fit(X, y)  # y is a quadratic function of X
    tree_reg2.fit(X, y)

    # data
    x1 = np.linspace(0, 1, 500).reshape(-1, 1)
    y_pred1 = tree_reg1.predict(x1)
    y_pred2 = tree_reg2.predict(x1)

    # plot
    plt.figure(figsize=(11, 4))

    plt.subplot(121)  # over-fit model without restriction
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred1, "r.-", linewidth=2, label=r"$\hat{y}$")  # $\hat{y}$      a y with a red hat on it
    plt.axis([0, 1, -0.2, 1.1])
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", fontsize=18, rotation=0)
    plt.legend(loc="upper center", fontsize=18)
    plt.title("No restrictions", fontsize=14)

    plt.subplot(122)
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred2, "r.-", linewidth=2, label=r"$\hat{y}$")
    plt.axis([0, 1, -0.2, 1.1])
    plt.xlabel("$x_1$", fontsize=18)
    plt.title("min_samples_leaf={}".format(tree_reg2.min_samples_leaf), fontsize=14)

    save_fig("tree_regression_regularization_plot")
    plt.show()