# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

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


def plot_digit(data):
    image = data.reshape(28, 28)  # reshape (70000, 784) into (28, 28)
    plt.imshow(image, cmap=matplotlib.cm.hot, interpolation="nearest")  # cmap: color map;
    # interpolation: refer to cloud note matplotlib
    plt.axis("off")  # no axis


def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)  # red line's x
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center", fontsize=16)
    plt.axis(axes)


if __name__ == '__main__':

    # Adaptive Boosting

    # moon data
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # build AdaBoost model
    ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), n_estimators=200,
        algorithm="SAMME.R", learning_rate=0.5, random_state=42)
    '''AdaBoostClassifier(algorithm='SAMME.R',
          base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=1,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best'),
          learning_rate=0.5, n_estimators=200, random_state=42)'''
    ada_clf.fit(X_train, y_train)

    # plot the ada boost classifier decision boundary
    plot_decision_boundary(ada_clf, X, y)

    # plot SVC decision boundary
    m = len(X_train)
    plt.figure(figsize=(11, 4))
    for subplot, learning_rate in ((121, 1), (122, 0.5)):
        sample_weights = np.ones(m)
        plt.subplot(subplot)
        for i in range(5):
            svm_clf = SVC(kernel="rbf", C=0.05, random_state=42)
            svm_clf.fit(X_train, y_train, sample_weight=sample_weights)
            y_pred = svm_clf.predict(X_train)
            sample_weights[y_pred != y_train] *= (1 + learning_rate)
            plot_decision_boundary(svm_clf, X, y, alpha=0.2)
            plt.title("learning_rate = {}".format(learning_rate), fontsize=16)
        if subplot == 121:
            plt.text(-0.7, -0.65, "1", fontsize=14)
            plt.text(-0.6, -0.10, "2", fontsize=14)
            plt.text(-0.5, 0.10, "3", fontsize=14)
            plt.text(-0.4, 0.55, "4", fontsize=14)
            plt.text(-0.3, 0.90, "5", fontsize=14)
    save_fig("boosting_plot")

    # plot random forests: much better
    plt.figure(figsize=(11, 4))
    for subplot, learning_rate in ((121, 1), (122, 0.5)):
        sample_weights = np.ones(m)
        plt.subplot(subplot)
        for i in range(5):
            rnd_clf = RandomForestClassifier(n_estimators=200, max_leaf_nodes=16, n_jobs=-1, random_state=42)
            rnd_clf.fit(X_train, y_train, sample_weight=sample_weights)
            y_pred = rnd_clf.predict(X_train)
            sample_weights[y_pred != y_train] *= (1 + learning_rate)
            plot_decision_boundary(rnd_clf, X, y, alpha=0.2)
            plt.title("learning_rate = {}".format(learning_rate), fontsize=16)
        if subplot == 121:
            plt.text(-0.7, -0.65, "1", fontsize=14)
            plt.text(-0.6, -0.10, "2", fontsize=14)
            plt.text(-0.5, 0.10, "3", fontsize=14)
            plt.text(-0.4, 0.55, "4", fontsize=14)
            plt.text(-0.3, 0.90, "5", fontsize=14)

    # plt.show()

    # Gradient Boosting Regression Trees - GBRT: a regression model

    # data set
    np.random.seed(42)
    X = np.random.rand(100, 1) - 0.5  # X: [[-0.12545988], [ 0.45071431], [ 0.23199394], ...]
    y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)

    # build gbrt model
    gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)
    '''
    GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=1.0, loss='ls', max_depth=2, max_features=None,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=3, presort='auto', random_state=42,
             subsample=1.0, verbose=0, warm_start=False)
    '''
    gbrt.fit(X, y)

    # build a shrinkage GBRT model with more estimators and low learning rate
    gbrt_slow = GradientBoostingRegressor(max_depth=2, n_estimators=200, learning_rate=0.1, random_state=42)
    gbrt_slow.fit(X, y)  # more base regressors and lower learning rate

    # plot these two different regressors
    plt.figure(figsize=(11, 4))

    plt.subplot(121)
    plot_predictions([gbrt], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="Ensemble predictions")
    plt.title("learning_rate={}, n_estimators={}".format(gbrt.learning_rate, gbrt.n_estimators), fontsize=14)

    plt.subplot(122)
    plot_predictions([gbrt_slow], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
    plt.title("learning_rate={}, n_estimators={}".format(gbrt_slow.learning_rate, gbrt_slow.n_estimators), fontsize=14)

    save_fig("gbrt_learning_rate_plot")
    # plt.show()

    # Gradient Boosting with Early stopping
    # With staged_predict(), find the best parameter and re-build model to fit

    # data set
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=49)

    # build GBRT model
    gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)
    gbrt.fit(X_train, y_train)

    # find the best estimator
    errors = [mean_squared_error(y_val, y_pred) for y_pred in gbrt.staged_predict(X_val)]  # len(errors) = 120
    # staged_predict(X) return a iterator which returns the prediction of each stage.
    # stage1: estimator1; stage2: estimator 1+2; stage3: estimator 2+3...
    bst_n_estimators = np.argmin(errors)  # bst_n_estimators = 55; argmin: return the index of the min value in list

    # re-build GBRT model with best estimator number parameter
    gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators, random_state=42)
    '''
    GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=2, max_features=None,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=55, presort='auto', random_state=42,
             subsample=1.0, verbose=0, warm_start=False)
    '''  # n_estimators=55
    gbrt_best.fit(X_train, y_train)

    # plot
    min_error = np.min(errors)  # min_error =  # 0.002712853325235463; min: return the min value in the list
    plt.figure(figsize=(11, 4))

    plt.subplot(121)
    plt.plot(errors, "b.-")
    plt.plot([bst_n_estimators, bst_n_estimators], [0, min_error], "k--")
    plt.plot([0, 120], [min_error, min_error], "k--")
    plt.plot(bst_n_estimators, min_error, "ko")
    plt.text(bst_n_estimators, min_error * 1.2, "Minimum", ha="center", fontsize=14)
    plt.axis([0, 120, 0, 0.01])
    plt.xlabel("Number of trees")
    plt.title("Validation error", fontsize=14)

    plt.subplot(122)
    plot_predictions([gbrt_best], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
    plt.title("Best model (%d trees)" % bst_n_estimators, fontsize=14)

    save_fig("early_stopping_gbrt_plot")
    plt.show()

    # Gradient Boosting with Early stopping
    # Traditional early stopping with MSE value going up in 5 successive stages
    gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True, random_state=42)
    # warm_start: keep the snap of the fitting and allow to specify the n_estimator before the next fitting round

    min_val_error = float("inf")
    error_going_up = 0
    for n_estimators in range(1, 120):
        gbrt.n_estimators = n_estimators  # stopped at n_estimators = 61; So the 61-5-1=55 is the best estimator
        gbrt.fit(X_train, y_train)
        y_pred = gbrt.predict(X_val)
        val_error = mean_squared_error(y_val, y_pred)
        if val_error < min_val_error:
            min_val_error = val_error
            error_going_up = 0
        else:
            error_going_up += 1
            if error_going_up == 5:  # if MSE value going up in 5 successive stages
                print(gbrt.n_estimators)  # 61 and 61-5-1=55 is the best n_estimator
                print("Minimum validation MSE:", min_val_error)  # 0.002712853325235463 is the same as the model above
                break  # early stopping

    # XGBoost
    # not shown in the book

    if False:  # cannot run this code for dll file problem.
        try:
            import xgboost
            print('importing XGBoost')
        except ImportError as ex:
            print("Error: the xgboost library is not installed.")
            xgboost = None

        if xgboost is not None:
            xgb_reg = xgboost.XGBRegressor(random_state=42)
            xgb_reg.fit(X_train, y_train)
            y_pred = xgb_reg.predict(X_val)
            val_error = mean_squared_error(y_val, y_pred)
            print("Validation MSE:", val_error)

        if xgboost is not None:  # not shown in the book
            xgb_reg.fit(X_train, y_train,
                        eval_set=[(X_val, y_val)], early_stopping_rounds=2)
            y_pred = xgb_reg.predict(X_val)
            val_error = mean_squared_error(y_val, y_pred)
            print("Validation MSE:", val_error)

    # Stacking

    # data set
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_dsel, y_train, y_dsel = train_test_split(X, y, test_size=0.5, random_state=42)

    # Train a pool of 10 classifiers
    pool_classifiers = RandomForestClassifier(n_estimators=10)
    pool_classifiers.fit(X_train, y_train)

    # Initialize the DES model
    knorae = KNORAE(pool_classifiers)

    # Preprocess the Dynamic Selection dataset (DSEL)
    knorae.fit(X_dsel, y_dsel)

    # Predict new examples:
    y_pred_des = knorae.predict(X_test)

    print('des accuracy score: ', accuracy_score(y_test, y_pred_des))  # 0.984 much better than bagging and pasting
    print('des RMSE: ', np.sqrt(mean_squared_error(y_test, y_pred_des)))  # 0.12649110640673517