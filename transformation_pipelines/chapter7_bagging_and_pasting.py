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

    # Out of Bagging Evaluating

    # build a new Bagging model with oob (out of bagging evaluating)
    bag_oob_clf = BaggingClassifier(
                                    DecisionTreeClassifier(random_state=42), n_estimators=500,
                                    bootstrap=True, n_jobs=-1, oob_score=True, random_state=40)
    bag_oob_clf.fit(X_train, y_train)
    # oob_score_: is the accuracy score of the validation with train data
    print('oob score(the accuracy score on oob of train data): ', bag_oob_clf.oob_score_)  # 0.9013333333333333

    # evaluating the accuracy score on test set
    y_pred_oob = bag_oob_clf.predict(X_test)
    print('oob accuracy score: ', accuracy_score(y_test, y_pred_oob))  # 0.912
    print('oob RMSE: ', np.sqrt(mean_squared_error(y_test, y_pred_oob)))  # 0.2966479394838265
    print(bag_oob_clf.oob_decision_function_)
    '''
        [[0.31746032 0.68253968]  # the classification probabilities of the first sample in test set
         [0.34117647 0.65882353]
         [1.         0.        ]  # the classification result(also it's the probability) of the third sample in test set
         [0.         1.        ]
         [0.         1.        ]
         [0.08379888 0.91620112]
         [0.31693989 0.68306011]
         [0.02923977 0.97076023]
         [0.97687861 0.02312139]
         [0.97765363 0.02234637]
         [0.74404762 0.25595238]
         [0.         1.        ]
         [0.71195652 0.28804348]
         [0.83957219 0.16042781]
         [0.97777778 0.02222222]
         [0.0625     0.9375    ]
         [0.         1.        ]
         [0.97297297 0.02702703]
         [0.95238095 0.04761905]
         [1.         0.        ]
         [0.01704545 0.98295455]
         [0.38947368 0.61052632]
         [0.88700565 0.11299435]
         [1.         0.        ]
         [0.96685083 0.03314917]
         [0.         1.        ]
         [0.99428571 0.00571429]
         [1.         0.        ]
         [0.         1.        ]
         [0.64804469 0.35195531]
         [0.         1.        ]
         [1.         0.        ]
         [0.         1.        ]
         [0.         1.        ]
         [0.13402062 0.86597938]
         [1.         0.        ]
         [0.         1.        ]
         [0.36065574 0.63934426]
         [0.         1.        ]
         [1.         0.        ]
         [0.27093596 0.72906404]
         [0.34146341 0.65853659]
         [1.         0.        ]
         [1.         0.        ]
         [0.         1.        ]
         [1.         0.        ]
         [1.         0.        ]
         [0.         1.        ]
         [1.         0.        ]
         [0.00531915 0.99468085]
         [0.98265896 0.01734104]
         [0.91428571 0.08571429]
         [0.97282609 0.02717391]
         [0.97029703 0.02970297]
         [0.         1.        ]
         [0.06134969 0.93865031]
         [0.98019802 0.01980198]
         [0.         1.        ]
         [0.         1.        ]
         [0.         1.        ]
         [0.97790055 0.02209945]
         [0.79473684 0.20526316]
         [0.41919192 0.58080808]
         [0.99473684 0.00526316]
         [0.         1.        ]
         [0.67613636 0.32386364]
         [1.         0.        ]
         [1.         0.        ]
         [0.87356322 0.12643678]
         [1.         0.        ]
         [0.56140351 0.43859649]
         [0.16304348 0.83695652]
         [0.67539267 0.32460733]
         [0.90673575 0.09326425]
         [0.         1.        ]
         [0.16201117 0.83798883]
         [0.89005236 0.10994764]
         [1.         0.        ]
         [0.         1.        ]
         [0.995      0.005     ]
         [0.         1.        ]
         [0.07272727 0.92727273]
         [0.05418719 0.94581281]
         [0.29533679 0.70466321]
         [1.         0.        ]
         [0.         1.        ]
         [0.81871345 0.18128655]
         [0.01092896 0.98907104]
         [0.         1.        ]
         [0.         1.        ]
         [0.22513089 0.77486911]
         [1.         0.        ]
         [0.         1.        ]
         [0.         1.        ]
         [0.         1.        ]
         [0.9368932  0.0631068 ]
         [0.76536313 0.23463687]
         [0.         1.        ]
         [1.         0.        ]
         [0.17127072 0.82872928]
         [0.65306122 0.34693878]
         [0.         1.        ]
         [0.03076923 0.96923077]
         [0.49444444 0.50555556]
         [1.         0.        ]
         [0.02673797 0.97326203]
         [0.98870056 0.01129944]
         [0.23121387 0.76878613]
         [0.5        0.5       ]
         [0.9947644  0.0052356 ]
         [0.00555556 0.99444444]
         [0.98963731 0.01036269]
         [0.25641026 0.74358974]
         [0.92972973 0.07027027]
         [1.         0.        ]
         [1.         0.        ]
         [0.         1.        ]
         [0.         1.        ]
         [0.80681818 0.19318182]
         [1.         0.        ]
         [0.0106383  0.9893617 ]
         [1.         0.        ]
         [1.         0.        ]
         [1.         0.        ]
         [0.98181818 0.01818182]
         [1.         0.        ]
         [0.01036269 0.98963731]
         [0.97752809 0.02247191]
         [0.99453552 0.00546448]
         [0.01960784 0.98039216]
         [0.18367347 0.81632653]
         [0.98387097 0.01612903]
         [0.29533679 0.70466321]
         [0.98295455 0.01704545]
         [0.         1.        ]
         [0.00561798 0.99438202]
         [0.75138122 0.24861878]
         [0.38624339 0.61375661]
         [0.42708333 0.57291667]
         [0.86315789 0.13684211]
         [0.92964824 0.07035176]
         [0.05699482 0.94300518]
         [0.82802548 0.17197452]
         [0.01546392 0.98453608]
         [0.         1.        ]
         [0.02298851 0.97701149]
         [0.96721311 0.03278689]
         [1.         0.        ]
         [1.         0.        ]
         [0.01041667 0.98958333]
         [0.         1.        ]
         [0.0326087  0.9673913 ]
         [0.01020408 0.98979592]
         [1.         0.        ]
         [1.         0.        ]
         [0.93785311 0.06214689]
         [1.         0.        ]
         [1.         0.        ]
         [0.99462366 0.00537634]
         [0.         1.        ]
         [0.38860104 0.61139896]
         [0.32065217 0.67934783]
         [0.         1.        ]
         [0.         1.        ]
         [0.31182796 0.68817204]
         [1.         0.        ]
         [1.         0.        ]
         [0.         1.        ]
         [1.         0.        ]
         [0.00588235 0.99411765]
         [0.         1.        ]
         [0.98387097 0.01612903]
         [0.         1.        ]
         [0.         1.        ]
         [1.         0.        ]
         [0.         1.        ]
         [0.62264151 0.37735849]
         [0.92344498 0.07655502]
         [0.         1.        ]
         [0.99526066 0.00473934]
         [1.         0.        ]
         [0.98888889 0.01111111]
         [0.         1.        ]
         [0.         1.        ]
         [1.         0.        ]
         [0.06451613 0.93548387]
         [1.         0.        ]
         [0.05154639 0.94845361]
         [0.         1.        ]
         [1.         0.        ]
         [0.         1.        ]
         [0.03278689 0.96721311]
         [1.         0.        ]
         [0.95808383 0.04191617]
         [0.79532164 0.20467836]
         [0.55665025 0.44334975]
         [0.         1.        ]
         [0.18604651 0.81395349]
         [1.         0.        ]
         [0.93121693 0.06878307]
         [0.97740113 0.02259887]
         [1.         0.        ]
         [0.00531915 0.99468085]
         [0.         1.        ]
         [0.44623656 0.55376344]
         [0.86363636 0.13636364]
         [0.         1.        ]
         [0.         1.        ]
         [1.         0.        ]
         [0.00558659 0.99441341]
         [0.         1.        ]
         [0.96923077 0.03076923]
         [0.         1.        ]
         [0.21649485 0.78350515]
         [0.         1.        ]
         [1.         0.        ]
         [0.         1.        ]
         [0.         1.        ]
         [0.98477157 0.01522843]
         [0.8        0.2       ]
         [0.99441341 0.00558659]
         [0.         1.        ]
         [0.08379888 0.91620112]
         [0.98984772 0.01015228]
         [0.01142857 0.98857143]
         [0.         1.        ]
         [0.02747253 0.97252747]
         [1.         0.        ]
         [0.79144385 0.20855615]
         [0.         1.        ]
         [0.90804598 0.09195402]
         [0.98387097 0.01612903]
         [0.20634921 0.79365079]
         [0.19767442 0.80232558]
         [1.         0.        ]
         [0.         1.        ]
         [0.         1.        ]
         [0.         1.        ]
         [0.20338983 0.79661017]
         [0.98181818 0.01818182]
         [0.         1.        ]
         [1.         0.        ]
         [0.98969072 0.01030928]
         [0.         1.        ]
         [0.48663102 0.51336898]
         [1.         0.        ]
         [0.         1.        ]
         [1.         0.        ]
         [0.         1.        ]
         [0.         1.        ]
         [0.07821229 0.92178771]
         [0.11176471 0.88823529]
         [0.99415205 0.00584795]
         [0.03015075 0.96984925]
         [1.         0.        ]
         [0.40837696 0.59162304]
         [0.04891304 0.95108696]
         [0.51595745 0.48404255]
         [0.51898734 0.48101266]
         [0.         1.        ]
         [1.         0.        ]
         [0.         1.        ]
         [0.         1.        ]
         [0.59903382 0.40096618]
         [0.         1.        ]
         [1.         0.        ]
         [0.24157303 0.75842697]
         [0.81052632 0.18947368]
         [0.08717949 0.91282051]
         [0.99453552 0.00546448]
         [0.82142857 0.17857143]
         [0.         1.        ]
         [0.         1.        ]
         [0.125      0.875     ]
         [0.04712042 0.95287958]
         [0.         1.        ]
         [1.         0.        ]
         [0.89150943 0.10849057]
         [0.1978022  0.8021978 ]
         [0.95238095 0.04761905]
         [0.00515464 0.99484536]
         [0.609375   0.390625  ]
         [0.07692308 0.92307692]
         [0.99484536 0.00515464]
         [0.84210526 0.15789474]
         [0.         1.        ]
         [0.99484536 0.00515464]
         [0.95876289 0.04123711]
         [0.         1.        ]
         [0.         1.        ]
         [1.         0.        ]
         [0.         1.        ]
         [1.         0.        ]
         [0.26903553 0.73096447]
         [0.98461538 0.01538462]
         [1.         0.        ]
         [0.         1.        ]
         [0.00574713 0.99425287]
         [0.85142857 0.14857143]
         [0.         1.        ]
         [1.         0.        ]
         [0.76506024 0.23493976]
         [0.8969697  0.1030303 ]
         [1.         0.        ]
         [0.73333333 0.26666667]
         [0.47727273 0.52272727]
         [0.         1.        ]
         [0.92473118 0.07526882]
         [0.         1.        ]
         [1.         0.        ]
         [0.87709497 0.12290503]
         [1.         0.        ]
         [1.         0.        ]
         [0.74752475 0.25247525]
         [0.09146341 0.90853659]
         [0.44329897 0.55670103]
         [0.22395833 0.77604167]
         [0.         1.        ]
         [0.87046632 0.12953368]
         [0.78212291 0.21787709]
         [0.00507614 0.99492386]
         [1.         0.        ]
         [1.         0.        ]
         [1.         0.        ]
         [0.         1.        ]
         [0.02884615 0.97115385]
         [0.96571429 0.03428571]
         [0.93478261 0.06521739]
         [1.         0.        ]
         [0.49756098 0.50243902]
         [1.         0.        ]
         [0.         1.        ]
         [1.         0.        ]
         [0.01604278 0.98395722]
         [1.         0.        ]
         [1.         0.        ]
         [1.         0.        ]
         [0.         1.        ]
         [0.96987952 0.03012048]
         [0.         1.        ]
         [0.05747126 0.94252874]
         [0.         1.        ]
         [0.         1.        ]
         [1.         0.        ]
         [1.         0.        ]
         [0.         1.        ]
         [0.98989899 0.01010101]
         [0.01675978 0.98324022]
         [1.         0.        ]
         [0.13541667 0.86458333]
         [0.         1.        ]
         [0.00546448 0.99453552]
         [0.         1.        ]
         [0.41836735 0.58163265]
         [0.11309524 0.88690476]
         [0.22110553 0.77889447]
         [1.         0.        ]
         [0.97647059 0.02352941]
         [0.22826087 0.77173913]
         [0.98882682 0.01117318]
         [0.         1.        ]
         [0.         1.        ]
         [1.         0.        ]
         [0.96428571 0.03571429]
         [0.33507853 0.66492147]
         [0.98235294 0.01764706]
         [1.         0.        ]
         [0.         1.        ]
         [0.99465241 0.00534759]
         [0.         1.        ]
         [0.06043956 0.93956044]
         [0.97619048 0.02380952]
         [1.         0.        ]
         [0.03108808 0.96891192]
         [0.57291667 0.42708333]]
    '''