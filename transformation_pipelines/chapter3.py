from __future__ import division, print_function, unicode_literals  # To support both python 2 and python 3

# Common imports
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

ROOT_PATH = "D:\\AI\\handson-ml-master\\"
CHAPTER_ID = "classification"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(ROOT_PATH, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=plt.cm.binary,
               interpolation="nearest")
    plt.axis("off")
    # plt.show()  # commented for not showing


def plot_digits(instances, images_per_row=10, **options):
    size = 28
    if len(instances) > 0:
        images_per_row = min(len(instances), images_per_row)  # smaller one in a row; len(instances): count of instances
    else:
        images_per_row = 1
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1  # count of rows
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)  # count of empty images
    images.append(np.zeros((size, size * n_empty)))  # eg.  np.zeros((2, 1)) output: array([[ 0.], [ 0.]])
    for row in range(n_rows):  # first: append the array into list as above, then concatenate arrays together as below.
        rimages = images[row * images_per_row: (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))  # concatenate(X, axis=1): same as np.c_
    image = np.concatenate(row_images, axis=0)  # concatenate(X, axis=0): same as np.r_
    plt.imshow(image, cmap=matplotlib.cm.binary, **options)
    plt.axis("off")


class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)  # when get a sample picture like 5, return a zero array.


def plot_precision_recall_vs_threshold(_precisions, _recalls, _thresholds):
    plt.plot(_thresholds, _precisions[:-1], "b--", label="Precision", linewidth=2)  # blue dotted line; plot(x, y axis)
    plt.plot(_thresholds, _recalls[:-1], "g-", label="Recall", linewidth=2)  # green line; [:-1] ignore the last element
    # the last element of precisions and recalls lists are 1. so ignore it as above. refer to precision_recall_curve doc
    # http://www.360doc.com/content/15/0113/23/16740871_440559122.shtml  refer to the parameters in plot function
    plt.xlabel("Threshold", fontsize=16)  # the label
    plt.legend(loc="upper left", fontsize=16)  # show the label on the upper left of the picture
    plt.ylim([0, 1])  # y axis limit [0, 1]


def plot_precision_vs_recall(_precisions, _recalls):
    plt.plot(_recalls, _precisions, "b-", linewidth=2)  # x axis is recall, y axis is precisions
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])  # axises' limits


def plot_roc_curve(_fpr, _tpr, label=None):
    plt.plot(_fpr, _tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')  # k--: black dotted line; range: x axis 0~1, y axis 0~1
    plt.axis([0, 1, 0, 1])  # axises' limits
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)


if __name__ == '__main__':

    # to make this notebook's output stable across runs
    np.random.seed(42)

    # prepare to plot
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    # fetch MNIST
    mnist = fetch_mldata('MNIST original')
    # print(mnist)
    # output:{'DESCR': 'mldata.org dataset: mnist-original', 'COL_NAMES': ['label', 'data'], 'target': array([0., 0., 0.
    # , ..., 9., 9., 9.]), 'data': array([[0, 0, 0, ..., 0, 0, 0]...[0, 0, 0, ..., 0, 0, 0]], dtype=uint8)}

    X, y = mnist['data'], mnist['target']
    # print(X.shape)  # output: (70000, 784) 70000 images, that is 70000 rows; 784 = 28 * 28 (pixel)
    # print(y.shape)  # (70000) 70000 rows, so 70000 labels

    some_digit = X[36011]  # the 36001th image
    # some_digit_image = some_digit.reshape(28, 28)  # turn 784 elements into 28*28 array

    # codes below could be replaced by def plot_digit(data)

    # plt.imshow(some_digit_image, cmap=plt.cm.binary, interpolation="nearest") # plt can be replaced by matplotlib
    # image show function(X, cmap=color map)
    # plt.axis("off")  # no axis
    # plt.show()  # a pic of number 5 shown
    # print(y[36011])  # output: 5.0
    plot_digit(some_digit)

    plt.figure(figsize=(9, 9))
    example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]  # [bgn:end:step]
    plot_digits(example_images, images_per_row=10)
    # save_fig("more_digits_plot")  # save 0-9 picture
    # plt.show()

    # Training a classifier

    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    shuffle_index = np.random.permutation(60000)  # return a range sequence with a random order
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]  # shuffle the train set

    sample_number = 5
    y_train_5 = (y_train == sample_number)  # return a list: if 5, then y_train_5 is True. Else, y_train_5 is False.
    y_test_5 = (y_test == sample_number)

    sgd_clf = SGDClassifier(max_iter=5, random_state=42)  # define a SGD classifier with seed == 42
    sgd_clf_FIT = sgd_clf.fit(X_train, y_train_5)  # train X
    # output of sgd_clf_FIT:
    # SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
    #               eta0=0.0, fit_intercept=True, l1_ratio=0.15,
    #               learning_rate='optimal', loss='hinge', max_iter=5, n_iter=None,
    #               n_jobs=1, penalty='l2', power_t=0.5, random_state=42, shuffle=True,
    #               tol=None, verbose=0, warm_start=False)
    some_digit_classified = sgd_clf.predict([some_digit])  # To predict some test set with the trained classifier
    example_images_classified = sgd_clf.predict(X[:60000:600])
    # print(some_digit_classified)  # output: [True]
    # print(example_images_classified)  # output a list: [False, False, False ... True, False, ...] True:5, False:non-5

    # Here is a script below, written by Lich, to show all pictures predicted

    # BGN
    index_list = []
    for true_example, X_index in zip(example_images_classified, X[:60000:600]):  # get sample number's array from X
        if true_example:
            index_list.append(X_index)  # make array list
    sample_images = np.r_[index_list]  # join arrays together
    # plot_digits(sample_images)
    # plt.show()  # to show all picture predicted. commented for now.
    # END

    # Performance Measures

    sgd_val_result = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")  # return scores
    # the "accuracy" should not be used for skewed data set
    # print(sgd_val_result)  # return a score of a classifier's cross validation, score > 95% is great.
    # output: [0.9502  0.96565 0.96495] (the score of each run of estimator's validation, cv=3)
    # The score means the accuracy of this classifier, which tells 5 in numbers, is about 96%

    never_5_clf = Never5Classifier()  # filter the number 5 pictures
    never5_val_result = cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")  # return scores
    # print(never5_val_result)  # output: [0.909   0.90715 0.9128 ]
    # the accuracy is about 90% for pictures of number 5 is about 10% in total pictures

    # confusion matrix
    # measure performance by confusion matrix (refer to P85 in my hands book)
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)  # return predictions based on each test sets
    # print(y_train_pred)  # output: [False False False ... False False False]

    y_train_5_matrix = confusion_matrix(y_train_5, y_train_pred)  # confusion matrix refer to my note or book P85 pic
    # print(y_train_5_matrix)  # output: matrix like [[TN, FP], [FN, TP]]
    #  output: [[53272  1307], [ 1077  4344]] FP or FN is not 0

    y_train_perfect_predictions = y_train_5
    y_train_perfect_5_matrix = confusion_matrix(y_train_5, y_train_perfect_predictions)
    # print(y_train_perfect_5_matrix)  # output: [[54579  0], [0  5421]] FP & FN == 0, because all predictions are right

    # precision
    # precision = TP / (TP + FP)
    prediction_precision = precision_score(y_train_5, y_train_pred)  # input: y_true, y_predicted
    # print(prediction_precision)
    # output: 0.7687135020350381.   That is 4344 / (4344 + 1307) as above
    # The best value is 1 and the worst value is 0.

    # recall
    # recall = TP / (TP + FN)
    prediction_recall = recall_score(y_train_5, y_train_pred)  # # input: y_true, y_predicted
    # print(prediction_recall)
    # output: 0.801328168234643     That is 4344 / (4344 + 1077) as above
    # The best value is 1 and the worst value is 0.

    # f1 score
    # f1 = TP / [TP + (FN + FP)/2]  That is F1 = 2 * (precision * recall) / (precision + recall)
    prediction_f1 = f1_score(y_train_5, y_train_pred)
    # print(prediction_f1)  # output: 0.7846820809248555

    # precision and recall trade off (refer to hands on book P87)
    y_some_digit_score = sgd_clf.decision_function([some_digit])  # output a score of classification
    # print(y_some_digit_score)  # output: [34930.7725491]

    threshold = 0  # if y score > threshold, then returns true
    y_some_digit_pred = (y_some_digit_score > threshold)
    # print(y_some_digit_pred)  # output: [ True]

    # another example of decision score function
    threshold = 200000  # raise the threshold value
    y_some_digit_pred = (y_some_digit_score > threshold)
    # print(y_some_digit_pred)  # output: [False]

    y_train_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                                       method="decision_function")  # add "method" to get scores instead of True, False
    # print(y_train_scores)
    # output: [ -434076.49813641 -1825667.15281624  -767086.76186905 ... -565357.11420164  -366599.16018198]

    # plot the scores cures of precisions and recalls to make the trade off relationship clear
    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_train_scores)
    plt.figure(figsize=(8, 4))
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    plt.xlim([-700000, 700000])
    save_fig("precision_recall_vs_threshold_plot")
    # plt.show()

    # try to set the threshold to get 90% precision
    y_train_pred_90 = (y_train_scores > 70000)
    precision_90 = precision_score(y_train_5, y_train_pred_90)
    # print(precision_90)  # output: 0.8659205116491548   It's about 90% precision. So the threshold could be 70000.
    recall_with_precision_90 = recall_score(y_train_5, y_train_pred_90)
    # print(recall_with_precision_90)  # output: 0.6993174691016417
    # Note: Do not try to get too high precision. It will slow your python running.

    # plot the precisions against recalls curve
    plt.figure(figsize=(8, 6))
    plot_precision_vs_recall(precisions, recalls)
    # use "precision_recall_curve" 's return to plot. because it's a list of precision and recall values.
    save_fig("precision_vs_recall_plot")
    # plt.show()

    # ROC Curve
    # ROC: x axis = FPR = FP / (FP + TN) = 1 - TNR = 1 - specificity
    #      y axis = TPR = TP / (TP + FN) = recall = sensitivity
    fpr, tpr, thresholds = roc_curve(y_train_5, y_train_scores)  # input: y_true, y_score

    plt.figure(figsize=(8, 6))
    plot_roc_curve(fpr, tpr)  # the ROC curve is the curve of the sgd classifier
    save_fig("roc_curve_plot")
    # plt.show()

    # AUC
    # AUC: the area under the curve
    # perfect classifier AUC = 1; purely random classifier AUC = 0.5
    y_sgd_auc = roc_auc_score(y_train_5, y_train_scores)
    # print(y_auc)  # output: 0.9624496555967155  it's good as it's close to 1.

    forest_clf = RandomForestClassifier(random_state=42)  # define a random forest classifier
    y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                        method="predict_proba")
    # decision_function method not in random forest, so use predict_proba instead.
    # But predict_proba returns a probability instead of y_score. So use this probability as the score.

    # print(y_probas_forest)
    # output: n dimension array shows each picture's probability of representing a 5 or not. [1  0] may shows it's non-5
    # [[1.  0.]
    #  [0.9 0.1]
    # [1. 0.]
    # ...
    # [1. 0.]
    # [1.  0.]
    # [1. 0.]]

    y_scores_forest = y_probas_forest[:, 1]  # use [0.9  0.1] as the score of this random forest classifier
    fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)  # get forest's fpr, tpr for ROC

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")  # plot SGD classifier's curve: blue dotted line
    plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")  # plot the random forest classifier's curve: blue line
    plt.legend(loc="lower right", fontsize=16)
    save_fig("roc_curve_comparison_plot")
    # plt.show()  # output: the random forest is better than SGD curve

    y_forest_auc = roc_auc_score(y_train_5, y_scores_forest)
    # print(y_forest_auc)  # output: 0.9931243366003829 it's better than sgd auc value.

    # multi-class classification