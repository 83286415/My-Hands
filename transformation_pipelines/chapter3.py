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
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

WORKING_PATH = os.path.abspath(os.path.join(os.getcwd(), '..'))
ROOT_PATH = os.path.join(WORKING_PATH, 'Hands on SK and TS\\')
CHAPTER_ID = "classification"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(ROOT_PATH, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()  # tight_layout: picture border auto-control
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


def plot_confusion_matrix(matrix):
    cmap = matplotlib.cm.jet  # set the color map to "jet" (low: blue - > high: red)
    fig = plt.figure(figsize=(8, 8))  # 8*8 inches picture include the border frame
    ax = fig.add_subplot(111)  # add the matrix image, 1:row count, 1:column count, 1:picture position.in row and column
    cax = ax.matshow(matrix, cmap=cmap)  # plot the matrix image with color map
    fig.colorbar(cax)  # show the color bar


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

    some_digit = X[36011]  # the 36001th image, its number is 5
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
    y_train_5 = (y_train == sample_number)  # return a np list: if 5, then y_train_5 is True. Else, y_train_5 is False.
    y_test_5 = (y_test == sample_number)  # actually it returns a np matrix. For this case, it is a list as above.

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
    # cv=3 3 folds: 2 for training and 1 for validation
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

    # multi-class classification OvA OvO
    # binary classifier and multi-class classifier refer to my notebook

    # 1. OvA
    sgd_clf.fit(X_train, y_train)  # train the classifier
    sgd_clf.predict([some_digit])  # some_digit == 5
    some_digit_scores = sgd_clf.decision_function([some_digit])  # the classifier runs with the OvA strategy
    # OvA strategy makes sure all classes have their own binary classifier. And all of them will run over the object.

    # print(some_digit_scores)
    # the output is list of ten scores, which are the scores of number 0-9. The sixth is highest for some_digit is 5.
    # output: [[-211564.05865206 -219445.21022825 -461783.93374972  -16252.73324556 -288195.70441995   34930.7725491
    # -335369.12969411 -282270.17392149 -25547.54596887 -339794.68286819]] the sixth (#5) is the biggest one.

    some_digit_max = np.argmax(some_digit_scores)
    # print(some_digit_max)  # output: 5  The max score's index is 5.
    # print(sgd_clf.classes_)  # output: [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]

    # 2. OvO
    ovo_clf = OneVsOneClassifier(SGDClassifier(max_iter=5, random_state=42))  # force sgd into OvO
    ovo_clf.fit(X_train, y_train)
    ovo_predicted = ovo_clf.predict([some_digit])
    # print(ovo_predicted)  # output: [5.]
    # The binary classifier.predict returns True in this case but OvO returns 5 for OvO runs 45 classifiers.

    ovo_clf_count = len(ovo_clf.estimators_)  # show the count of ovo_clf's classifiers
    # print(ovo_clf_count)  # output: 45      That is N*(N-1)/2, N=10

    # multi-classifier forest as the OvO contrast
    forest_clf.fit(X_train, y_train)  # forest_clf is capable of handling multi-classes. No need to force it into OvO
    forest_clf.predict([some_digit])  # so its predict returns [5]
    forest_probability_predicted = forest_clf.predict_proba([some_digit])
    # print(forest_probability_predicted)
    # output: [[0.1 0.  0.  0.  0.  0.9 0.  0.  0.  0. ]] the probability of each class can be assigned

    # evaluate the sgd classifier with y_train set in OvA
    sgd_val_result = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
    # print(sgd_val_result)  # output: [0.84063187 0.84899245 0.86652998]
    # The scores of 3 times validation are lower than that with parameter y_train_5 for this time its ten classifiers.

    # StandardScaler improves the validation score of sgd classifier (hands on book P96)
    scalar = StandardScaler()  # define a scalar
    X_train_scaled = scalar.fit_transform(X_train.astype(np.float64))
    # astype: transform np array elements' data type into float64
    # Here the fit and transform can standardize X_train set and reduce the error
    sgd_val_result_improved = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
    # print(sgd_val_result_improved)  # output: [0.91011798 0.90874544 0.906636  ]  the score is higher than line338

    # Error Analysis

    y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)  # y_train, not y_train_5.So 10 classifiers
    conf_mx = confusion_matrix(y_train, y_train_pred)
    # print(conf_mx)
    # output: a 10 dimension array for ten classifier of number 0~9. The sixth row is the 5s classifier, 4582 classified
    # [[5725    3   24    9   10   49   50   10   39    4]
    #  [   2 6493   43   25    7   40    5   10  109    8]
    #  [  51   41 5321  104   89   26   87   60  166   13]
    #  [  47   46  141 5342    1  231   40   50  141   92]
    #  [  19   29   41   10 5366    9   56   37   86  189]
    #  [  73   45   36  193   64 4582  111   30  193   94]
    #  [  29   34   44    2   42   85 5627   10   45    0]
    #  [  25   24   74   32   54   12    6 5787   15  236]
    #  [  52  161   73  156   10  163   61   25 5027  123]
    #  [  43   35   26   92  178   28    2  223   82 5240]]

    # plot the confusion matrix
    # plt.matshow(conf_mx, cmap=plt.cm.gray)  # this can plot a gray matrix image without a color bar
    plot_confusion_matrix(conf_mx)  # plot a matrix with color bar
    save_fig("confusion_matrix_plot", tight_layout=False)  # tight_layout: picture border auto-control
    # plt.show()

    # plot the confusion matrix with error rate
    row_sums = conf_mx.sum(axis=1, keepdims=True)  # axis=1 (y axis). so it returns the sums of each row (classifier).
    # print(row_sums)  # output: [[5923] [6742] [5958] [6131] [5842] [5421] [5918] [6265] [5851] [5949]]
    norm_conf_mx = conf_mx / row_sums  # norm_conf_mx value: big: error - > small: correct
    # print(norm_conf_mx)  # output the array with each element divided by row_sums accordingly
    np.fill_diagonal(norm_conf_mx, 0)  # replace the diagonal elements with 0 in this array
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)  # the color of diagonal elements should be black for their values are 0
    save_fig("confusion_matrix_errors_plot", tight_layout=False)
    # plt.show()  # classified correctly: black (smaller value) - > misclassified: white (bigger value)

    # Analysing individual error on 3 and 5 classifier
    cl_a, cl_b = 3, 5
    X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]  # the picture of all 3 label in y_train and predicted
    X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]  # X_train[label_set_index]
    X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]  # &: the intersection of y label and y predicted label
    X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

    plt.figure(figsize=(8, 8))
    plt.subplot(221)  # 4 matrix image, 2 rows, 2 columns, position: 1 (left upper)
    plot_digits(X_aa[:25], images_per_row=5)  # the first 25 elements
    plt.subplot(222)  # position: 2 right upper
    plot_digits(X_ab[:25], images_per_row=5)
    plt.subplot(223)  # position: 3 left bottom
    plot_digits(X_ba[:25], images_per_row=5)
    plt.subplot(224)  # position: 4 right bottom
    plot_digits(X_bb[:25], images_per_row=5)
    save_fig("3_5_error_analysis_digits_plot")
    # plt.show()

    # multi-label classification:
    # a train data matches more than 1 label. like # 5 is < 7 and odd, so it label as [false, true]

    # make multi-label train target data set
    y_train_large = (y_train >= 7)  # all numbers >= 7; y_train_larger: [true, false, true, true], true if the #>=7 in X
    y_train_odd = (y_train % 2 == 1)  # all odd numbers;
    y_multi_label = np.c_[y_train_large, y_train_odd]  # join the left np matrix to the right np matrix
    # y_multi_label: [[true, false, true,...], [false, false, true, ...]] two labels joined. so the predicted as below.

    knn_clf = KNeighborsClassifier()  # define a multi-label classifier
    knn_clf_FIT = knn_clf.fit(X_train, y_multi_label)  # train this classifier with multi-target data set
    # print(knn_clf_FIT)  # output:
    # KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
    #        metric_params=None, n_jobs=1, n_neighbors=5, p=2,
    #        weights='uniform')

    # test this classifier with some digit 5
    some_digit_knn_result = knn_clf.predict([some_digit])  # some_digit is 5
    # print(some_digit_knn_result)  # output: [[False  True]] The first False is large label; The 2nd True is odd label.

    # evaluate the multi-label classifier with F1 score
    # note: it will take a very long time to compute the F1 score. So comment it when running this py file
    y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multi_label, cv=3, n_jobs=-1)
    multi_label_f1 = f1_score(y_multi_label, y_train_knn_pred, average="macro")
    # average=macro, f1=unweighted mean; average=weighted, f1=weighted mean, maybe not between precision and recall
    # print(multi_label_f1)  # output: 0.97709078477525002

    # multi-output classification: the label is not true or false, but it's like a value in (0, 255) to a point's grey.

    # make train data set with noise
    noise = np.random.randint(0, 100, (len(X_train), 784))  # low=0, high=100, size=(len(X_train), 784)
    # len(X_train)=60000, 784=28*28     To generate a 6000*28*28 random int (0<=int<100)
    X_train_mod = X_train + noise  # the elements in the same row # and column # can +-*/ compute
    # the code above is to add noise into each pixel of images
    noise = np.random.randint(0, 100, (len(X_test), 784))
    X_test_mod = X_test + noise
    y_train_mod = X_train  # change the target into original train set as a contrast
    y_test_mod = X_test

    # plot the noisy image the original image
    some_index = 5500  # number 5
    plt.subplot(121)  # one row two columns: position 1 is left upper one
    plot_digit(X_test_mod[some_index])
    plt.subplot(122)
    plot_digit(y_test_mod[some_index])
    save_fig("noisy_digit_example_plot")

    # remove the noise with multi-output classifier KNeighborsClassifier()
    knn_clf.fit(X_train_mod, y_train_mod)  # train the classifier(noisy_data, target_data)
    clean_digit = knn_clf.predict([X_test_mod[some_index]])  # remove the noise
    plot_digit(clean_digit)
    save_fig("cleaned_digit_example_plot")
    plt.show()
