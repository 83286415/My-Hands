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

from sklearn.metrics import confusion_matrix

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
    # print(y_train_5_matrix)

    y_train_perfect_predictions = y_train_5
    y_train_perfect_5_matrix = confusion_matrix(y_train_5, y_train_perfect_predictions)
    print(y_train_perfect_5_matrix)
