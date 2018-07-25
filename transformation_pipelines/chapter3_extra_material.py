import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from scipy.ndimage.interpolation import shift


def plot_roc_curve(_fpr, _tpr, label=None):
    plt.plot(_fpr, _tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')  # k--: black dotted line; range: x axis 0~1, y axis 0~1
    plt.axis([0, 1, 0, 1])  # axises' limits
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)


def shift_digit(digit_array, dx, dy, new=0):
    return shift(digit_array.reshape(28, 28), [dy, dx], cval=new).reshape(784)  # re-schedule x and y's coordinates
    # cval: values for the points outside the boundaries. 0: white, 1000: black


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=plt.cm.binary,
               interpolation="nearest")
    plt.axis("off")


if __name__ == '__main__':

    # preparation the data
    mnist = fetch_mldata('MNIST original')
    X, y = mnist['data'], mnist['target']

    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

    sample_number = 5
    y_train_5 = (y_train == sample_number)
    y_test_5 = (y_test == sample_number)
    some_digit = X[36011]

    # Extra material

    # Dummy classifier
    dmy_clf = DummyClassifier()  # This classifier is to get the baseline for that random predictions
    y_probas_dmy = cross_val_predict(dmy_clf, X_train, y_train_5, cv=3, method="predict_proba")  # probability
    y_scores_dmy = y_probas_dmy[:, 1]  # get the second column data and return as a list [0 0 0 ... 0 0 1]

    fprr, tprr, thresholdsr = roc_curve(y_train_5, y_scores_dmy)
    plot_roc_curve(fprr, tprr)  # the roc line in the middle
    plt.show()

    # KNN classifier
    knn_clf = KNeighborsClassifier(n_jobs=4, weights='distance', n_neighbors=4)  # n_jobs -1: low cpu usage
    # distance: the point is more closer, its weight is greater; n_neighbors: choose 4 neighbor points as references

    # This piece of codes takes a long time to run. So comment it for testing followings.
    knn_clf.fit(X_train, y_train)
    y_knn_pred = knn_clf.predict(X_test)
    knn_accuracy = accuracy_score(y_test, y_knn_pred)
    print(knn_accuracy)

    plot_digit(shift_digit(some_digit, 5, 1, new=100))
    # new coordinate: x' = x + 5, y' = y + 1; new: 100 gray (0 white, 1000 black)
    plt.show()

    X_train_expanded = [X_train]  # make the data array into a list
    y_train_expanded = [y_train]
    print(X_train_expanded, y_train_expanded)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shifted_images = np.apply_along_axis(shift_digit, axis=1, arr=X_train, dx=dx, dy=dy)
        # make changes according to shift_digit to X_train array with new dx and dy in this for loop
        X_train_expanded.append(shifted_images)
        y_train_expanded.append(y_train)

    X_train_expanded = np.concatenate(X_train_expanded)  # join X_train and shifted_images along x axis
    y_train_expanded = np.concatenate(y_train_expanded)
    print(X_train_expanded.shape, y_train_expanded.shape)  # output: (300000, 784) (300000,)

    knn_clf.fit(X_train_expanded, y_train_expanded)  # train this classifier with expanded data sets
    y_knn_expanded_pred = knn_clf.predict(X_test)
    knn_accuracy_expanded = accuracy_score(y_test, y_knn_expanded_pred)
    print(knn_accuracy_expanded)

    ambiguous_digit = X_test[2589]  # choose an uncertain number's image
    proba_ambiguous_digit = knn_clf.predict_proba([ambiguous_digit])  # find its probability of each number
    print(proba_ambiguous_digit)
    plot_digit(ambiguous_digit)
    plt.show()