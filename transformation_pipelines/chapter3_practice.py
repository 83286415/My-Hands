import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_mldata
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier
from scipy.ndimage.interpolation import shift


TITANIC_PATH = os.path.join("datasets", "titanic")


def shift_digit(digit_array, dx, dy, new=0):
    return shift(digit_array.reshape(28, 28), [dy, dx], cval=new).reshape(784)


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=plt.cm.binary,
               interpolation="nearest")
    plt.axis("off")


def load_titanic_data(filename, titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path, filename)
    return pd.read_csv(csv_path)


if __name__ == '__main__':

    # preparation the data
    mnist = fetch_mldata('MNIST original')
    X, y = mnist['data'], mnist['target']

    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

    sample_number = 2
    y_train_5 = (y_train == sample_number)
    y_test_5 = (y_test == sample_number)
    some_digit = X[36011]

    # my solutions
    print('my solutions')

    # question 1
    print('question 1')

    # knn really takes a long time to fit. So i comment it out and find another way to go.
    knn_clf = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=4)
    knn_clf.fit(X_train, y_train)
    y_knn_pred = knn_clf.predict(X_test)
    knn_accuracy = accuracy_score(y_test, y_knn_pred)
    print(knn_accuracy)  # 97.14%

    sgd_clf = SGDClassifier(max_iter=5, random_state=42, )
    # sgd_clf = SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
    #        eta0=0.0, fit_intercept=True, l1_ratio=0.15,
    #        learning_rate='optimal', loss='hinge', max_iter=None, n_iter=None,
    #        n_jobs=1, penalty='l2', power_t=0.5, random_state=42,
    #        shuffle=True, tol=None, verbose=0, warm_start=False)
    sgd_clf.fit(X_train, y_train)
    y_test_sgd_pred = sgd_clf.predict(X_test)
    sgd_accuracy = accuracy_score(y_test, y_test_sgd_pred)
    print(sgd_accuracy)  # 86%
    sgd_val_accuracy = cross_val_score(sgd_clf, X_test, y_test, cv=3, scoring="accuracy")
    print(sgd_val_accuracy)  # [0.86557688 0.83744187 0.88078212]

    # question 2
    print('question 2')

    X_train_expanded = [X_train]
    y_train_expanded = [y_train]

    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shifted_images = np.apply_along_axis(shift_digit, axis=1, arr=X_train, dx=dx, dy=dy)
        X_train_expanded.append(shifted_images)
        y_train_expanded.append(y_train)

    X_train_expanded = np.concatenate(X_train_expanded)
    y_train_expanded = np.concatenate(y_train_expanded)

    sgd_clf = SGDClassifier(max_iter=10, random_state=42, )
    sgd_clf.fit(X_train_expanded, y_train_expanded)
    y_test_sgd_pred = sgd_clf.predict(X_test)
    sgd_accuracy = accuracy_score(y_test, y_test_sgd_pred)
    print(sgd_accuracy)  # 85%
    sgd_val_accuracy = cross_val_score(sgd_clf, X_test, y_test, cv=3, scoring="accuracy")
    print(sgd_val_accuracy)  # [0.83163571 0.86858686 0.85040553]

    knn_clf = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=4)
    knn_clf.fit(X_train_expanded, y_train_expanded)
    y_knn_pred = knn_clf.predict(X_test)
    knn_accuracy = accuracy_score(y_test, y_knn_pred)
    print(knn_accuracy)  # 97.63% better than 97.14% to the question 1

    # question 3
    print('question 3: tackle the Titanic data set refer to chapter3_titanic')
    print('refer to chapter3_titanic.py')
    print('\n')

    # question 4
    print('question 4: spam classifier')
    print('refer to chapter3_spam.py')
