import os
import tarfile
from six.moves import urllib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.model_selection import train_test_split
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit


ROOT_PATH = "D:\\AI\\handson-ml-master\\"
# HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_TGZ_PATH = ROOT_PATH + "datasets\\housing\\housing.tgz"
HOUSING_PATH = ROOT_PATH + "datasets\\housing\\"


def fetch_housing_data(housing_path=HOUSING_PATH, housing_tgz_path=HOUSING_TGZ_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    # tgz_path = os.path.join(housing_path, "housing.tgz")
    # urllib.request.urlretrieve(housing_url, tgz_path)
    if not housing_tgz_path:
        print('No housing tgz file. Exit!')
        exit()
    housing_tgz = tarfile.open(housing_tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    if not csv_path:
        fetch_housing_data(housing_path=HOUSING_PATH, housing_tgz_path=HOUSING_TGZ_PATH)
        csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


if __name__ == '__main__':
    housing = load_housing_data(housing_path=HOUSING_PATH)

    head = housing.head(n=5)    # the first five rows of csv
    tail = housing.tail(n=5)    # the last five rows of csv
    # print(head)

    info = housing.info()       # description of csv: this line will output info if not commented

    ocean_proximity = housing['ocean_proximity'].value_counts()
    # the total numbers of each item in the column "ocean_proximity"
    # print(ocean_proximity.INLAND)  # the total number of INLAND in col ocean proximity

    describe = housing.describe()  # the count,mean,std,min,25%,50%,75%,max's values of each column in csv
    # print(describe.total_rooms)

    # matplotlib in line pls refer to jupyter notebook

    # data set split 1: random seed 42 makes sure train set and test set are not changed after each split operation
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)  # split housing into two parts
    train_set_tail = train_set.tail()
    # print(train_set_tail)
    test_set_head = test_set.head()
    # print(test_set_head)

    # data set split 2: split set according to id
    housing_with_id = housing.reset_index()  # creates an `index` column
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
    # print(test_set.head())

    housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]  # adds an in column using existing data
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
    # print(test_set.head())

    # the histogram picture of housing data show
    housing.hist(bins=30, figsize=(20, 12))
    # plt.show()
    housing["median_income"].hist(bins=50, figsize=(7, 4))  # bins: the bar count
    # plt.show()

    # Divide by 1.5 to limit the number of income categories
    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)  # ceil: round up to int value
    # Label those above 5 as 5
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
    # print(housing["income_cat"].value_counts())
    housing["income_cat"].hist(bins=30, figsize=(20, 12))
    # plt.show()

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  # spliter of full data set
    for train_index, test_index in split.split(housing, housing["income_cat"]):  # housing=X, income_cat=y
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    # the proportion of splited result of the stratified sampling
    # print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
    # the proportion of random splited result
    # print(housing["income_cat"].value_counts() / len(housing))
    # the two proportion results are almost same