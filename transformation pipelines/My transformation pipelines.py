import os
import tarfile
# from six.moves import urllib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.model_selection import train_test_split
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.image as mpimg
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import Imputer
from sklearn.preprocessing.future_encoders import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression


ROOT_PATH = "D:\\AI\\handson-ml-master\\"
# HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_TGZ_PATH = ROOT_PATH + "datasets\\housing\\housing.tgz"
HOUSING_PATH = ROOT_PATH + "datasets\\housing\\"


# column index
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    # TransformerMixin: get fit_transform()   BaseEstimator: if no *args or **kargs, get_params() set_params() available
    def __init__(self, add_bedrooms_per_room=True):   # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room  # add a gate to open or not in next data preparation step

    def fit(self, data_set, data_set_label=None):
        return self  # nothing else to do

    def transform(self, data_set, data_set_label=None):
        rooms_per_household = data_set[:, rooms_ix] / data_set[:, household_ix]  # rooms'count in one family
        population_per_household = data_set[:, population_ix] / data_set[:, household_ix]  # the people count in one family
        if self.add_bedrooms_per_room:
            bedrooms_per_room = data_set[:, bedrooms_ix] / data_set[:, rooms_ix]
            return np.c_[data_set, rooms_per_household, population_per_household,
                         bedrooms_per_room]  # c_: join left arrays with right arrays
        else:
            return np.c_[data_set, rooms_per_household, population_per_household]


# Create a class to select numerical or categorical columns
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, data_set, data_set_label=None):
        return self

    def transform(self, data_set):
        return data_set[self.attribute_names].values


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


def test_set_check(identifier, test_ratio, _hash=hashlib.md5):
    return _hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


if __name__ == '__main__':
    housing = load_housing_data(housing_path=HOUSING_PATH)

    head = housing.head(n=5)    # the first five rows of csv
    tail = housing.tail(n=5)    # the last five rows of csv
    # print(head)
    # print(housing.loc[1, "population"])  # loc usage refer to notebook

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
    # x axis: the values of each column. y axis: the total count of items in each interval
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

    strat_train_set = housing
    strat_test_set = housing
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  # spliter of full data set
    for train_index, test_index in split.split(housing, housing["income_cat"]):  # housing=X, income_cat=y
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    # the proportion of splited result of the stratified sampling
    # print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
    # the proportion of random splited result
    # print(housing["income_cat"].value_counts() / len(housing))
    # the two proportion results are almost same

    # drop income_cat column and reset data sets
    # print(strat_test_set.head())
    for _set in (strat_train_set, strat_test_set):
        _set.drop(["income_cat"], axis=1, inplace=True)
    # print(strat_test_set.head())  #  the income_cat is removed from sets

    # Visualize the data to gain insights

    #  backup the train data set
    housing = strat_train_set.copy()

    # plot the scatter picture upon train set
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)  # alpha:transparency, 0:transparent 1:opaque
    # plt.show()

    # plot the scatter picture with population circles and median house values upon train set
    # s:size of point=population, c:color red:high price blue:low price, cmap:color map named jet, sharex:share x?
    # https://blog.csdn.net/zhangqilong120/article/details/72633115
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                 s=housing["population"] / 100, label="population", figsize=(10, 7),
                 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
                 sharex=False)
    plt.legend()  # show the label "population" on the picture's right higher corner
    # plt.show()

    # plot scatter picture with california image
    california_img = mpimg.imread(ROOT_PATH + '/images/end_to_end_project/california.png')
    ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10, 7),
                      s=housing['population'] / 100, label="Population",
                      c="median_house_value", cmap=plt.get_cmap("jet"),
                      colorbar=False, alpha=0.4,
                      )
    # extent: the coordinators of left lower point and right higher point
    plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
               cmap=plt.get_cmap("jet"))
    plt.ylabel("Latitude", fontsize=14)  # make y axis's explanation
    plt.xlabel("Longitude", fontsize=14)  # make x axis's explanation
    prices = housing["median_house_value"]
    # linspace: Return evenly spaced numbers over a specified interval.
    tick_values = np.linspace(prices.min(), prices.max(), 7)  # 7: the number of scalar shown on the right
    cbar = plt.colorbar()
    # the first list shows $500k, $403k... round(): si she wu ru
    cbar.ax.set_yticklabels(["$%dk" % (round(v / 1000)) for v in tick_values], fontsize=14)
    cbar.set_label('Median House Value', fontsize=16)  # title
    plt.legend(fontsize=16)  # show the legend "population"
    # plt.show()

    # correlations

    # get the correlations matrix shown in text
    corr_matrix = housing.corr()
    # show the correlation to median house value in ascending order
    # print(corr_matrix["median_house_value"].sort_values(ascending=False))

    # get the correlations matrix shown in picture tables
    #  For older versions of Pandas: make the n*n matrix according to list len
    attributes = ["median_house_value", "median_income", "total_rooms",
                  "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    # plt.show()

    # analysis the correlation between median income and median house value, shown in picture table
    housing.plot(kind="scatter", x="median_income", y="median_house_value",
                 alpha=0.1)
    plt.axis([0, 16, 0, 550000])  # list's [0][1] make the x axis, [2][3] make the y axis
    # plt.show()

    # attributes combination test
    # add new columns into housing full data set and find out the correlation
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]
    corr_matrix = housing.corr()
    # print(corr_matrix["median_house_value"].sort_values(ascending=False))

    # Prepare the data for Machine Learning algorithms

    # data clean
    housing = strat_train_set.drop("median_house_value", axis=1)  # drop label for training set
    housing_labels = strat_train_set["median_house_value"].copy()  # make label with median house value
    # print(housing_labels.sort_values(ascending=True))

    # show the missing items and their rows
    sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
    # print(sample_incomplete_rows)

    # drop

    # option 1: drop the rows or columns with missing items. do NOT affect sample incomplete rows
    sample_rows = sample_incomplete_rows.dropna(subset=["total_bedrooms"], how='any')
    # print(sample_rows)  # nothing will be printed.

    # option 2: drop the rows or columns specified. do NOT affect sample incomplete rows
    sample_rows = sample_incomplete_rows.drop("total_bedrooms", axis=1)
    # print(sample_rows)

    # option 3: fill the missing items with values. affect sample incomplete rows
    median = housing["total_bedrooms"].median()
    sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True)
    # print(sample_incomplete_rows)

    # option 4:
    imputer = Imputer(strategy="median")  # set imputer to "median" computing mode
    housing_num = housing.drop('ocean_proximity', axis=1)  # drop the non-number column
    # alternatively: housing_num = housing.select_dtypes(include=[np.number])
    # Only train set data could be put into fit()
    IMPUTER_FIT = imputer.fit(housing_num)  # put housing_num into imputer to compute median values of each column
    # print(IMPUTER_FIT)  # output: Imputer(axis=0, copy=True, missing_values='NaN', strategy='median', verbose=0)
    # print(imputer.statistics_)  # show the median values of each column
    # print(housing.median().values)  # the same result as the one above
    IMPUTER_TRANSFORM = imputer.transform(housing_num)  # generate plain number list IMPUTER_TRANSFORM
    # print(IMPUTER_TRANSFORM)  # plain number list. Not pandas data frame.
    housing_tr = pd.DataFrame(IMPUTER_TRANSFORM, columns=housing_num.columns,
                              index=list(housing.index.values))  # turn housing num back to pandas data frame
    # print(housing_tr.loc[sample_incomplete_rows.index.values])  # loc[index]: output incomplete rows due to [index]
    # print(housing_tr.head())  # output all data set (head: first 5 rows)

    # handling text

    housing_cat = housing[['ocean_proximity']]  # return dataframe's ocean proximity column
    # housing_cat_1d = housing['ocean_proximity']  # dict property: return series data type
    # print(housing_cat.head(10))
    # print(housing_cat_1d.head(10))

    ordinal_encoder = OrdinalEncoder()  # make an encoder. ordinal: encode one by one in orders.
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
    # print(housing_cat_encoded[:10])  # show the first ten encoded non-number values
    # print(ordinal_encoder.categories_)  # show the categories unchanged

    cat_encoder = OneHotEncoder(sparse=True)  # return a sparse or dense array.
    #  make the 0/1 one hot encoder. 1:value==category; 0:value!=category.
    # sparse array: show a tuple and a bool: (index, category number value), 1 or 0
    # dense array: show 2d matrix. each row get a list of categories. only 1 element ==1, else ==0
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    # print(housing_cat_1hot)
    # print(housing_cat_1hot.toarray())

    # standardscaler

    standardscaler = StandardScaler(copy=True, with_mean=True, with_std=True)  # define a scalar
    STANDARDSCALER_FIT = standardscaler.fit(housing_tr)  # compute mean and std of housing_tr (train set)
    # print(STANDARDSCALER_FIT)  # return: StandardScaler(copy=True, with_mean=True, with_std=True)
    # print(STANDARDSCALER_FIT.mean_)  # return the mean of each column
    # print(STANDARDSCALER_FIT.var_)  # return the fangcha of each column
    # print(STANDARDSCALER_FIT.scale_)  # return the relative scaling of each column
    STANDARDSCALER_TRANSFORM = standardscaler.transform(housing_tr)
    # print(STANDARDSCALER_TRANSFORM)  # generate plain array which will be used to train the test sets
    housing_tr = pd.DataFrame(STANDARDSCALER_TRANSFORM, columns=housing_num.columns,
                              index=list(housing.index.values))  # turn housing num back to pandas data frame
    # print(housing_tr.head())  # output all data set (head: first 5 rows)

    # Transformer

    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=True)
    housing_extra_attribs = attr_adder.transform(housing.values)  # numpy return a plain array
    housing_extra_attribs = pd.DataFrame(
        housing_extra_attribs,
        columns=list(housing.columns) + ["rooms_per_household", "population_per_household", "bedrooms_per_room"])
    # make it Data frame
    # print(housing_extra_attribs.head())  # new added columns are shown following the original data frame

    # Pipeline
    # All steps except the last one should be transformer. The last step could be fit, transformer etc.

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    # put the selector, imputer, transformer and scalar into one pipeline
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),  # return number type values
        ('imputer', Imputer(strategy="median")),  # compute the median values and put it into NA box
        ('attribs_adder', CombinedAttributesAdder()),  # add new columns
        ('std_scaler', StandardScaler(with_mean=False)),  # define a scaler to compute mean and std for train set
    ])
    # housing_num_tr = num_pipeline.fit_transform(housing_num)  # transform the number type housing set

    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', OneHotEncoder(sparse=False)),  # make this non-number type value into 1/0 bool
    ])

    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])  # unite two pipelines into one

    housing_prepared = full_pipeline.fit_transform(housing)  # fit and transform in the full pipeline
    # print(housing_prepared)  # np return a plain array
    # print(housing_prepared.shape)  # shape returns a tuple: (rows count, columns count)
    # shape returns (16512, 16): 16512 rows, 16 columns
    # 16 columns as below: 11 housing number type columns + 5 categories of non-number
    housing_prepared_df = pd.DataFrame(
                            housing_prepared,
                            columns=list(housing_num.columns) + ['rooms_per_household', 'population_per_household',
                                                                 'bedrooms_per_room', '<1H OCEAN', 'INLAND', 'ISLAND',
                                                                 'NEAR BAY', 'NEAR OCEAN'])
    # print(housing_prepared_df.head())  # convert np plain array into pd dataframe

    # Train model
    # Linear Regression

    lin_reg = LinearRegression()
    LIN_REG_FIT = lin_reg.fit(housing_prepared, housing_labels)  # fit(X, y)
    # compute the coef(a) and intercept(b) of the data prepared previously. a, b refer to explanation below

    # print(LIN_REG_FIT)  # return: LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    # print(LIN_REG_FIT.coef_)  # coefficients: y=aX+b, a means coefficient
    # print(LIN_REG_FIT.intercept_)  # intercept: y=aX+b, b means intercept

    # pick up some data for testing the linear regression predictor
    some_data = housing.iloc[:5]  # the first 5 rows (if loc[:5]: the first row to the row whose index is 5)
    some_labels = housing_labels.iloc[:5]  # the first 5 rows' median house value (refer to housing label's definition)

    some_data_prepared = full_pipeline.transform(some_data)  # prepare the data for testing via pipeline defined before
    print("Predictions:", lin_reg.predict(some_data_prepared))  # some_data_prepared=X, y is defined in fit() above
    # the output of lig_reg.predict(X): new y
    print("Labels:", list(some_labels))  # some_labels: y