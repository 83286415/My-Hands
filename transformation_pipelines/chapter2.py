import os
import tarfile
import time
import pandas as pd
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from six.moves import urllib
from pandas.plotting import scatter_matrix
from scipy.stats import randint
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.preprocessing.future_encoders import OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


WORKING_PATH = os.path.abspath(os.path.join(os.getcwd(), '..'))
ROOT_PATH = os.path.join(WORKING_PATH, 'Hands on SK and TS\\')
HOUSING_TGZ_PATH = ROOT_PATH + "datasets\\housing\\housing.tgz"
HOUSING_PATH = ROOT_PATH + "datasets\\housing\\"
HOUSING_SAVED_PATH = ROOT_PATH + "Model saved\\"


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
        population_per_household = data_set[:, population_ix] / data_set[:, household_ix]  # the people count in family
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


def display_scores(_scores):
    print("Scores:", _scores)
    print("Mean:", _scores.mean())
    print("Standard deviation:", _scores.std())


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


def save_model(model, model_name, time_stamp_gate=False):
    if time_stamp_gate:
        # time gate to decide the time stamp added into file name or not
        time_stamp = time.strftime('_%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
        model_saved_path = HOUSING_SAVED_PATH + model_name + time_stamp + ".pkl"
    else:
        model_saved_path = HOUSING_SAVED_PATH + model_name + ".pkl"
    if not os.path.exists(model_saved_path):
        joblib.dump(model, model_saved_path)
    else:
        print("Cannot save your model %s. Change its name and try it again." % model_name)
        exit()


def load_model(model_name, load_mode=None):
    # load_mode: refer to joblib.load
    model_saved_path = HOUSING_SAVED_PATH + model_name + ".pkl"
    try:
        model_loaded = joblib.load(model_saved_path, load_mode)
        return model_loaded
    except FileNotFoundError:
        print("Cannot load the file %s. Check it again." % model_name)
        exit()


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
    plt.show()

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
    # standardscaler's effect refer to chapter3.py line 338

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
    # the "housing" is the train data which drops the "train label" median_house_value

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
    # print("Predictions:", lin_reg.predict(some_data_prepared))  # some_data_prepared=X, y is defined in fit() above
    some_data_predictions = lin_reg.predict(X=some_data_prepared)
    # some_data_predictions: the output of lig_reg.predict(X): new y
    # print("Labels:", list(some_labels))  # some_labels: y

    # predict and check the RMSE & MAE results
    housing_predictions = lin_reg.predict(X=housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)  # mean_squared_error(y, new_y)
    lin_rmse = np.sqrt(lin_mse)  # root-mean-square error
    # print(lin_rmse)  # the RMSE 68628 is too big. So the model under fits the train data set.
    # The best result of RMSE is 0.
    lin_mae = mean_absolute_error(housing_labels, housing_predictions)  # mean_absolute_error(y, new_y)
    # print(lin_mae)  # the MAE is 49439, not satisfying either.

    # Decision Tree Regression

    tree_reg = DecisionTreeRegressor(random_state=42)  # decision tree regressor is in chapter 6
    TREE_REG_FIT = tree_reg.fit(housing_prepared, housing_labels)  # fit(X, y)
    # print(TREE_REG_FIT)
    # FIT output: DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
    #                                   max_leaf_nodes=None, min_impurity_decrease=0.0,
    #                                   min_impurity_split=None, min_samples_leaf=1,
    #                                   min_samples_split=2, min_weight_fraction_leaf=0.0,
    #                                   presort=False, random_state=42, splitter='best')
    housing_predictions = tree_reg.predict(housing_prepared)  # housing_predictions: new y
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    # print(tree_rmse)  # result is 0, perfect?
    tree_mae = mean_absolute_error(housing_labels, housing_predictions)  # mean_absolute_error(y, new_y)
    # print(tree_mae)  # result is 0, perfect?

    # Random Forest Regression

    forest_reg = RandomForestRegressor(random_state=42)
    FOREST_REG_FIT = forest_reg.fit(housing_prepared, housing_labels)
    # print(FOREST_REG_FIT)  # the output of this fit:
    # RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
    #                       max_features='auto', max_leaf_nodes=None,
    #                       min_impurity_decrease=0.0, min_impurity_split=None,
    #                       min_samples_leaf=1, min_samples_split=2,
    #                       min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
    #                       oob_score=False, random_state=42, verbose=0, warm_start=False)
    housing_predictions = forest_reg.predict(housing_prepared)  # predict new y
    forest_mse = mean_squared_error(housing_labels, housing_predictions)
    forest_rmse = np.sqrt(forest_mse)
    # print(forest_rmse)  # the RMSE result is 21993

    # support vector regression - SVR (refer to chapter 5 SVM)

    svm_reg = SVR(kernel="linear")  # define the kernel trick
    svm_reg.fit(housing_prepared, housing_labels)  # train
    housing_predictions = svm_reg.predict(housing_prepared)  # compute new y
    svm_mse = mean_squared_error(housing_labels, housing_predictions)
    svm_rmse = np.sqrt(svm_mse)
    # display_scores(svm_rmse)
    # score:111094 mean: 111094 std: 0.0

    # Cross-Validation

    # cross validate tree decision regression
    scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)  # cv: the count of folds and validation times
    tree_rmse_scores = np.sqrt(-scores)  # Hands on P70: score is negative, opposite of the MSE
    # display_scores(tree_rmse_scores)  # show score, mean and std
    # mean: 71773, std: 2531

    # cross validate linear regression
    lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                                 scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    # display_scores(lin_rmse_scores)  # the tree_decision_reg (over fitting) is worse than the lin_reg
    # mean: 69053, std: 2732

    # cross validate random forest regression
    forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                    scoring="neg_mean_squared_error", cv=10)
    forest_rmse_scores = np.sqrt(-forest_scores)
    # display_scores(forest_rmse_scores)  # the result shows the random forest regression is the best predictor
    # mean: 52612, std: 2302

    # cross validate SVR
    svm_scores = cross_val_score(svm_reg, housing_prepared, housing_labels,
                                 scoring="neg_mean_squared_error", cv=10)
    svm_rmse_scores = np.sqrt(-svm_scores)
    # display_scores(svm_rmse_scores)
    # mean: 111809 std: 2762

    # save and load model (P71): example shown as below

    # save_model(forest_rmse_scores, "forest_rmse_scores")
    # scores_loaded = load_model("forest_rmse_scores")
    # display_scores(scores_loaded)

    # fine tune

    # grid search (chapter 7) - used in small search space

    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {'n_estimators': [10, 30, 60], 'max_features': [4, 6, 8, 10]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {'bootstrap': [False], 'n_estimators': [10, 90], 'max_features': [3, 4, 5]},
    ]  # define the hyper parameter: detailed info refer to chapter 7
    # i changed the estimator and max features parameters above.
    # Now the best params: {'bootstrap': False, 'max_features': 4, 'n_estimators': 90}

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                               scoring='neg_mean_squared_error', return_train_score=True, refit=True)
    # refit: when found the best estimator, re-train the whole train data set at once
    grid_search.fit(housing_prepared, housing_labels)
    # print(grid_search.best_estimator_)  # the best forest_reg searched
    # print(grid_search.best_score_)
    # print(grid_search.best_params_)  # will show the best combination of max feature and n_esitmators
    # best params output:
    # {'bootstrap': False, 'max_features': 4, 'n_estimators': 90} can be used as input of SVR(**rnd_search.best_params_)
    # print(grid_search.best_estimator_)
    # output: RandomForestRegressor(bootstrap=False, criterion='mse', max_depth=None,
    #        max_features=4, max_leaf_nodes=None, min_impurity_decrease=0.0,
    #        min_impurity_split=None, min_samples_leaf=1,
    #        min_samples_split=2, min_weight_fraction_leaf=0.0,
    #        n_estimators=90, n_jobs=1, oob_score=False, random_state=42,
    #        verbose=0, warm_start=False)
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):  # zip: more output in one loop
        # the attributes in cv_results_ refer to the comments of the function GridSearchCV
        # print(np.sqrt(-mean_score), params)
        pass
    # output:
    #     52756.81345142999{'max_features': 4, 'n_estimators': 10}
    #     50393.610091085786{'max_features': 4, 'n_estimators': 30}
    #     49818.13921915004{'max_features': 4, 'n_estimators': 60}
    #     52010.94743275823{'max_features': 6, 'n_estimators': 10}
    #     50138.721309007444{'max_features': 6, 'n_estimators': 30}
    #     49540.62257720069{'max_features': 6, 'n_estimators': 60}
    #     51735.028037950986{'max_features': 8, 'n_estimators': 10}
    #     49707.90394629965{'max_features': 8, 'n_estimators': 30}
    #     49210.35061639113{'max_features': 8, 'n_estimators': 60}
    #     52270.43569062347{'max_features': 10, 'n_estimators': 10}
    #     50250.89980971333{'max_features': 10, 'n_estimators': 30}
    #     49649.03197562726{'max_features': 10, 'n_estimators': 60}
    #     52726.36568489697{'bootstrap': False, 'max_features': 3, 'n_estimators': 10}
    #     49658.133497518174{'bootstrap': False, 'max_features': 3, 'n_estimators': 90}
    #     50986.86163811137{'bootstrap': False, 'max_features': 4, 'n_estimators': 10}
    #     48703.08873596659{'bootstrap': False, 'max_features': 4, 'n_estimators': 90}
    #     52086.12659387112{'bootstrap': False, 'max_features': 5, 'n_estimators': 10}
    #     48772.63432878675{'bootstrap': False, 'max_features': 5, 'n_estimators': 90}

    cv_results_pd = pd.DataFrame(grid_search.cv_results_)  # cv_results_ can be packed into pd data frame easily
    # print(cv_results_pd.head())

    # random search - used in large search space

    param_distribs = {
        'n_estimators': randint(low=1, high=200),  # make a random number between the range defined
        'max_features': randint(low=1, high=8),
    }  # define the hyper parameter

    forest_reg = RandomForestRegressor(random_state=42)  # define a regression
    rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                    n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
    # n_iter: the number of the different values for hyper parameter in search running (P74)
    rnd_search.fit(housing_prepared, housing_labels)
    cvres = rnd_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        # print(np.sqrt(-mean_score), params)
        pass
        # output:  10 iterations: best param 7*180
        # 49157.171340988294{'max_features': 7, 'n_estimators': 180}
        # 51413.92656821079{'max_features': 5, 'n_estimators': 15}
        # 50805.51558518083{'max_features': 3, 'n_estimators': 72}
        # 50856.794284175405{'max_features': 5, 'n_estimators': 21}
        # 49290.97465838148{'max_features': 7, 'n_estimators': 122}
        # 50782.43794687328{'max_features': 3, 'n_estimators': 75}
        # 50689.00847519324{'max_features': 3, 'n_estimators': 88}
        # 49618.00702383847{'max_features': 5, 'n_estimators': 100}
        # 50473.304986200324{'max_features': 3, 'n_estimators': 150}
        # 64428.13365969648{'max_features': 5, 'n_estimators': 2}

    # find out the importance of the regression

    feature_importances = grid_search.best_estimator_.feature_importances_
    # feature_importances: the related importance of 16 columns to the regression defined
    extra_attribs = ["rooms_per_household", "pop_per_household", "bedrooms_per_room"]  # new added features
    cat_encoder = cat_pipeline.named_steps["cat_encoder"]
    cat_one_hot_attribs = list(cat_encoder.categories_[0])  # ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY','NEAR OCEAN']
    attributes = num_attribs + extra_attribs + cat_one_hot_attribs  # total 16 columns features
    sorted_result = sorted(zip(feature_importances, attributes), reverse=True)  # reverse=True: descending order
    # print(sorted_result)
    # output:
    # [(0.2857861152762817, 'median_income'), (0.1324122696549764, 'INLAND'), (0.09909466430417471,'pop_per_household'),
    #  (0.09277095010645679, 'longitude'), (0.09036715907531163, 'bedrooms_per_room'), (0.08350703833591074,'latitude'),
    #  (0.060899454099257934, 'rooms_per_household'), (0.04183603293822083, 'housing_median_age'),
    #  (0.022014693286380782, 'population'), (0.02120289808247002, 'total_rooms'),
    #  (0.019426620755410805, 'total_bedrooms'), (0.018613324304780684, 'households'),(0.01836309038370023,'<1H OCEAN'),
    #  (0.007851413742625421, 'NEAR OCEAN'), (0.005784846540272982, 'NEAR BAY'), (6.942911376821082e-05, 'ISLAND')]

    # Test: evaluate the final model on test set

    final_model = grid_search.best_estimator_  # the best regression (just like lin_reg or svr_reg)
    X_test = strat_test_set.drop("median_house_value", axis=1)  # drop y from test data set X_test
    y_test = strat_test_set["median_house_value"].copy()  # y: labels
    X_test_prepared = full_pipeline.transform(X_test)  # prepare X_test
    final_predictions = final_model.predict(X_test_prepared)  # compute new y based on prepared X_test set
    final_mse = mean_squared_error(y_test, final_predictions)
    # display_scores(final_mse)
    #     output:
    #     Mean: 2128657156.94195
    #     Standard deviation: 0.0
    final_rmse = np.sqrt(final_mse)
    # display_scores(final_rmse)
    #       output:
    #       Mean: 46137.372670557976
    #       Standard deviation: 0.0
    print('Chapter2 Done.')