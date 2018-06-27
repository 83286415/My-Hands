import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.preprocessing.future_encoders import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, reciprocal
from chapter2 import display_scores, load_housing_data, HOUSING_PATH, DataFrameSelector, CombinedAttributesAdder


class MostImportantSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, data_set, data_set_label=None):
        return self

    def transform(self, data_set):
        return data_set[self.attribute_names].values


class Predictor(BaseEstimator, TransformerMixin):
    def __init__(self, data_set_label):
        self.svm_reg = SVR(kernel="rbf", C=10, gamma="auto")
        self.data_set_label = data_set_label

    def fit(self, data_set, data_set_label=None):
        self.svm_reg.fit(data_set, data_set_label)
        return self

    def transform(self, data_set):
        prediction = self.svm_reg.predict(data_set)
        # return data_set[self.attribute_names].values
        svm_mse = mean_squared_error(self.data_set_label, prediction)
        svm_rmse = np.sqrt(svm_mse)
        display_scores(svm_rmse)
        print("Q4")
        return data_set.values


def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])  # sort: make a new copy with ascending order (default)
    # numpy.argpartition refer to my notebook or https://blog.csdn.net/weixin_37722024/article/details/64440133
    # here return the first top index of larger elements


class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k

    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self

    def transform(self, X):
        return X[:, self.feature_indices_]  # return those more important columns data set


if __name__ == '__main__':

    # prepare data input
    housing = load_housing_data(housing_path=HOUSING_PATH)
    # strat_train_set = housing_base_data
    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  # spliter of full data set
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
    housing_num = housing.drop('ocean_proximity', axis=1)


    # pipeline
    num_attribs = list(housing_num)
    most_important_attribs =["median_income"]
    cat_attribs = ["ocean_proximity"]

    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler(with_mean=False)),
    ])

    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', OneHotEncoder(sparse=False)),
    ])

    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

    housing_prepared = full_pipeline.fit_transform(housing)
    # print(housing_prepared)

    # Q1:
    svm_reg = SVR(kernel="rbf", C=10, gamma="auto")
    svm_reg.fit(housing_prepared, housing_labels)
    housing_predictions = svm_reg.predict(housing_prepared)
    svm_mse = mean_squared_error(housing_labels, housing_predictions)
    svm_rmse = np.sqrt(svm_mse)
    # display_scores(svm_rmse)
    # output:
    # kernel=linear      C=1        score:106874       mean: 106874        std: 0.0     linear default
    # kernel=linear      C=10        score:79574       mean: 79574        std: 0.0      increase C to 10
    # kernel=rbf      C=1       gamma=auto        score:118448       mean: 118448        std: 0.0      rbf default
    # kernel=rbf      C=10       gamma=auto        score:114211       mean: 114211        std: 0.0      increase C to 10

    # Q2
    # already done in chapter2.py line 572

    # Q3
    important_pipeline = Pipeline([
        ('selector', MostImportantSelector(most_important_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler(with_mean=False)),
    ])

    # Q4
    predict_pipeline = Pipeline([
        ('predictor', Predictor(housing_labels))])
    final_pipeline = FeatureUnion(transformer_list=[
        ("full_pipeline", full_pipeline),
        ("predict_pipeline", predict_pipeline),
    ])
    # final_result = final_pipeline.fit_transform(housing, housing_labels)
    # TODO: Q4 is half-baked.

    # Q5
    # n_jobs : int, default=1
    #     Number of jobs to run in parallel.
    # verbose : integer
    #     Controls the verbosity: the higher, the more messages.

    # solution:

    # Q1
    param_grid = [
        {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
         'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]  # multiple kernels and other params could be put in a grid search list "param_grid"!

    svm_reg = SVR()
    grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=4)
    grid_search.fit(housing_prepared, housing_labels)
    # print(grid_search.cv_results_)
    negative_mse = grid_search.best_score_
    rmse = np.sqrt(-negative_mse)
    # print(rmse)
    # print(grid_search.best_params_)

    # Q2
    param_distribs = {
        'kernel': ['linear', 'rbf'],
        'C': reciprocal(20, 200000),  # 1/(log(200000/20))
        'gamma': expon(scale=1.0),  # f(x) = λexp(-λx)   scale:λ
    }

    svm_reg = SVR()
    rnd_search = RandomizedSearchCV(svm_reg, param_distributions=param_distribs,
                                    n_iter=50, cv=5, scoring='neg_mean_squared_error',
                                    verbose=2, n_jobs=4, random_state=42)
    rnd_search.fit(housing_prepared, housing_labels)
    negative_mse = rnd_search.best_score_
    rmse = np.sqrt(-negative_mse)

    expon_distrib = expon(scale=1.)
    samples = expon_distrib.rvs(10000, random_state=42)  # rvs: draw sample
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.title("Exponential distribution (scale=1.0)")
    plt.hist(samples, bins=50)
    plt.subplot(122)
    plt.title("Log of this distribution")
    plt.hist(np.log(samples), bins=50)
    plt.show()

    reciprocal_distrib = reciprocal(20, 200000)
    samples = reciprocal_distrib.rvs(10000, random_state=42)
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.title("Reciprocal distribution (scale=1.0)")
    plt.hist(samples, bins=50)
    plt.subplot(122)
    plt.title("Log of this distribution")
    plt.hist(np.log(samples), bins=50)
    plt.show()

    # Q3
    k = 5
    feature_importances = ['longitude', 'latitude', 'median_income', 'pop_per_hhold', 'INLAND']
    preparation_and_feature_selection_pipeline = Pipeline([
        ('preparation', full_pipeline),
        ('feature_selection', TopFeatureSelector(feature_importances, k))
    ])
    housing_prepared_top_k_features = preparation_and_feature_selection_pipeline.fit_transform(housing)
    print(housing_prepared_top_k_features[0:3])

    # Q4
    prepare_select_and_predict_pipeline = Pipeline([
        ('preparation', full_pipeline),
        ('feature_selection', TopFeatureSelector(feature_importances, k)),
        ('svm_reg', SVR(**rnd_search.best_params_))  # add a regressor to predict =.=
    ])
    prepare_select_and_predict_pipeline.fit(housing, housing_labels)
    some_data = housing.iloc[:4]
    some_labels = housing_labels.iloc[:4]

    print("Predictions:\t", prepare_select_and_predict_pipeline.predict(some_data))
    print("Labels:\t\t", list(some_labels))

    # Q5
    param_grid = [
        {'preparation__num_pipeline__imputer__strategy': ['mean', 'median', 'most_frequent'],
         'feature_selection__k': list(range(1, len(feature_importances) + 1))}
    ]

    grid_search_prep = GridSearchCV(prepare_select_and_predict_pipeline, param_grid, cv=5,
                                    scoring='neg_mean_squared_error', verbose=2, n_jobs=4)
    grid_search_prep.fit(housing, housing_labels)
    print(grid_search_prep.best_params_)