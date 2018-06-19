import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.preprocessing.future_encoders import OrdinalEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
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
    ])  # unite two pipelines into one

    housing_prepared = full_pipeline.fit_transform(housing)

    # Q1:
    svm_reg = SVR(kernel="rbf", C=10, gamma="auto")
    svm_reg.fit(housing_prepared, housing_labels)
    housing_predictions = svm_reg.predict(housing_prepared)
    svm_mse = mean_squared_error(housing_labels, housing_predictions)
    svm_rmse = np.sqrt(svm_mse)
    display_scores(svm_rmse)
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
        ("predict_pipeline", predict_pipeline),
        ("full_pipeline", full_pipeline),
    ])
    final_result = final_pipeline.transform(housing_prepared)
    # TODO: Q4 is half-baked.

    # Q5
    # n_jobs : int, default=1
    #     Number of jobs to run in parallel.
    # verbose : integer
    #     Controls the verbosity: the higher, the more messages.