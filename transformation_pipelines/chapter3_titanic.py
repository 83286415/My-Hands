import os
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Imputer
from sklearn.preprocessing.future_encoders import OneHotEncoder
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


WORKING_PATH = os.path.abspath(os.path.join(os.getcwd(), '..'))
ROOT_PATH = os.path.join(WORKING_PATH, 'Hands on SK and TS\\')
DATA_PATH = os.path.join(ROOT_PATH, 'datasets\\')
TITANIC_PATH = os.path.join(DATA_PATH, "titanic")


def load_titanic_data(filename, titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path, filename)
    return pd.read_csv(csv_path)


# A class to select numerical or categorical columns
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]


# fill the blank with the most frequent used data
# Inspired from stackoverflow.com/questions/25239958
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)  # pd.Series(data, index)
        # value_counts().index[0]: the most frequent; .index[1]: the second frequent
        return self

    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


if __name__ == '__main__':

    # load data
    train_data = load_titanic_data("train.csv")
    test_data = load_titanic_data("test.csv")

    # peek the top of the train data
    print('the top of the train data:')
    print(train_data.head())
    print('----------')

    # the missing data
    print('the missing data:')
    print(train_data.info())  # Age 714, Cabin 204, Embarked 889, Else 891
    print('----------')

    # numeric attributes
    print('numeric attributes:')
    print(train_data.describe())
    print('----------')

    # target check
    print('target check')
    print(train_data['Survived'].value_counts())  # 0: 549 dead;    1: 342 survived
    print('----------')

    # pipeline
    imputer = Imputer(strategy="median")

    num_pipeline = Pipeline([  # numeric pipeline
        ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),  # get the numeric columns
        ("imputer", Imputer(strategy="median")),  # compute their mean
    ])
    numeric_attributes = num_pipeline.fit_transform(train_data)
    print('numeric pipeline')
    print(numeric_attributes)
    print('----------')

    cat_pipeline = Pipeline([  # non-numeric pipeline
        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
        ("imputer", MostFrequentImputer()),  # fill the blank with most frequent data
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])  # make the 0/1 one hot encoder. 1:value==category; 0:value!=category.
    cat_attributes = cat_pipeline.fit_transform(train_data)
    print('cat pipeline with one hot encoder according their categories')
    print(cat_attributes)
    print('----------')

    preprocess_pipeline = FeatureUnion(transformer_list=[  #
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

    # train/test data and label by pipeline
    X_train = preprocess_pipeline.fit_transform(train_data)
    y_train = train_data['Survived']

    X_test = preprocess_pipeline.fit_transform(test_data)
    # y_test = test_data['Survived']  # no survived column in test data excel file

    # train a classifier
    svm_clf = SVC()
    svm_clf.fit(X_train, y_train)

    # prediction on test set
    y_pred = svm_clf.predict(X_test)
    print('predictions:')
    print(y_pred)
    print('----------')

    # evaluate the classifier
    scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
    print('SVM classifier cross validation score:')
    print(scores.mean())  # 73.65%  That is not good at all.
    print('----------')

    # try another way to higher score
    forest_clf = RandomForestClassifier(random_state=42)
    scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
    print('random forest classifier cross validation score:')
    print(scores.mean())  # 81.157% Better than svm
    print('----------')

    print('Here above is the conclusion of the first 3 chapters in hands on ML')
    print('After finishing the later chapters, I will come back and improve the accuracy.')