import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from .chapter2 import housing_labels, housing_prepared, display_scores


if __name__ == '__main__':

    # practice 1

    svm_reg = SVR(kernel="linear")  # define the kernel trick
    svm_reg.fit(housing_prepared, housing_labels)  # train
    housing_predictions = svm_reg.predict(housing_prepared)  # compute new y
    svm_mse = mean_squared_error(housing_labels, housing_predictions)
    svm_rmse = np.sqrt(svm_mse)
    display_scores(svm_rmse)
    # score:111094 mean: 111094 std: 0.0