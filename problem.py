import os

import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error


problem_title = 'Prediction of success of application on app store'
_target_column_name = 'success'
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()
# An object implementing the workflow
workflow = rw.workflows.EstimatorExternalData()


class RMSE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='rmse', precision=5):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return rmse 


score_types = [
    RMSE(name='rmse', precision=5),
]


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=8, test_size=0.20, random_state=42)
    return cv.split(X, y)


def _read_data(path, f_name):
    data = pd.read_json(os.path.join(path, 'data', f_name))
    data = data.drop(data[data.success == 0].index)
    y_array = np.log( 1 + data[_target_column_name].values)
    X_df = data.drop(_target_column_name, axis=1)
    return X_df, y_array


def get_train_data(path='./data/'):
    f_name = 'train.json'
    return _read_data(path, f_name)


def get_test_data(path='./data/'):
    f_name = 'test.json'
    return _read_data(path, f_name)

def contentAdvisoryRating_recode(val):
    if val=='4_plus':
        output = 1
    elif val == '9_plus':
        output = 2
    elif val == '12_plus':
        output = 3
    else:
        output = 4
    return output