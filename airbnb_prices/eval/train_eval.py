import logging

import numpy as np
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

logging.basicConfig(level=logging.INFO)


def train_eval_once(
    model, x_train: DataFrame, y_train: DataFrame, x_test: DataFrame, y_test: DataFrame
):
    """_Train a model and evaluate it on the test set.

    Args:
        model : _model to train_
        x_train (DataFrame): train set
        y_train (DataFrame): train set target_
        x_test (DataFrame): _test set
        y_test (DataFrame): _test set target

    Returns:
        _model: trained model
    """
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = mean_squared_error(y_test, y_pred)
    logging.info(" MSE score on test set: {}".format(score))
    return model, score


def cross_validation(model, x: DataFrame, y: DataFrame):
    """Perform a 5-fold crossvalidation with the model.

    Args:
        model : model to train and evaluate
        x (DataFrame): _train set
        y (DataFrame): _train set target_

    Returns:
        _score: crossvalidation score. It is the average of the
            evaluation scores._
    """
    # list to keep track of the scores
    cv_scores = []
    # 5-fold crossvalidation
    kfold = KFold(n_splits=5)
    for train_index, test_index in kfold.split(x):
        x_train, y_train = x.iloc[train_index], y.iloc[train_index]
        x_test, y_test = x.iloc[test_index], y.iloc[test_index]
        _, score = train_eval_once(
            model=model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
        )
        cv_scores.append(score)
    return np.mean(cv_scores)
