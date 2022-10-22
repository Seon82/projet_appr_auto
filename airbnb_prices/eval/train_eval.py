import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


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
    score = np.sqrt(mean_squared_error(y_test, y_pred))
    logger.info(" RMSE score on test set: {}".format(score))
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


def export_results_to_csv(model_name: str, hyperparameters: str, score: float):
    """Generate a csv file to keep track of the results.

    Args:
        model_name (str): _name of the model
        hyperparameters (str): _string of hyperparameters
        score (float): RMSE score on the test set
    """
    results_dict = {"Model name": [model_name], "Hyperparameters": [hyperparameters], "RMSE": [score]}
    results_df = DataFrame.from_dict(results_dict)
    results_dir = Path("./results")
    if not results_dir.exists():
        results_dir.mkdir()
    date_str = datetime.now().strftime("%m%d%Y%H%M%S")
    results_df.to_csv(results_dir / (date_str + "_results.csv"))
