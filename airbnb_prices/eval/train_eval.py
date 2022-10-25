import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


def train_eval_once(
    model,
    x_train: DataFrame,
    y_train: DataFrame,
    x_test: DataFrame,
    y_test: DataFrame,
    crossvalidation: bool = False,
):
    """_Train a model and evaluate it on the test set.

    Args:
        model : _model to train_
        x_train (DataFrame): train set
        y_train (DataFrame): train set target_
        x_test (DataFrame): _test set
        y_test (DataFrame): _test set target
        crossvalidation (bool): indicates whether crossvalidation is
            is used to evaluate the model or not

    Returns:
        model: trained model
        train_score: RMSE on the train set
        test_score: RMSE on the test set
    """
    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    train_score = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_score = np.sqrt(mean_squared_error(y_test, y_pred_test))
    if not crossvalidation:
        logger.info(" RMSE score on train set: {}".format(train_score))
        logger.info(" RMSE score on validation set: {}".format(test_score))
    return model, train_score, test_score


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
    cv_scores_train = []
    cv_scores_val = []
    # 5-fold crossvalidation
    kfold = KFold(n_splits=5)
    for train_index, test_index in kfold.split(x):
        x_train, y_train = x.iloc[train_index], y.iloc[train_index]
        x_test, y_test = x.iloc[test_index], y.iloc[test_index]
        _, train_score, val_score = train_eval_once(
            model=model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
        )
        cv_scores_train.append(train_score)
        cv_scores_val.append(val_score)
    cv_train_mean = np.mean(cv_scores_train)
    cv_test_mean = np.mean(cv_scores_val)
    return cv_train_mean, cv_test_mean


def export_results_to_csv(
    model_name: str,
    hyperparameters: str,
    train_score: float,
    val_score: float,
    crossvalidation: bool,
    test_score: float = np.nan,
):
    """Generate a csv file to keep track of the results.

    Args:
        model_name (str): _name of the model
        hyperparameters (str): _string of hyperparameters
        train_score (float): RMSE on the train set
        val_score (float): RMSE on the validation set
        crossvalidation (bool): indicates whether the model is evaluated with
            crossvalidation or not
        test_score (float): RMSE on the test set. It is set to NAN if crossvalidation is False.
            In this case, the test set is considered a validation set.

    """
    results_dict = {
        "Model name": [model_name],
        "Hyperparameters": [hyperparameters],
        "RMSE train": [train_score],
        "RMSE val": [val_score],
        "crossvalidation": [crossvalidation],
        "RMSE test": [test_score],
    }
    results_df = DataFrame.from_dict(results_dict)
    results_dir = Path("./results")
    if not results_dir.exists():
        results_dir.mkdir()
    date_str = datetime.now().strftime("%m%d%Y%H%M%S")
    results_df.to_csv(results_dir / (date_str + "_results.csv"))
