import logging
from pandas import DataFrame
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO)


def train_eval_once(
    model, x_train: DataFrame, y_train: DataFrame, x_test: DataFrame, y_test: DataFrame
):
    """_Train a model and eval it on the test set.

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
    return model
