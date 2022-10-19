from typing import Dict

import sklearn.ensemble as ensemble


def get_model(model: str, hyperparams: Dict):
    """Wrapper to easily get models.

    Args:
        model (str): chosen model name
        hyperparams (Dict): dictionary of hyperparameters

    Returns:
        chosen model object.
    """
    if model == "random forest":
        return ensemble.RandomForestRegressor(**hyperparams)
    if model == "extreme random forest":
        return ensemble.ExtraTreesRegressor(**hyperparams)
    if model == "adaboost":
        return ensemble.AdaBoostRegressor(**hyperparams)
    if model == "gradient boosting":
        return ensemble.GradientBoostingRegressor(**hyperparams)
