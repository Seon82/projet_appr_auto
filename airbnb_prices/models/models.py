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
    model = model.lower()
    if model in ("random_forest", "rf"):
        return ensemble.RandomForestRegressor(**hyperparams)
    if model in ("extreme_random_forest", "erh"):
        return ensemble.ExtraTreesRegressor(**hyperparams)
    if model == "adaboost":
        return ensemble.AdaBoostRegressor(**hyperparams)
    if model == ("gradient_boosting", "gb"):
        return ensemble.GradientBoostingRegressor(**hyperparams)
