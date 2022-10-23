from pydoc import locate
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
    model_lower = model.lower()
    if model_lower in ("random_forest", "rf"):
        return ensemble.RandomForestRegressor(**hyperparams)
    if model_lower in ("extreme_random_forest", "erh"):
        return ensemble.ExtraTreesRegressor(**hyperparams)
    if model_lower == "adaboost":
        return ensemble.AdaBoostRegressor(**hyperparams)
    if model_lower == ("gradient_boosting", "gb"):
        return ensemble.GradientBoostingRegressor(**hyperparams)
    for module in ["linear_model", "svm", "ensemble", "neighbors", "neural_network"]:
        model_obj = locate(f"sklearn.{module}.{model}")
        if model_obj is not None:
            return model_obj(**hyperparams)
    raise ValueError(f"Model {model} not found")
