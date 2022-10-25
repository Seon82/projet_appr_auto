"""Run the airbnb prices detection system."""

import logging
from pathlib import Path

import click
import numpy as np

from airbnb_prices.data import DataPipeline
from airbnb_prices.eval import train_eval
from airbnb_prices.models import models

CONFIG_PATH = Path("./examples/config.json")
DATA_PATH = Path("./data/train_airbnb_berlin.xls")


def create_logger(log_level: str | int, name: str = None) -> logging.Logger:
    """
    Generate a logger object with an attached console handler.

    Args:
        log_level: CRITICAL|ERROR|INFO|DEBUG
        name: The logger's name. All logs from submodules will be captured by this logger.
    """
    if isinstance(log_level, str):
        try:
            log_level = getattr(logging, log_level)
        except AttributeError as error:
            raise ValueError(f"{log_level} isn't a valid logging level.") from error
    logger = logging.getLogger(name=name)
    logger.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("[%(levelname)s] (%(name)s) - %(message)s")
    console_handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(console_handler)
    return logger


def hyper_to_dict(hyper: str):
    """Convert string of hyperparameters to a dictionary."""
    if not hyper:
        return {}
    # Remove whitespaces
    hyper = hyper.strip()
    hyper = hyper.replace(" ", "")
    hyper_blocks = hyper.split(",")
    hyper_dict = {}
    for block in hyper_blocks:
        block_list = block.split("=")
        if block_list[1][0].isdigit():
            hyper_dict[block_list[0]] = int(block_list[1])
        elif block_list[1].isdigit():
            hyper_dict[block_list[0]] = float(block_list[1])
        else:
            hyper_dict[block_list[0]] = block_list[1]
    return hyper_dict


@click.command()
@click.argument("model")
@click.option(
    "--hyperparameters",
    "--hyper",
    "--params",
    help="Hyperparameters to be passed to the model: \"n_estimators=100,criterion='gini'\"",
)
@click.option(
    "--verbosity",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Verbosity level.",
)
@click.option(
    "--crossvalidation",
    "--cv",
    is_flag=True,
    default=False,
    help="Evaluate the model with crossvalidation",
)
@click.option(
    "--no-export",
    is_flag=True,
    default=False,
    help="Do not save the results to a csv file.",
)
@click.option(
    "--data",
    type=click.Path(dir_okay=False),
    default=DATA_PATH,
    help="Path to the data csv.",
)
@click.option(
    "--config",
    type=click.Path(dir_okay=False),
    default=CONFIG_PATH,
    help="Path to the configuration json.",
)
def main(
    model: str,
    hyperparameters: str,
    verbosity: str,
    crossvalidation: bool,
    no_export: bool,
    data: Path,
    config: Path,
):
    logger = create_logger(verbosity, "airbnb_prices")
    model_name = model
    logger.info("Loading the dataset ...")
    pipeline = DataPipeline.from_file(data, config)
    logger.info("Replacing missing data ...")
    pipeline.infere_nan()
    logger.info("Feature engineering ...")
    pipeline.apply_feature_engineering()
    logger.info("Preprocessing ...")
    pipeline.apply_preprocessing()
    pipeline.dropna()
    # Load the model
    hyperparams = hyper_to_dict(hyperparameters)
    model = models.get_model(model=model, hyperparams=hyperparams)
    logger.info("Training phase ...")
    # Training and evaluation
    X_train, y_train = pipeline.train_data
    X_val, y_val = pipeline.val_data
    if not crossvalidation:
        _, train_score, val_score = train_eval.train_eval_once(
            model=model, x_train=X_train, y_train=y_train, x_test=X_val, y_test=y_val
        )
        test_score = np.nan

    if crossvalidation:
        train_score, val_score = train_eval.cross_validation(model=model, x=X_train, y=y_train)
        _, _, test_score = train_eval.train_eval_once(
            model=model,
            x_train=X_train,
            y_train=y_train,
            x_test=X_val,
            y_test=y_val,
            crossvalidation=crossvalidation,
        )
        logger.info("RMSE on the test data: {}".format(test_score))

    if not no_export:
        logger.info("Exporting to csv ...")
        train_eval.export_results_to_csv(
            model_name=model_name,
            hyperparameters=hyperparameters,
            train_score=train_score,
            val_score=val_score,
            crossvalidation=crossvalidation,
            test_score=test_score,
        )
    logger.info("Done")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
