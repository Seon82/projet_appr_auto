"""Run the airbnb prices detection system."""
import argparse
import logging
from pathlib import Path

from airbnb_prices.data import dataset
from airbnb_prices.eval import train_eval
from airbnb_prices.models import models

CONFIG_PATH = Path("./examples/dataset/config.json")
DATA_PATH = Path("./data/train_airbnb_berlin.xls")


def hyper_to_dict(hyper: str):
    """Convert string of hyperparameters to a dictionary."""
    hyper = hyper.strip()
    hyper_blocks = hyper.split(",")
    hyper_dict = {}
    for block in hyper_blocks:
        block_list = block.split("=")
        if float(block_list[1]) % 1 == 0:
            hyper_dict[block_list[0]] = int(block_list[1])
        else:
            hyper_dict[block_list[0]] = float(block_list[1])
    return hyper_dict


def main(model_str: str, hyperparams_str: str):
    logging.info("Loading the dataset ...")
    config = dataset.load_config(CONFIG_PATH)
    data = dataset.load_data(DATA_PATH, config)
    logging.info("Replacing missing data ...")
    data = dataset.fillnan_dataset(data)
    logging.info("Preprocessing ...")
    x_train, y_train, x_test, y_test = dataset.dummy_preprocessing(data)
    # Load the model
    hyperparams = hyper_to_dict(hyperparams_str)
    model = models.get_model(model=model_str, hyperparams=hyperparams)
    logging.info("Training phase ...")
    # Training and evaluation
    _, score = train_eval.train_eval_once(
        model=model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
    )
    logging.info("Exporting to csv ...")
    train_eval.export_results_to_csv(model_name=model_str, hyperparameters=hyperparams_str, score=score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Airbnb price prediction program")
    parser.add_argument(
        "model",
        help="Model to train (available: random forest, adaboost, gradient boosting, extreme random forest)",
        type=str,
    )
    parser.add_argument("--hyper", help="Hyperparameters", type=str, default="")
    args = parser.parse_args()
    main(model_str=args.model, hyperparams_str=args.hyper)
