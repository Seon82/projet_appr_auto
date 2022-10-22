"""Run the airbnb prices detection system."""
import argparse
import logging
from pathlib import Path

from airbnb_prices.data import DataPipeline
from airbnb_prices.eval import train_eval
from airbnb_prices.models import models

CONFIG_PATH = Path("./examples/config.json")
DATA_PATH = Path("./data/train_airbnb_berlin.csv")


def hyper_to_dict(hyper: str):
    """Convert string of hyperparameters to a dictionary."""
    if not hyper:
        return {}
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
    pipeline = DataPipeline.from_file(DATA_PATH, CONFIG_PATH)
    logging.info("Replacing missing data ...")
    pipeline.infere_nan()
    logging.info("Feature engineering ...")
    pipeline.apply_feature_engineering()
    logging.info("Preprocessing ...")
    pipeline.apply_preprocessing()
    # Load the model
    hyperparams = hyper_to_dict(hyperparams_str)
    model = models.get_model(model=model_str, hyperparams=hyperparams)
    logging.info("Training phase ...")
    # Training and evaluation
    X_train, y_train = pipeline.train_data
    X_val, y_val = pipeline.val_data
    _, score = train_eval.train_eval_once(
        model=model, x_train=X_train, y_train=y_train, x_test=X_val, y_test=y_val
    )
    logging.info("Exporting to csv ...")
    train_eval.export_results_to_csv(model_name=model_str, hyperparameters=hyperparams_str, score=score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Airbnb price prediction program")
    parser.add_argument(
        "model",
        help="Model to train (available models: see doc)",
        type=str,
    )
    parser.add_argument("--hyper", help="Hyperparameters", type=str, default="")
    args = parser.parse_args()
    main(model_str=args.model, hyperparams_str=args.hyper)
