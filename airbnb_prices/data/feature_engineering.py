import json
from typing import Dict, List, Union

import pandas as pd
from numpy import sqrt

from airbnb_prices.data.dataset import load_config, load_data


class FeatureEngineeringConfiguration:
    """
    Feature Engineering Configuration:

    Has three attributes :
    self.distance_to_monuments, a list of dicts of the following form :
        {
            "name": Name of the monument,
            "coordinates": [
                Latitude,
                Longitude
            ]
        }

    self.dates_delta, a list of dicts of the following form :
        {
            "left_column": Name of a column to which the right column will be substraced,
            "right_column": Name of the right column
        }

    self.date_to_duration, a list of dicts of the following form :
        {
            "column": Name of a column containing dates,
            "max_or_min": "max" or "min"
        }
    """

    def __init__(self, feature_engineering: Dict[str, List]):
        self.dates_delta = feature_engineering["dates_delta"]
        self.date_to_duration = feature_engineering["date_to_duration"]
        self.distance_to_monuments: List[Dict[str, Union[str, List[float]]]] = feature_engineering[
            "distance_to_monuments"
        ]


def load_feature_engineering_config(path: str) -> FeatureEngineeringConfiguration:
    """Loads the feature engineering config from a json file

    Args:
        path (str): JSON File

    Returns:
        DatasetConfiguration
    """
    with open(path, "r") as file:
        file_content = json.load(file)
        return FeatureEngineeringConfiguration(feature_engineering=file_content["feature_engineering"])


def add_distance_to_monuments(
    dataset: pd.DataFrame, feature_engineering_config: FeatureEngineeringConfiguration
) -> None:
    """For each monument listed in config, add a new column containing the distance between
    the AirBnB and the monument.
    The name of the new column is "Distance to [monument name]".

    Args:
        dataset (pd.DataFrame): the dataset onto which the columns will be added
        feature_engineering_config (FeatureEngineeringConfiguration): the config, obtained
        with load_feature_engineering_config
    """
    for monument_dict in feature_engineering_config.distance_to_monuments:
        dataset[f"Distance to {monument_dict['name']}"] = sqrt(
            (dataset["Latitude"] - monument_dict["coordinates"][0]) ** 2
            + (dataset["Longitude"] - monument_dict["coordinates"][1]) ** 2
        )
    return dataset


def add_date_to_duration(
    dataset: pd.DataFrame, feature_engineering_config: FeatureEngineeringConfiguration
) -> None:
    """For each column listed in config, add a new column containing for each instance
    the duration between its date and the maximum or minimum date of the column.
    The name of the new column is "[column name] Duration from [max | min]".

    The goal is to always obtain a positive duration, thus we compute :
    max - current if the parameter is max
    current - min if the parameter is min

    Args:
        dataset (pd.DataFrame): the dataset onto which the columns will be added
        feature_engineering_config (FeatureEngineeringConfiguration): the config, obtained
        with load_feature_engineering_config
    """
    for date_to_duration_config in feature_engineering_config.date_to_duration:
        column_name = date_to_duration_config["column"]
        if date_to_duration_config["max_or_min"] == "max":
            dataset[f"{column_name} Duration from max"] = (
                max(dataset[column_name]) - dataset[column_name]
            ).apply(lambda x: x.days)
        elif date_to_duration_config["max_or_min"] == "min":
            dataset[f"{column_name} Duration from min"] = (
                dataset[column_name] - min(dataset[column_name])
            ).apply(lambda x: x.days)

        else:
            raise ValueError(
                'The "max_or_min" parameter in the config of "date_to_duration" should either be "max" or "min".'
            )


def add_dates_delta(
    dataset: pd.DataFrame, feature_engineering_config: FeatureEngineeringConfiguration
) -> None:
    """For each duo of columns listed in config, add a new column containing the difference
    between these columns : left_column - right_column.
    The name of the new column is "Delta between [left_column] and [right_column]".

    Args:
        dataset (pd.DataFrame): the dataset onto which the columns will be added
        feature_engineering_config (FeatureEngineeringConfiguration): the config, obtained
        with load_feature_engineering_config
    """
    for dates_delta_config in feature_engineering_config.dates_delta:
        left_column = dates_delta_config["left_column"]
        right_column = dates_delta_config["right_column"]
        dataset[f"Delta between {left_column} and {right_column}"] = (
            dataset[left_column] - dataset[right_column]
        ).apply(lambda x: x.days)


def apply_feature_engineering(dataset: pd.DataFrame, config_path: str) -> None:
    """Applies successively all the existing feature engineering functions, i.e. :
    - Adds the distance to monuments
    - Adds deltas between date columns
    - Adds duration from date column

    Args:
        dataset (pd.DataFrame): the dataset onto which the columns will be added
        config_path (str): JSON file
    """
    feature_engineering_config = load_feature_engineering_config(config_path)
    add_distance_to_monuments(dataset, feature_engineering_config)
    add_dates_delta(dataset, feature_engineering_config)
    add_date_to_duration(dataset, feature_engineering_config)


if __name__ == "__main__":

    # Example of use :
    dataset_config = load_config("./examples/dataset/config.json")
    dataset = load_data("./data/train_airbnb_berlin.csv", dataset_config)
    apply_feature_engineering(dataset, "./examples/dataset/config.json")

    print(
        dataset[
            [
                "Distance to Berlin Cathedral",
                "Delta between Last Review and First Review",
                "Host Since Duration from max",
            ]
        ].head()
    )
