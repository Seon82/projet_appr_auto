import pandas as pd
from numpy import sqrt

from airbnb_prices.config import FeatureEngineeringConfig


def add_distance_to_monuments(
    dataset: pd.DataFrame, feature_engineering_config: FeatureEngineeringConfig
) -> None:
    """For each monument listed in config, add a new column containing the distance between
    the AirBnB and the monument.
    The name of the new column is "Distance to [monument name]".

    Args:
        dataset (pd.DataFrame): the dataset onto which the columns will be added
        feature_engineering_config (FeatureEngineeringConfiguration): the config, obtained
        with load_feature_engineering_config
    """
    for monument_config in feature_engineering_config.distance_to_monuments:
        dataset[f"Distance to {monument_config.name}"] = sqrt(
            (dataset["Latitude"] - monument_config.coordinates[0]) ** 2
            + (dataset["Longitude"] - monument_config.coordinates[1]) ** 2
        )
    return dataset


def add_date_to_duration(
    dataset: pd.DataFrame, feature_engineering_config: FeatureEngineeringConfig
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
        column_name = date_to_duration_config.column
        if date_to_duration_config.max_or_min == "max":
            dataset[f"{column_name} Duration from max"] = (
                max(dataset[column_name]) - dataset[column_name]
            ).apply(lambda x: x.days)
        elif date_to_duration_config.max_or_min == "min":
            dataset[f"{column_name} Duration from min"] = (
                dataset[column_name] - min(dataset[column_name])
            ).apply(lambda x: x.days)


def add_dates_delta(
    dataset: pd.DataFrame, feature_engineering_config: FeatureEngineeringConfig
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
        left_column = dates_delta_config.left_column
        right_column = dates_delta_config.right_column
        dataset[f"Delta between {left_column} and {right_column}"] = (
            dataset[left_column] - dataset[right_column]
        ).apply(lambda x: x.days)


def apply_feature_engineering(dataset: pd.DataFrame, config: FeatureEngineeringConfig) -> pd.DataFrame:
    """Applies successively all the existing feature engineering functions, i.e. :
    - Adds the distance to monuments
    - Adds deltas between date columns
    - Adds duration from date column

    Args:
        dataset (pd.DataFrame): the dataset onto which the columns will be added
        config (str): Configuration
    """
    add_distance_to_monuments(dataset, config)
    add_dates_delta(dataset, config)
    add_date_to_duration(dataset, config)
    return dataset
