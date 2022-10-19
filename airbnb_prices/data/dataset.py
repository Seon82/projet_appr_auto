import json
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from pandas import DataFrame


@dataclass
class DatasetConfiguration:
    """
    Dataset Configuration:

            drop (List[str]): The columns to drop
    """

    drop: List[str]


def load_config(path: str) -> DatasetConfiguration:
    """Loads the dataset config from a json file

    Args:
        path (str): JSON File

    Returns:
        DatasetConfiguration
    """
    with open(path, "r") as file:
        file_content = json.load(file)
        return DatasetConfiguration(drop=file_content["drop"])


def load_data(path: str, config: DatasetConfiguration) -> DataFrame:
    """Loads the dataset as a pandas DataFrame

    Args:
        path (str): Path to the dataset CSV file
        config (DatasetConfiguration): Configuration for the loading step

    Returns:
        pd.DataFrame: Dataframe of the dataset
    """
    dtype = {
        "Host Since": "datetime64[ns]",
        "Host Response Time": "category",
        "Host Response Rate": float,
        "Is Superhost": float,
        "neighbourhood": "category",
        "Neighborhood Group": "category",
        "Postal Code": float,
        "Latitude": float,
        "Longitude": float,
        "Is Exact Location": float,
        "Property Type": "category",
        "Room Type": "category",
        "Accomodates": float,
        "Bathrooms": float,
        "Bedrooms": float,
        "Beds": float,
        "Square Feet": float,
        "Guests Included": float,
        "Min Nights": float,
        "Reviews": float,
        "First Review": "datetime64[ns]",
        "Last Review": "datetime64[ns]",
        "Overall Rating": float,
        "Accuracy Rating": float,
        "Cleanliness Rating": float,
        "Checkin Rating": float,
        "Communication Rating": float,
        "Location Rating": float,
        "Value Rating": float,
        "Instant Bookable": float,
        "Price": float,
    }

    for col in config.drop:
        dtype.pop(col)

    na_values = ["*", "", " "]

    def percentage_parser(x):
        if x in na_values:
            return np.NaN
        return x.rstrip("%")

    df = pd.read_csv(
        path,
        usecols=dtype.keys(),
        na_values=na_values,
        true_values=["t"],
        false_values=["f"],
        parse_dates=["Host Since", "First Review", "Last Review"],
        converters={"Host Response Rate": percentage_parser},
        dayfirst=True,
    )

    for col in df:
        df[col] = df[col].astype(dtype[col])
    return df


def fillnan_dataset(data: DataFrame) -> DataFrame:
    """Replace NaNs in the dataset.

    Args:
        data: dataset already loaded

    Returns:
        data: dataset cleaned
    """
    data = clean_ratings_columns(data)
    data = clean_host_response_time(data)
    data = clean_reviews(data)
    # Clean host response rate
    data["Host Response Rate"] = data["Host Response Rate"].fillna(
        value=data["Host Response Rate"].median()
    )
    data = data.dropna()
    return data


def clean_ratings_columns(data: DataFrame) -> DataFrame:
    """Replace NaNs in the ratings columns.

    Estimate the median value for each column
    and replace NaNs by this value.

    Args:
        data: dataset loaded

    Returns:
        data: dataset cleaned
    """
    ratings_cols = [
        "Overall Rating",
        "Accuracy Rating",
        "Cleanliness Rating",
        "Checkin Rating",
        "Communication Rating",
        "Location Rating",
        "Value Rating",
    ]
    rating_med_df = data[ratings_cols].median()
    data = data.fillna(value=rating_med_df)
    return data


def clean_host_response_time(data: pd.DataFrame) -> DataFrame:
    """Replace NaNs in the host response time column.

    Estimate the probabilistic distribution of the column.
    Replace NaNs by random values drawn with this distribution.

    Args:
        data: dataset loaded

    Returns:
        dataset cleaned
    """
    # Set a seed for reproducibility
    np.random.seed(42)
    nb_values = np.sum(data["Host Response Time"].isna())
    host_resp_time = data["Host Response Time"].dropna()
    time_values = ["within a few hours", "within an hour", "within a day", "a few days or more"]
    host_resp_time = host_resp_time.replace(to_replace=time_values, value=[1, 2, 3, 4])
    proba = np.bincount(host_resp_time)
    proba = proba / np.sum(proba)
    proba = proba[proba > 0]
    rand_values = np.random.choice(a=time_values, p=proba, size=nb_values)
    data["Host Response Time"] = data["Host Response Time"].fillna(value=pd.Series(data=rand_values))
    return data


def clean_reviews(data: DataFrame) -> DataFrame:
    """Replace missing reviews by the median review date."""
    reviews_cols = ["First Review", "Last Review"]
    med_reviews_date = data[reviews_cols].median()
    data = data.fillna(value=med_reviews_date)
    return data
