import json
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


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


def load_data(path: str, config: DatasetConfiguration) -> pd.DataFrame:
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
