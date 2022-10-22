import numpy as np
import pandas as pd
from pandas import DataFrame


def load_data(path: str) -> DataFrame:
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

    parse_dates = {"Host Since", "First Review", "Last Review"}

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
        parse_dates=list(parse_dates.intersection(dtype)),
        converters={"Host Response Rate": percentage_parser},
        dayfirst=True,
    )

    for col in df:
        df[col] = df[col].astype(dtype[col])
    return df
