import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame


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
        "Accuracy Rating" "Cleanliness Rating",
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
    proba, _, _ = plt.hist(data["Host Response Time"].dropna())
    # Normalize probabilities
    proba = proba / np.sum(proba)
    proba = proba[proba > 0]
    time_values = ["within a few hours", "within an hour", "within a day", "a few days or more"]
    rand_values = np.random.choice(a=time_values, p=proba, size=nb_values)
    data["Host Response Time"] = data["Host Response Time"].fillna(value=pd.Series(data=rand_values))
    return data


def clean_reviews(data: DataFrame) -> DataFrame:
    """Replace missing reviews by the median review date."""
    reviews_cols = ["First Review", "Last Review"]
    med_reviews_date = data[reviews_cols].median()
    data = data.fillna(value=med_reviews_date)
    return data


def remove_nan(data: DataFrame):
    """Remove columns with too many NaNs."""
    to_drop = ["Square Feet", "Postal Code"]
    data = data.drop(to_drop, axis=1)
    return data
