import numpy as np
import pandas as pd


def infere_nan(data: pd.DataFrame) -> pd.DataFrame:
    """Replace NaNs in select columns of the dataset.
    The filled columns are the rating columns, Host Response Time/Rate
    and First/Last Review.

    Args:
        data: dataset already loaded

    Returns:
        data: dataset cleaned
    """
    data = clean_ratings_columns(data)
    data = clean_host_response_time(data)
    data = clean_reviews(data)
    # Clean host response rate
    if "Host Response Rate" in data.columns:
        data["Host Response Rate"] = data["Host Response Rate"].fillna(
            value=data["Host Response Rate"].median()
        )
    return data


def clean_ratings_columns(data: pd.DataFrame) -> pd.DataFrame:
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


def clean_host_response_time(data: pd.DataFrame) -> pd.DataFrame:
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
    # Normalize probabilities
    proba = proba / np.sum(proba)
    proba = proba[proba > 0]
    rand_values = np.random.choice(a=time_values, p=proba, size=nb_values)
    data["Host Response Time"][data["Host Response Time"].isna()] = rand_values
    return data


def clean_reviews(data: pd.DataFrame) -> pd.DataFrame:
    """Replace missing reviews by the median review date."""
    reviews_cols = ["First Review", "Last Review"]
    med_reviews_date = data[reviews_cols].median(numeric_only=False)
    data = data.fillna(value=med_reviews_date)
    return data
