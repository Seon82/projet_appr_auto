from typing import Dict, List

import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, path: str):
        """Initialize the dataset.

        Args:
            path (str): path to csv file to load
        """
        self._columns = {
            "category": [
                "Host Response Time",
                "Is Superhost",
                "Neighborhood Group",
                "Is Exact Location",
                "Property Type",
                "Room Type",
                "Instant Bookable",
            ],
            "numerical": [
                "Host Response Rate",
                "Latitude",
                "Longitude",
                "Accomodates",
                "Bathrooms",
                "Bedrooms",
                "Beds",
                "Square Feet",
                "Guests Included",
                "Reviews",
                "Overall Rating",
                "Accuracy Rating",
                "Cleanliness Rating",
                "Checkin Rating",
                "Communication Rating",
                "Location Rating",
                "Value Rating",
                "Price",
            ],
            "date": ["Host Since", "First Review", "Last Review"],
        }

        _all_columns = []
        for value in self.columns.values():
            _all_columns.extend(value)

        _dtypes = {}
        for col in self.columns["category"]:
            _dtypes[col] = "category"
        for col in self.columns["numerical"]:
            _dtypes[col] = "float64"

        self._data = pd.read_csv(
            path,
            usecols=_all_columns,
            na_values=["*"],
            true_values=["t"],
            false_values=["f"],
            dtype=_dtypes,
            parse_dates=self.columns["date"],
            dayfirst=True,
            converters={"Host Response Rate": self._convert_percentage},
        )

    def _convert_percentage(self, percentage: str):
        if percentage in {"", "*"}:
            return np.NaN
        return float(percentage.rstrip("%"))

    @property
    def data(self) -> pd.DataFrame:
        """The data as a Pandas Dataframe."""
        return self._data

    @property
    def columns(self) -> Dict[str, List[str]]:
        """The columns by type: "category", "numerical" or "date" """
        return self._columns
