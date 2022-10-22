from typing import Literal

from pydantic import BaseModel

from .utils import ColumnName


class DistanceToMonumentsConfig(BaseModel):
    name: str
    coordinates: list[float]


class DatesDeltaConfig(BaseModel):
    left_column: ColumnName
    right_column: ColumnName


class DatesToDurationConfig(BaseModel):
    column: ColumnName
    max_or_min: Literal["max", "min"]


class FeatureEngineeringConfig(BaseModel):
    distance_to_monuments: list[DistanceToMonumentsConfig]
    dates_delta: list[DatesDeltaConfig]
    date_to_duration: list[DatesToDurationConfig]
