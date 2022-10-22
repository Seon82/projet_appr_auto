from pydantic import BaseModel

from .utils import ColumnName


class PreprocessingConfig(BaseModel):
    ohe: list[ColumnName]
    drop: list[ColumnName]
    standardize: list[ColumnName]
