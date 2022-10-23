from pydantic import BaseModel, validator

from .utils import ColumnName


class PreprocessingConfig(BaseModel):
    ohe: list[ColumnName]
    standardize: list[ColumnName]
    min_max: list[ColumnName]
    drop: list[ColumnName] = []
    keep: list[ColumnName] = []

    @validator("keep")
    def keep_xor_drop(cls, keep, values):
        if keep:
            if values["drop"]:
                raise ValueError("Config error: please use either drop or keep in the config")
            return keep + ["Price"]
        return keep

    @validator("min_max")
    def min_max_xor_standardize(cls, min_max, values):
        common = set(values["standardize"]) & set(min_max)
        if common:
            raise ValueError(f"Config error: cannot apply multiple scalers to these columns: {common}")
        return min_max
