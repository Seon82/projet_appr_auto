import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from airbnb_prices.config import PreprocessingConfig


def apply_preprocessing(
    df: pd.DataFrame, config: PreprocessingConfig, scalers: None | dict[str, TransformerMixin] = None
):
    for col in config.ohe:
        df = pd.concat((df, pd.get_dummies(df[col], prefix=col)), axis=1)
    if scalers is None:
        scalers = {}
    for col in config.standardize:
        df, scalers = apply_scaling(df, col, scalers, StandardScaler)
    for col in config.min_max:
        df, scalers = apply_scaling(df, col, scalers, MinMaxScaler)

    if config.drop:
        df = df.drop(config.drop, axis=1)
    elif config.keep:
        df = df.drop(df.columns.difference(config.keep), axis=1)
    return scalers, df


def apply_scaling(
    df: pd.DataFrame, col: str, scalers: dict[str, TransformerMixin], scaler_type: TransformerMixin
):
    if col in scalers:
        df[col] = scalers[col].transform(df[col].values.reshape(-1, 1))
    else:
        scalers[col] = scaler_type()
        df[col] = scalers[col].fit_transform(df[col].values.reshape(-1, 1))
    return df, scalers
