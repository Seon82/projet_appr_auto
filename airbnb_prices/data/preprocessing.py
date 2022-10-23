import pandas as pd
from sklearn.preprocessing import StandardScaler

from airbnb_prices.config import PreprocessingConfig


def apply_preprocessing(
    df: pd.DataFrame, config: PreprocessingConfig, scalers=None | dict[str, StandardScaler]
):
    for col in config.ohe:
        df = pd.concat((df, pd.get_dummies(df[col], prefix=col)), axis=1)
    if scalers is None:
        scalers = {}
    for col in config.standardize:
        if col in scalers:
            df[col] = scalers[col].transform(df[col])
        else:
            scalers[col] = StandardScaler()
            df[col] = scalers[col].fit_transform(df[col])
    df = df.drop(config.drop, axis=1)
    return scalers, df
