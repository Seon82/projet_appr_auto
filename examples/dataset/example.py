import pandas as pd

from airbnb_prices.data.dataset import DatasetConfiguration, load_config, load_data

config: DatasetConfiguration = load_config("examples/dataset/config.json")
df: pd.DataFrame = load_data("examples/dataset/sample.csv", config)

print(df.head())
