import pandas as pd
from sklearn.model_selection import train_test_split

from airbnb_prices.config import ExperimentConfig
from airbnb_prices.data.fillna import infere_nan
from airbnb_prices.data.preprocessing import apply_preprocessing

from .dataset import load_data
from .feature_engineering import apply_feature_engineering


class DataPipeline:
    def __init__(self, data: pd.DataFrame, config: ExperimentConfig):
        self.config = config
        self.df_train, tmp = train_test_split(data, test_size=0.3, random_state=42)  # 70% train
        self.df_val, self.df_test = train_test_split(
            tmp, test_size=0.5, random_state=42
        )  # 15% val and 15% test
        self.scalers = None

    @classmethod
    def from_file(cls, data_path, config_path):
        df = load_data(data_path)
        config = ExperimentConfig.parse_file(config_path)
        return cls(df, config)

    def run(self):
        self.infere_nan()
        self.apply_feature_engineering()
        self.apply_preprocessing()

    def infere_nan(self):
        self.df_train = infere_nan(self.df_train)
        self.df_val = infere_nan(self.df_val)
        self.df_test = infere_nan(self.df_test)

    def apply_feature_engineering(self):
        self.df_train = apply_feature_engineering(self.df_train, self.config.feature_engineering)
        self.df_val = apply_feature_engineering(self.df_val, self.config.feature_engineering)
        self.df_test = apply_feature_engineering(self.df_test, self.config.feature_engineering)

    def apply_preprocessing(self):
        self.scalers, self.df_train = apply_preprocessing(self.df_train, self.config.preprocessing)
        _, self.df_val = apply_preprocessing(self.df_val, self.config.preprocessing, self.scalers)
        _, self.df_test = apply_preprocessing(self.df_test, self.config.preprocessing, self.scalers)

    def __call__(self, df):
        df = infere_nan(df)
        df = apply_feature_engineering(df, self.config.feature_engineering)
        _, df = apply_preprocessing(df, self.config.preprocessing, self.scalers)
        return df

    @property
    def train_data(self):
        return self.df_train[self.df_train.columns.difference(["Price"])], self.df_train["Price"]

    @property
    def val_data(self):
        return self.df_val[self.df_val.columns.difference(["Price"])], self.df_val["Price"]

    @property
    def test_data(self):
        return self.df_test[self.df_test.columns.difference(["Price"])], self.df_test["Price"]
