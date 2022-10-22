from pydantic import BaseModel

from .feature_engineering_config import FeatureEngineeringConfig
from .preprocessing_config import PreprocessingConfig


class ExperimentConfig(BaseModel):
    feature_engineering: FeatureEngineeringConfig
    preprocessing: PreprocessingConfig
