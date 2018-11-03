import pandas as pd
import numpy as np

from fastai.imports import *
from fastai.torch_imports import *
from fastai.dataset import *
from fastai.learner import *
from fastai.structured import *
from fastai.column_data import *
from attrdict import AttrDict
import yaml

class FeatureEngineering:
    """
    Uses fastai method 'add_datepart' from structured.py to convert dates to features.
    """
    
    def convertDatesToFeatures(self, df: pd.DataFrame, column_name: string)-> pd.DataFrame:
        add_datepart(df, column_name)

    def testIfDateTimeWorks(self) -> bool:
        df = pd.DataFrame({'A': pd.to_datetime(['3/11/2000', '3/12/2000', '3/13/2000'], infer_datetime_format=False)})
        self.convertDatesToFeatures(df, 'A')
        if 'AYear' in df.columns.values:
            return True
        return False
