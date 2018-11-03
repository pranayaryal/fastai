import numpy as np
import pandas as pd
from fastai.imports import *
from fastai.torch_imports import *
from fastai.dataset import *
from fastai.learner import *
from fastai.structured import train_cats
from fastai.column_data import *
from attrdict import AttrDict
import yaml

class Cleaning:
    """
    Initial cleaning of dataframe
    """

    def convertFeatureToItsLog(self, df: pd.DataFrame, feature_name: string) -> pd.DataFrame:
        df[feature_name] = np.log(df[feature_name])
        return df
    
    def turnStringToCategorical(self, df: pd.DataFrame) -> pd.DataFrame:
        return train_cats(df)

    #split to training and validation sets
    def split_vals(self, a: pd.DataFrame, n: int) -> pd.DataFrame: 
        return a[:n].copy(), a[n:].copy()

    
    def displayMissingValues(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.isnull().sum().sort_index()/len(df)

    def read_yaml(self, filepath: string) -> AttrDict:
        with open(filepath) as f:
            config = yaml.load(f)
        return AttrDict(config)