from fastai.imports import *
from fastai.torch_imports import *
from fastai.dataset import *
from fastai.learner import *
from fastai.structured import *
from fastai.column_data import *
from attrdict import AttrDict
from scoring import *
from feature_engineering import *

from data_cleaning import *
import yaml
import pickle
import re


def setUpDataFrame():
    cleaning = Cleaning()
    train_path = cleaning.read_yaml('baseConfig.yaml').parameters.houseprice_train_filepath
    train_raw = pd.read_csv(train_path, low_memory=False)
    train_raw = cleaning.turnStringToCategorical(train_raw)

    try:
        os.makedirs('tmp', exist_ok=True)
        train_raw.to_feather('tmp/houseprice_raw')
    except (FileNotFoundError, IOError) as e:
        print(e)

setUpDataFrame()
    



