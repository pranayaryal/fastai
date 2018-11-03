
from fastai.imports import *
from fastai.torch_imports import *
from fastai.dataset import *
from fastai.learner import *
from fastai.structured import *
from fastai.column_data import *
from attrdict import AttrDict

from data_cleaning import *

class Scoring:
    """
    This will have scoring related methods
    """

    def rmse(self, x:pd.Series, y:pd.Series):
        return math.sqrt(((x-y)**2).mean())
    
