
import yaml
import pickle
import re

from fastai.imports import *
from fastai.torch_imports import *
from fastai.dataset import *
from fastai.learner import *
from fastai.column_data import *
from attrdict import AttrDict
from scoring import Scoring
from feature_engineering import FeatureEngineering

from data_cleaning import *



from sklearn.ensemble import RandomForestRegressor
from utils import Config

# # Load in our data from last lesson


dep = 'SalePrice'
PATH = "data/"
#df_raw = pd.read_feather('tmp/bulldozers-raw')
#keep_cols = list(np.load('tmp/keep_cols.npy'))

def read_yaml(filepath: string) -> AttrDict:
    with open(filepath) as f:
        config = yaml.load(f)
    return AttrDict(config)

"""
Initial setting up of the data frame before putting in the feather file. 
Includes 
1. reading directly from the .csv file which can be time consuming
2. Convering saleprice to its log
3. Turning columns with strings to categorical variables 
4. Setting the categorical variables in order
5. Converting dates to features
"""
def setUpDataFrame():
    train_filepath = read_yaml('baseConfig.yaml')
    df_raw = pd.read_csv(train_filepath, low_memory=False, parse_dates=['saledate'] )
    print('The shape of dataframe is %s' %(str(df_raw.shape)))
    cleaning = Cleaning()
    print('Converting sale price to log of sale price')
    df_raw = cleaning.convertFeatureToItsLog(df_raw, 'SalePrice')
    
    print("Turning string to categorical variables")
    df_raw = cleaning.turnStringToCategorical(df_raw)
    #Aligning the levels properly
    df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)

    #converting date and time to features
    feat_eng = FeatureEngineering()
    feat_eng.convertDatesToFeatures(df_raw, 'saledate')
    #saving as feather
    try:
        os.makedirs('tmp', exist_ok=True)
        df_raw.to_feather('tmp/raw')
    except (FileNotFoundError, IOError) as e:
        print(e)

feat_eng = FeatureEngineering()
print(feat_eng.testIfDateTimeWorks());


base_config = read_yaml('baseConfig.yaml')

#Reading files
try:
    df_raw = pd.read_feather(base_config.parameters.bulldozer_train_feather)
    print('Finished reading feather file')
    if 'saleYear' in df_raw.columns:
        print('Features from dates are present in this feather file')
except (IOError, OSError) as e:
    print('Feather file does not exist')
    print(e)
    print('Doing the initial setup')
    setUpDataFrame()


print('Reading done')
print('The shape of dataframe is %s' %(str(df_raw.shape)))


#replace missing values with median, separate the dependent variable 
df, y, _ = proc_df(df_raw, 'SalePrice')

#splitting to training and validation sets
cleaning = Cleaning()

n_valid = 12000
n_training = len(df) - n_valid

print("splitting raw")
raw_train, raw_valid = cleaning.split_vals(df_raw, n_training)

print("Splitting X")
X_train, X_valid = cleaning.split_vals(df, n_training)

print("Splitting y")
y_train, y_valid = cleaning.split_vals(y, n_training)

print("Shape of X_train is %s, shape of X_valid is %s, shape of y_train is %s and of y_valid is %s"
            %(str(X_train.shape), str(X_valid.shape), str(y_train.shape), str(y_valid.shape)))


scoring = Scoring()

def print_score(m):
    res = [scoring.rmse(m.predict(X_train), y_train), scoring.rmse(m.predict(X_valid), y_valid),
            m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)

m = RandomForestRegressor(n_jobs=1)

print('Fitting the model')
m.fit(X_train, y_train)

print_score(m)

