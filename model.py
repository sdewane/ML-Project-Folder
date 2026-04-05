#Imprt necessary libraries for data manipulation, visualization, and machine learning.
# %pip install -q numpy pandas seaborn matplotlib scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

print("All libraries imported successfully.")   

#config
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: f"{x:.3f}")
sns.set_theme(style="darkgrid")

plt.rcParams.update({
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
})

RANDOM_STATE = 42
CSV_PATH = "data/housing.csv" #change depending on dataset
TARGET_COL= "median_house_value" #target variable for prediction

#load data
data = pd.read_csv(CSV_PATH)
print("Data loaded successfully.")
#display basic info about the dataset
print("Dataset shape:", data.shape)
print("Dataset columns:", data.columns.tolist())

data.head()