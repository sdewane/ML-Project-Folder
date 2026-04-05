# Imprt necessary libraries for data manipulation, visualization, and machine learning.
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
df = pd.read_csv(CSV_PATH)
print("Data loaded successfully.")
#display basic info about the dataset
def data_info(df):
    print("Dataset shape:", df.shape)
    print(df.head())
    print("Column data: ")
    print(df.columns)
    print(df.info())

#data_info(df)
#^^^^^^^^^^^^^^^ uncomment to see basic data info 
#1. longitude: A measure of how far west a house is; a higher value is farther west
#2. latitude: A measure of how far north a house is; a higher value is farther north
#3. housingMedianAge: Median age of a house within a block; a lower number is a newer building
#4. totalRooms: Total number of rooms within a block
#5. totalBedrooms: Total number of bedrooms within a block
#6. population: Total number of people residing within a block
#7. households: Total number of households, a group of people residing within a home unit, for a block
#8. medianIncome: Median income for households within a block of houses (measured in tens of thousands of US Dollars)
#9. medianHouseValue: Median house value for households within a block (measured in US Dollars)
#10. oceanProximity: Location of the house w.r.t ocean/sea

#list of colums and their datatype
num_cols = df.select_dtypes(include=[np.number]).columns.to_list()
categorical_cols = df.select_dtypes(include=["str"]).columns.to_list()
def missing_data(df):
    print("\nMissing values per column: ")
    print(df.isna().sum())

#missing_data(df)
#^^^^^^^^^^^^^^^^^^uncomment to print out how much data is missing per columns

def find_encoded_data(df):
    for col in df.columns:
        print(df[col].value_counts().head(20))

#find_encoded_data(data)
#^^^^^^^^^^^^^^^^^^^^^^^uncomment to see if there are any encoded missing value entries per column 

def find_duplicates(df):
    duplicate_mask = df.duplicated()
    num_duplicates = duplicate_mask.sum()
    print("Num of duplicated rows: ", num_duplicates)
    #optional
    #df = df.drop_duplicates()
    #print("Shape after dropping duplicates: ", df.shape)

#find_duplicates(df)
#^^^^^^^^^^^^^^^^^^^^^uncomment to find ducplicates in data and optionaly clean up duplicates

#print(df[num_cols].describe().T)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^uncomment to see data statistics transposed

def countplot_cattegorical_columns(categorical_cols):
    for col in categorical_cols:
        plt.figure(figsize=(10,3))
        sns.countplot(x=col, data=df)
        plt.title(f"Distribution of {col}")
        print(df[col].value_counts())
        plt.show()
    
#countplot_cattegorical_columns(categorical_cols)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^uncomment to see plot of categorical data and exact numbers

def target_col_distribution(target,df):
        plt.figure(figsize=(6,4))
        sns.histplot(df[target], bins=40, kde=True)
        plt.title(f"Target distribution: "+ target)
        plt.xlabel(""+ target)
        print(df[TARGET_COL].value_counts())
        plt.show()
        
#target_col_distribution(TARGET_COL,df)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^uncomment to see graph and exact numbers

def all_num_column_distribution(num_cols,df):
    fig, axes = plt.subplots(3, 3,figsize=(8, 6))
    axes= axes.flatten()
    for i, col in enumerate(num_cols):
        sns.histplot(df[col],kde=True, ax=axes[i])
        axes[i].set_title(col, fontsize=8)
    plt.tight_layout()
    plt.show()
#all_num_column_distribution(num_cols,df)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^uncomment to see graph 
def boxplot_visial(num_cols, df):
    fig, axes = plt.subplots(3, 3,figsize=(8, 6))
    axes= axes.flatten()
    for i, col in enumerate(num_cols):
        sns.boxplot(x=df[col], ax= axes[i])
        axes[i].set_title(col, fontsize=8)
        axes[i].set_xlabel("")
    plt.tight_layout()
    plt.show()
#boxplot_visial(num_cols,df)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^uncomment to see box plot to detect outliers

def id_corrolated_col_features(num_cols,df):
    plt.figure(figsize=(10,5))
    sns.heatmap(
        df[num_cols].corr(),
        annot=True,
        cmap="coolwarm",
        center=0
    )

    plt.title("Correlation Heatmap")
    plt.show()
#id_corrolated_col_features(num_cols,df)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^Uncomment to see heat graph,high correlation means using a linear model, other models would require highly correlated columns to be dropped

def corr_wth_trg_col (target, num_cols, df):
    corr_wth_taget= df[num_cols].corr()[target].sort_values(ascending=False)
    print("\ncorrelation with target: ")
    print(corr_wth_taget)
#corr_wth_trg_col(TARGET_COL, num_cols,df)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^uncomment to see correlation of numerical datapoints and target value