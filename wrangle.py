import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def wrangle_data():
    """ 
    This function get the data from the csv file, adds columns and returns a DataFrame.
    """
    
    df = pd.read_csv('predictive_maintenance.csv')
    
    df['Temp Delta [K]'] = df['Process temperature [K]'] - df['Air temperature [K]']
    cols_to_drop = 'Failure Type'
    df = df.drop(columns = cols_to_drop)
    
    return df


# 60% train, 20% validate, 20% test.
def train_validate(df, stratify_col = None, random_seed=1969):
    """
    This function takes in a DataFrame and column name for the stratify argument (defualt is None).
    It will split the data into three parts for training, testing and validating.
    """
    #This is logic to set the stratify argument:
    stratify_arg = ''
    if stratify_col != None:
        stratify_arg = df[stratify_col]
    else:
        stratify_arg = None
    
    #This splits the DataFrame into 'train' and 'test':
    train, test = train_test_split(df, train_size=.8, stratify=stratify_arg, random_state = random_seed)
    
    #The length of the stratify column changed and needs to be adjusted:
    if stratify_col != None:
        stratify_arg = train[stratify_col]
        
    #This splits the larger 'train' DataFrame into a smaller 'train' and 'validate' DataFrames:
    train, validate = train_test_split(train, train_size=.6, stratify=stratify_arg, random_state = random_seed)
    return train, validate, test