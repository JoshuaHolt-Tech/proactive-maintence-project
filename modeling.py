#Tools to build machine learning models and reports
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler


import pandas as pd
import numpy as np


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


def train_val_test(train, val, test, target_col):
    """
    Seperates out the target variable and creates a series with only the target variable to test accuracy.
    """
    #Seperating out the target variable
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]

    X_val = val.drop(columns = [target_col])
    y_val = val[target_col]

    X_test = test.drop(columns = [target_col])
    y_test = test[target_col]
    return X_train, y_train, X_val, y_val, X_test, y_test


def scale_cont_columns(train, val, test, cont_columns, scaler_model = 1):
    """
    This takes in the train, validate and test DataFrames, scales the cont_columns using the
    selected scaler and returns the DataFrames.
    *** Inputs ***
    train: DataFrame
    validate: DataFrame
    test: DataFrame
    scaler_model (1 = MinMaxScaler, 2 = StandardScaler, else = RobustScaler)
    - default = MinMaxScaler
    cont_columns: List of columns to scale in DataFrames
    *** Outputs ***
    train: DataFrame with cont_columns scaled.
    val: DataFrame with cont_columns scaled.
    test: DataFrame with cont_columns scaled.
    """
    #Create the scaler
    if scaler_model == 1:
        scaler = MinMaxScaler()
    elif scaler_model == 2:
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()
    
    #Make a copy
    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()

    
    #Fit the scaler
    scaler = scaler.fit(train[cont_columns])
    
    #Build the new DataFrames
    train_scaled[cont_columns] = pd.DataFrame(scaler.transform(train[cont_columns]),
                                                  columns=train[cont_columns].columns.values).set_index([train.index.values])

    val_scaled[cont_columns] = pd.DataFrame(scaler.transform(val[cont_columns]),
                                                  columns=val[cont_columns].columns.values).set_index([val.index.values])

    test_scaled[cont_columns] = pd.DataFrame(scaler.transform(test[cont_columns]),
                                                 columns=test[cont_columns].columns.values).set_index([test.index.values])
    #Sending them back
    return train_scaled, val_scaled, test_scaled

def dec_tree(X_train, y_train, X_val, y_val, metric = 1):
    """
    This function runs the Decission Tree classifier on the training and validation test sets.
    """
    #Create the model
    clf = DecisionTreeClassifier(max_depth=5, random_state=1969)
    
    #Train the model
    clf = clf.fit(X_train, y_train)
    
    #Accuracy
    if metric == 1:
        train_score = clf.score(X_train, y_train)
        val_score =  clf.score(X_val, y_val)
        method = 'Accuracy'
    #Precision
    elif metric == 2:
        #Make a prediction from the model
        y_pred = clf.predict(X_train)
        y_pred_val = clf.predict(X_val)

        train_score = precision_score(y_train, y_pred)
        val_score = precision_score(y_val, y_pred_val)
        method = 'Precision'
        
    #Recall
    elif metric == 3:
        
        #Make a prediction from the model
        y_pred = clf.predict(X_train)
        y_pred_val = clf.predict(X_val)

        train_score = recall_score(y_train, y_pred)
        val_score = recall_score(y_val, y_pred_val)
        method = 'Recall'
        
    #Print the score
    print(f'{method} of Decision Tree classifier on training set:   {train_score:.4f}')
    print(f'{method} of Decision Tree classifier on validation set: {val_score:.4f}')
    
def rand_forest(X_train, y_train, X_val, y_val, metric = 1):
    """
    This function runs the Random Forest classifier on the training and validation test sets.
    """
    #Creating the random forest object
    rf = RandomForestClassifier(bootstrap=True,
                                class_weight=None,
                                criterion='gini',
                                min_samples_leaf=3,
                                n_estimators=100,
                                max_depth=5,
                                random_state=1969)
    
    #Fit the model to the train data
    rf.fit(X_train, y_train)

    #Accuracy
    if metric == 1:
        train_score = rf.score(X_train, y_train)
        val_score =  rf.score(X_val, y_val)
        method = 'Accuracy'
    
    #Precision
    elif metric == 2:
        #Make a prediction from the model
        y_pred = rf.predict(X_train)
        y_pred_val = rf.predict(X_val)

        train_score = precision_score(y_train, y_pred)
        val_score = precision_score(y_val, y_pred_val)
        method = 'Precision'
        
    #Recall
    elif metric == 3:
        
        #Make a prediction from the model
        y_pred = rf.predict(X_train)
        y_pred_val = rf.predict(X_val)

        train_score = recall_score(y_train, y_pred)
        val_score = recall_score(y_val, y_pred_val)
        method = 'Recall'
        
    #Print the score
    print(f'{method} of Random Forest classifier on training set:   {train_score:.4f}')
    print(f'{method} of Random Forest classifier on validation set: {val_score:.4f}')

    
def knn_mod(X_train, y_train, X_val, y_val, metric = 1):
    """
    This function runs the KNN classifier on the training and validation test sets.
    """
    #Creating the model
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')

    #Fitting the KNN model
    knn.fit(X_train, y_train)

    #Accuracy
    if metric == 1:
        train_score = knn.score(X_train, y_train)
        val_score =  knn.score(X_val, y_val)
        method = 'Accuracy'

    #Precision
    elif metric == 2:
        #Make a prediction from the model
        y_pred = knn.predict(X_train)
        y_pred_val = knn.predict(X_val)

        train_score = precision_score(y_train, y_pred)
        val_score = precision_score(y_val, y_pred_val)
        method = 'Precision'

    #Recall
    elif metric == 3:
        
        #Make a prediction from the model
        y_pred = knn.predict(X_train)
        y_pred_val = knn.predict(X_val)

        train_score = recall_score(y_train, y_pred)
        val_score = recall_score(y_val, y_pred_val)
        method = 'Recall'
        
    #Print the score
    print(f'{method} of KNN classifier on training set:   {train_score:.4f}')
    print(f'{method} of KNN classifier on validation set: {val_score:.4f}')
    
def lr_mod(X_train, y_train, X_val, y_val, metric = 1):
    """
    This function runs the Logistic Regression classifier on the training and validation test sets.
    """
    #Creating a logistic regression model
    logit = LogisticRegression(random_state=1969)

    #Fitting the model to the train dataset
    logit.fit(X_train, y_train)

    #Accuracy
    if metric == 1:
        train_score = logit.score(X_train, y_train)
        val_score =  logit.score(X_val, y_val)
        method = 'Accuracy'

    
    #Precision
    elif metric == 2:
        #Make a prediction from the model
        y_pred = logit.predict(X_train)
        y_pred_val = logit.predict(X_val)

        train_score = precision_score(y_train, y_pred)
        val_score = precision_score(y_val, y_pred_val)
        method = 'Precision'

    #Recall
    elif metric == 3:
        
        #Make a prediction from the model
        y_pred = logit.predict(X_train)
        y_pred_val = logit.predict(X_val)

        train_score = recall_score(y_train, y_pred)
        val_score = recall_score(y_val, y_pred_val)
        method = 'Recall'
        
    #Print the score
    print(f'{method} of Logistic Regression classifier on training set:   {train_score:.4f}')
    print(f'{method} of Logistic Regression classifier on validation set: {val_score:.4f}')