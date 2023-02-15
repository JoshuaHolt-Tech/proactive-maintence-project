#Tools to build machine learning models and reports
from sklearn.metrics import classification_report, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.dummy import DummyClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#global variable
random_seed = 1969


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
    train, validate = train_test_split(train, train_size=.75, stratify=stratify_arg, random_state = random_seed)
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

def prep_maint_data(df, target_col = None):
    """
    This function prepares the data for machine learning processing.
    """
    target_col = 'Target'
    dumb_cols = ['Type']
    cont_columns = ['Air temperature [K]', 'Process temperature [K]',
                    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
                    'Temp Delta [K]']
    
    
    df = pd.get_dummies(df, columns = dumb_cols)
    df = df.drop(columns=['UDI', 'Product ID'])
    
    
    train, val, test = train_validate(df, stratify_col = target_col)
    
    
    train_scaled, val_scaled, test_scaled = scale_cont_columns(train, val, test, cont_columns, scaler_model = 1)
    
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test(train_scaled, val_scaled, test_scaled, target_col)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def find_baseline(X_train, y_train, print_baseline = False):
    """
    This function gets a baseline.
    """
    baseline = DummyClassifier(strategy='uniform', random_state=random_seed)
    baseline.fit(X_train, y_train)
    b = baseline.predict(X_train)
    if print_baseline == True:
        print(f'Baseline score: {recall_score(y_train, b):.4f}')
    return recall_score(y_train, b)




def dec_tree(X_train, y_train, X_val, y_val, metric = 1, print_scores = False):
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
    if print_scores == True:
        print(f'{method} for Decision Tree classifier on training set:   {train_score:.4f}')
        print(f'{method} for Decision Tree classifier on validation set: {val_score:.4f}')
        print(classification_report(y_val, y_pred_val))
    
    return train_score, val_score


def rand_forest(X_train, y_train, X_val, y_val, metric = 1, print_scores = False):
    """
    This function runs the Random Forest classifier on the training and validation test sets.
    """
    #Creating the random forest object
    rf = RandomForestClassifier(max_depth=4, 
                                class_weight = 'balanced', 
                                criterion = 'entropy',
                                n_jobs = -1,
                                min_samples_leaf = 3,
                                n_estimators = 100,
                                random_state = 1969)
    
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
    if print_scores == True:
        print(f'{method} for Random Forest classifier on training set:   {train_score:.4f}')
        print(f'{method} for Random Forest classifier on validation set: {val_score:.4f}')
        print(classification_report(y_val, y_pred_val))

    return train_score, val_score

def knn_mod(X_train, y_train, X_val, y_val, metric = 1, print_scores = False):
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
    if print_scores == True:
        print(f'{method} for KNN classifier on training set:   {train_score:.4f}')
        print(f'{method} for KNN classifier on validation set: {val_score:.4f}')
        print(classification_report(y_val, y_pred_val))

    return train_score, val_score
    
def lr_mod(X_train, y_train, X_val, y_val, metric = 1, print_scores = False):
    """
    This function runs the Logistic Regression classifier on the training and validation test sets.
    """
    #Creating a logistic regression model
    logit = LogisticRegression(random_state=1969,
                               class_weight='balanced',
                               solver = 'sag',
                               penalty = 'none')

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
    if print_scores == True:
        print(f'{method} for Logistic Regression classifier on training set:   {train_score:.4f}')
        print(f'{method} for Logistic Regression classifier on validation set: {val_score:.4f}')
        print(classification_report(y_val, y_pred_val))
    
    return train_score, val_score
    
def find_model_scores(X_train, y_train, X_val, y_val, metric = 1, print_scores = False):
    """
    This function takes in the target DataFrame, runs the data against four
    machine learning models and outputs some visuals.
    """

    #Eastablishes the standard to beat
    baseline = find_baseline(X_train, y_train)
    
    #List for gathering metrics
    model_scores = []

    
    """ *** Builds and fits Decision Tree *** """
    
    
    train_score, val_score = dec_tree(X_train, y_train, X_val, y_val, metric=metric)

    #Adds score to metrics list for later comparison
    model_scores.append({'Model':'Decision Tree',
                    'Recall on Train': round(train_score,4),
                    'Recall on Validate': round(val_score,4)})
    
    
    """ *** Builds and fits Random Forest Model *** """
   
    
    train_score, val_score = rand_forest(X_train, y_train, X_val, y_val, metric=metric)
    
    #Adds score to metrics list for later comparison
    model_scores.append({'Model':'Random Forest',
                    'Recall on Train': round(train_score,4),
                    'Recall on Validate': round(val_score,4)})
    
    
    """ *** Builds and fits KNN Model *** """
    
    train_score, val_score = knn_mod(X_train, y_train, X_val, y_val, metric=metric)
    
    #Adds score to metrics list for later comparison
    model_scores.append({'Model':'KNN',
                        'Recall on Train': round(train_score,4),
                        'Recall on Validate': round(val_score,4)})
    
    
    """ *** Builds and fits Polynomial regression Model *** """

    
    train_score, val_score = lr_mod(X_train, y_train, X_val, y_val, metric=metric)

    #Adds score to metrics list for later comparison
    model_scores.append({'Model':'Logistic Regression',
                        'Recall on Train': round(train_score,4),
                        'Recall on Validate': round(val_score,4)})
    
    """ *** Later comparison section to display results *** """
    
    #Builds and displays results DataFrame
    model_scores = pd.DataFrame(model_scores)
    model_scores['Difference'] = round(model_scores['Recall on Train'] - model_scores['Recall on Validate'],2)    
    
    #Results were too close so had to look at the numbers
    if print_scores == True:
        print(model_scores)
    
    #Building variables for plotting
    recall_min = min([model_scores['Recall on Train'].min(),
                    model_scores['Recall on Validate'].min(), baseline])
    recall_max = max([model_scores['Recall on Train'].max(),
                    model_scores['Recall on Validate'].max(), baseline])

    lower_limit = recall_min * 0.8
    upper_limit = recall_max * 1.05


    x = np.arange(len(model_scores))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(facecolor="gainsboro")
    ax.axhspan(0, baseline, facecolor='red', alpha=0.2)
    ax.axhspan(baseline, upper_limit, facecolor='palegreen', alpha=0.3)
    rects1 = ax.bar(x - width/2, model_scores['Recall on Train'],
                    width, label='Training data', color='#4e5e33',
                    edgecolor='dimgray') #Codeup dark green
    rects2 = ax.bar(x + width/2, model_scores['Recall on Validate'],
                    width, label='Validation data', color='#8bc34b',
                    edgecolor='dimgray') #Codeup light green

    # Need to have baseline input:
    plt.axhline(baseline, label="Baseline Recall", c='red', linestyle=':')

    # Add some text for labels, title and custom x-axis tick labels, etc.

    ax.set_ylabel('Recall Score')
    ax.set_xlabel('Machine Learning Models')
    ax.set_title('Model Recall Scores')
    ax.set_xticks(x, model_scores['Model'])

    plt.ylim(bottom=lower_limit, top = upper_limit)

    ax.legend(loc='upper left', framealpha=.9, facecolor="whitesmoke",
              edgecolor='darkolivegreen')

    #ax.bar_label(rects1, padding=4)
    #ax.bar_label(rects2, padding=4)
    fig.tight_layout()
    #plt.savefig('best_model_all_features.png')
    plt.show()
    
    
    
def final_test(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    This function takes in the target DataFrame, runs the data against the
    machine learning model selected for the final test and outputs some visuals.
    """
    
    #Eastablishes the standard to beat
    baseline = find_baseline(X_train, y_train)
    
    #List for gathering metrics
    final_model_scores = []
    
    """ *** Builds and fits Random Forest Model *** """  
    
    #Creating the random forest object
    rf = RandomForestClassifier(max_depth=4, 
                                class_weight = 'balanced', 
                                criterion = 'entropy',
                                n_jobs = -1,
                                min_samples_leaf = 3,
                                n_estimators = 100,
                                random_state = 1969)

    #Fit the model to the train data
    rf.fit(X_train, y_train)
    
    #Make a prediction from the model
    y_pred = rf.predict(X_train)
    y_pred_val = rf.predict(X_val)
    y_pred_test = rf.predict(X_test)

    #Get the recall scores
    train_score = recall_score(y_train, y_pred)
    val_score = recall_score(y_val, y_pred_val)
    test_score = recall_score(y_test, y_pred_test)    
    
    #Adds score to metrics list for comparison
    final_model_scores.append({'Model':'Random Forest',
                              'Recall on Train': round(train_score,4), 
                              'Recall on Validate': round(val_score,4), 
                              'Recall on Test': round(test_score,4)})
    #Turn scores into a DataFrame
    final_model_scores = pd.DataFrame(data = final_model_scores)
    print(final_model_scores)
    
    #Create visuals to show the results
    fig, ax = plt.subplots(facecolor="gainsboro")

    plt.figure(figsize=(6,6))
    ax.set_title('Random Forest results')
    ax.axhspan(0, baseline, facecolor='red', alpha=0.2)
    ax.axhspan(baseline, ymax=2, facecolor='palegreen', alpha=0.3)
    ax.axhline(baseline, label="Baseline", c='red', linestyle=':')

    ax.set_ylabel('RMS Error')    

    #x_pos = [0.5, 1, 1.5]
    width = 0.25

    bar1 = ax.bar(0.5, height=final_model_scores['Recall on Train'],width =width, color=('#4e5e33'), label='Train', edgecolor='dimgray')
    bar2 = ax.bar(1, height= final_model_scores['Recall on Validate'], width =width, color=('#8bc34b'), label='Validate', edgecolor='dimgray')
    bar3 = ax.bar(1.5, height=final_model_scores['Recall on Test'], width =width, color=('tomato'), label='Test', edgecolor='dimgray')

    # Need to have baseline input:
    ax.set_xticks([0.5, 1.0, 1.5], ['Training', 'Validation', 'Test']) 
    ax.set_ylim(bottom=0, top=1)
    #Zoom into the important area
    #plt.ylim(bottom=200000, top=400000)
    ax.legend(loc='lower right', framealpha=.9, facecolor="whitesmoke", edgecolor='darkolivegreen')
    
    
def lr_mod_preds_plot(X_train, y_train, X_val):
    """
    This function runs the Logistic Regression classifier on the training and validation test sets.
    """

    
def show_preds(X_train, y_train, X_val):
    """
    This function shows how the predictions are distributed.
    """

    rf = RandomForestClassifier(max_depth=4, 
                                class_weight = 'balanced', 
                                criterion = 'entropy',
                                n_jobs = -1,
                                min_samples_leaf = 3,
                                n_estimators = 100,
                                random_state = 1969)

    #Fit the model to the train data
    rf.fit(X_train, y_train)
    
    #Make a prediction from the model
    y_pred_rf = rf.predict(X_train)
    y_pred_val_rf = rf.predict(X_val)
    
    #Creating a logistic regression model
    logit = LogisticRegression(random_state=1969,
                               class_weight='balanced',
                               solver = 'sag',
                               penalty = 'none')

    #Fitting the model to the train dataset
    logit.fit(X_train, y_train)
    
    #Make a prediction from the model
    y_pred_lr = logit.predict(X_train)
    y_pred_val_lr = logit.predict(X_val)
    
    #Creates the subplots
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, facecolor="gainsboro")
    
    #Subplot for random forest
    ax1.hist(y_pred_val_rf, color = '#8bc34b', ec = '#4e5e33')
    ax1.set_xticks([0,1], ['No failure', 'Failure'])
    ax1.set_title("Random Forest Predictions")
    
    #Subplot for logistic regression
    ax2.hist(y_pred_val_lr, color = 'tomato', ec = '#4e5e33')
    ax2.set_xticks([0,1], ['No failure', 'Failure'])
    ax2.set_title('Logistic Regression Predictions')
