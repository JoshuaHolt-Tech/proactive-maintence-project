import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from env import get_connection
from scipy import stats
import os


from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor


#Removes warnings and imporves asthenics
import warnings
warnings.filterwarnings("ignore")

def correlation_test(df, target_col, alpha=0.05):
    """
    Maybe create a function that automatically seperates continuous from discrete columns.
    """
    
    list_of_cols = df.select_dtypes(include=[int, float]).columns
              
    metrics = []
    for col in list_of_cols:
        result = stats.anderson(df[col])
        #Checks skew to pick a test
        if result.statistic < result.critical_values[2]:
            corr, p_value = stats.pearsonr(df[target_col], df[col])
            test_type = '(P)'
        else:
            # I'm unsure how this handles columns with null values in it.
            corr, p_value = stats.spearmanr(df[target_col],
                                            df[col], nan_policy='omit')
            test_type = '(S)'

        #Answer logic
        if p_value < alpha:
            test_result = 'relationship'
        else:
            test_result = 'independent'
        if col == target_col:
            pass
        else:
            temp_metrics = {"Column":f'{col} {test_type}',
                        "Correlation": corr,
                        "P Value": p_value,
                        "Test Result": test_result}
            metrics.append(temp_metrics)
    distro_df = pd.DataFrame(metrics)              
    distro_df = distro_df.set_index('Column')

    #Plotting the relationship with the target variable (and stats test result)
    my_range=range(1,len(distro_df.index) + 1)
    hue_colors = {'relationship': 'green', 'independent':'red'}

    plt.figure(figsize=(5,4))
    plt.axvline(0, c='tomato', alpha=.6)

    plt.hlines(y=my_range, xmin=-1, xmax=1, color='grey', alpha=0.4)
    sns.scatterplot(data=distro_df, x="Correlation",
                    y=my_range, hue="Test Result", palette=hue_colors,
                    style="Test Result", style_order= hue_colors)
    plt.legend(title="Stats test result")

    # Add title and axis names
    plt.yticks(my_range, distro_df.index)
    plt.title(f'Statistics tests of {target_col}', loc='center')
    plt.xlabel('Neg Correlation            No Correlation            Pos Correlation')
    plt.ylabel('Feature')
    
    #Saves plot when it has a name and uncommented
    #plt.savefig(f'{train.name}.png')

    
def explore_torque(df):
    
    
    
    #Color dictionary
    cb_colors = {0:'#377eb8', 1:'#ff7f00', 2:'#4daf4a',
                      3:'#f781bf', 4:'#a65628', 5:'#984ea3',
                      6:'#999999', 7:'#e41a1c', 8:'#dede00'}

    #Creates the figure and axis objects to build upon
    fig, ax = plt.subplots(facecolor='gainsboro', edgecolor='dimgray')
    
    #Groups by the target to put scatterplots on the same figure
    grouped = df.groupby('Target')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='Torque [Nm]', y='Rotational speed [rpm]', marker='.', label=key, color=cb_colors[key])
    #Plots the centroids
    ax.set_xlabel(f'Torque [Nm]')
    ax.set_ylabel(f'Rotational speed [rpm]')
    ax.set_title("Relationship of Torque and Speed")
    ax.legend(['No Failure', 'Failure'])
    plt.show()