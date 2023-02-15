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
        if (result.statistic < result.critical_values[0]) and (len(df[col]) > 100):
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
    """
    Shows the relationship between torque, speed and failure.
    """    
    #Color dictionary
    cb_colors = {0:'#377eb8', 1:'#ff7f00', 2:'#4daf4a',
                      3:'#f781bf', 4:'#a65628', 5:'#984ea3',
                      6:'#999999', 7:'#e41a1c', 8:'#dede00'}

    #Creates the figure and axis objects to build upon
    fig, ax = plt.subplots(facecolor='gainsboro', edgecolor='dimgray')
    
    #Groups by the target to put scatterplots on the same figure
    grouped = df.groupby('Target')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='Torque [Nm]', y='Rotational speed [rpm]',
                   marker='.', label=key, color=cb_colors[key])
    #Plots the centroids
    ax.set_xlabel(f'Torque [Nm]')
    ax.set_ylabel(f'Rotational speed [rpm]')
    ax.set_title("Relationship of Torque and Speed")
    ax.legend(['No Failure', 'Failure'])
    plt.show()

def explore_speed(df):
    """
    Shows the relationship between torque, speed and failure.
    """    
    #Color dictionary
    cb_colors = {0:'#377eb8', 1:'#ff7f00', 2:'#4daf4a',
                      3:'#f781bf', 4:'#a65628', 5:'#984ea3',
                      6:'#999999', 7:'#e41a1c', 8:'#dede00'}

    #Creates the figure and axis objects to build upon
    fig, ax = plt.subplots(facecolor='gainsboro', edgecolor='dimgray')
    
    #Groups by the target to put scatterplots on the same figure
    grouped = df.groupby('Target')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x = 'Tool wear [min]', y= 'Rotational speed [rpm]',
                   marker='.', label=key, color=cb_colors[key])
    #Plots the centroids
    ax.set_xlabel(f'Tool wear [min]')
    ax.set_ylabel(f'Rotational speed [rpm]')
    ax.set_title("Relationship of Tool wear and Speed")
    ax.legend(['No Failure', 'Failure'])
    plt.show()


def t_test_cats(train, target_column, alpha = 0.05):
    """
    Input DataFrame and a string of the target_column name.
    Performs chi^2 test with a default alpha of 0.05 on each categorical feature.
    Prints a visualization and list of columns whos data occures exclusivly 
    in the target group or non-target group.
    """

    #Lists to hold variables
    distros = []
    drivers = []
    non_drivers = []
    t_test_result = []
    
    #This snags int columns and drops those that have more than 2 values.
    plot_df = train.select_dtypes(exclude=['object', 'bool', 'datetime'])
    
    #Seperating target rows
    target_df = plot_df[plot_df[target_column] == 1]

    #Warning that the below is prefered... IDK why:
    #df.loc[:,('one','second')]
    target_df.drop(columns=target_column, inplace = True)
    
    #Seperating non-target rows
    not_target = plot_df[plot_df[target_column] == 0]
    not_target.drop(columns=target_column, inplace = True)
    
    #Creating the Target Indication DataFrame

    for item in target_df:
        target = round(target_df[item].mean(),3)
        not_tar = round(not_target[item].mean(),3)
        if item == 'UDI':
            pass
        else:
            output = {"Column" : item,
                      "Target %": target, 
                      "Not Target %": not_tar,
                      "Target Indication":(target - not_tar)}

            distros.append(output)

    #This turns the info into a DataFrame
    distro_df = pd.DataFrame(distros)              
    distro_df = distro_df.set_index('Column')

    #Seperate out columns to investigate, Target Indication = 1 or -1

    for feature in distro_df.T:

    # Let's run a chi squared to compare proportions, to have more confidence
        null_hypothesis = f'{feature} and {target_column} have no difference in means.'
        alternative_hypothesis = f'there is a difference in means between {feature} and {target_column}'

        #Stats test
        t_stat, p_value = stats.ttest_ind(target_df[feature], not_target[feature])

        #Answer logic
        if p_value < alpha:
            t_test_result.append('Different Means')

        else:
            t_test_result.append('Similar Means')
        
    distro_df['t_test_result'] = t_test_result

    
    #Plotting the relationship with the target variable (and stats test result)
    my_range=range(1,len(distro_df.index) + 1)
    hue_colors = {'Different Means': 'green', 'Similar Means':'red'}
    style_order = {'Different Means': 'o', 'Similar Means':'x'}

    plt.figure(figsize=(6,3))
    plt.axvline(0, c='tomato', alpha=.6)

    plt.hlines(y=my_range, xmin=-50, xmax=50, color='grey', alpha=0.4)
    sns.scatterplot(data=distro_df, x='Target Indication',
                    y=my_range, hue='t_test_result', palette=hue_colors,
                    style='t_test_result', style_order=style_order)
    plt.legend(title='T-test result')

    # Add title and axis names
    plt.yticks(my_range, distro_df.index)
    plt.title(f'Comparison of means for {target_column}', loc='center')
    plt.xlabel('T-Statistic')
    plt.ylabel('Feature')
    