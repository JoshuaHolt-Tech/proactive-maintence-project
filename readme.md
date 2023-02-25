Predictive Maintenance Classification: A study in class imbalance
# Abstract:

This project focused on predicting equipment failure when given the physical properties. After starting, I realized it was more about working with imbalanced classes than it was about classification. I learned how a class' weight affect different algorithms and their performance. My Random Forest machine learning model was able to reduce recall or the number of false negative (failure occurred when it was predicted not to) incidents by 89%. It also had a minimal amount of false positive (predict failure when there is none) occurrences to minimize equipment downtime.

Imbalanced classes occur in multiple domains from cyber security intrusion detection, finance fraud detection to medical disease screening.

# Goal:
- Discover the indicators for equipment failure.
- Use indicators to develop a machine learning model to classify equipment as likely or unlikely to fail.

# Initial Thoughts:
	⁃	My initial hypothesis is that speed and torque will be indicators of equipment failure. 

# Data Dictionary:
| Feature |	Definition |
|:--------|:-----------|
|UID| Unique identifier ranging from 1 to 10000|
|ProductID| A variant-specific serial number that begins with the Type.|
|Type| A quality rating indicated by a letter L, M, or H for low (50% of all products), medium (30%), and high (20%) as product quality variants and a variant-specific serial number|
|Air temperature [K]| Generated using a random walk process later normalized to a standard deviation of 2 K around 300 K|
|Process temperature [K]| Generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.|
|Rotational speed [rpm]| Calculated from powepower of 2860 W, overlaid with a normally distributed noise|
|Torque [Nm]| Torque values are normally distributed around 40 Nm with an Ïƒ = 10 Nm and no negative values.|
|Tool wear [min]|  The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process.|
|Target|  Label that indicates, whether the machine has failed.|
|Failure Type| Includes: No Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure, Tool Wear Failure, Random Failures.|

# Acquire:
- Data acquired from Kaggle for Machine Predictive Maintenance Classification.
    - https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification?select=predictive_maintenance.csv
- Originally from:
    - https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset
- The query resulted in 10 columns and 10,000 rows.
- Each row represents a record about a piece of equipment.
- Each column represents a feature associated with that equipment.

# Prepare:
1. No null values.
2. Changed column for temperature difference between the process and air.
3. Created dummy columns for categorical fields.
4. Dropped unused columns.
5. Split data into Train, Validate and Test sets (approx. 60/20/20), stratifying on 'Target'.

# Steps to Reproduce:
1. Copy this repo.
2. Acquire the csv file from Kaggle.
3. Ensure the wrangle.py, explore.py and modeling.py are in the same folder as the final notebook.
3. Run the final notebook.

# Takeaways:
All measurements which were shown to have a relationship with equipment failure. The strongest correlations are:
- Torque [Nm]
- Rotational speed [rpm]
- Tool wear [min]

# Modeling Summary:

- Baseline normally would be 0. I used 54% based of sklearn DummyClassifier function.
- Logistic Regression and Random Forest performed the best using Recall
- Looking at the classification report, Logistic Regression is predicting failure most of the time.
- Random Forest is doing a much better job overall.

# Conclusion:
The best performing model was the Random Forest classifier
- Training set recall: 0.9458
- Validation set recall: 0.9706
- Test set recall: 0.8971

# Summary:
Failures occurs at 3.4% in the dataset. The indicators of failure are: 
- Torque [Nm]
- Rotational speed [rpm]
- Tool wear [min]

# Recommendations:
- Establish inspection methods to quickly identify false positive cases and restart equipment.
- The Random Forest Model is the best for identifying failure cases and has the least amount of impact on operations.

# Next Steps:
- Explore upsampling to improve model performance (reduce false positive cases).
- Explore clustering failure groups.
- Attempt to identify failure types.
- Establish a better baseline metric.
- Explore the imbalanced-learn library for better solutions.