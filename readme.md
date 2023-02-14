Predictive Maintenance Classification: A study in class imbalance

Goal:
- Discover the indicators for equipment failure.
- Use indicators to develop a machine learning model to classify equipment as likely or unlikely to fail.

Acquire:
- Data acquired from Kaggle for Machine Predictive Maintenance Classification.
    - https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification?select=predictive_maintenance.csv
- Origionally from:
    - https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset
- The query resulted in 10 columns and 10,000 rows.
- Each row represents a record about a piece of equipment.
- Each column represents a feature associated with that equipment.

Prepare:
1. No null values.
2. Changed column for temperature difference between the process and air.
3. Created dummy columns for categorical fields.
4. Dropped unused columns.
5. Split data into Train, Validate and Test sets (approx. 60/20/20), stratifying on 'Target'.

Data Dictionary:
| Feature |	Definition |
|:--------|:-----------|
|UID| Unique identifier ranging from 1 to 10000|
|ProductID| A variant-specific serial number that begines with the Type.|
|Type| A quality rating indicated by a letter L, M, or H for low (50% of all products), medium (30%), and high (20%) as product quality variants and a variant-specific serial number|
|Air temperature [K]| Generated using a random walk process later normalized to a standard deviation of 2 K around 300 K|
|Process temperature [K]| Generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.|
|Rotational speed [rpm]| Calculated from powepower of 2860 W, overlaid with a normally distributed noise|
|Torque [Nm]| Torque values are normally distributed around 40 Nm with an Ïƒ = 10 Nm and no negative values.|
|Tool wear [min]|  The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process.|
|Target|  Label that indicates, whether the machine has failed.|
|Failure Type| Includes: No Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure, Tool Wear Failure, Random Failures.|

