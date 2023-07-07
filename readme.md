# 🔧 Predictive Maintenance: Unleashing the Power of Machine Learning on Imbalanced Classes 📊

## 🎯 Overview:

In the realm of equipment maintenance, predicting failure before it happens is the Holy Grail. This project delves into the fascinating world of predictive maintenance, with a twist - it's not just about classification, it's about wrestling with imbalanced classes. 

Through the lens of machine learning, I embarked on a journey to understand how class weights influence algorithm performance. The result? A Random Forest model that slashed false negatives (predicted no failure when there was one) by a whopping 89%, while keeping false positives (predicted failure when there wasn't one) to a minimum. This ensures equipment downtime is kept at bay. 🚀

Imbalanced classes aren't unique to predictive maintenance. They're everywhere - from cyber security intrusion detection to finance fraud detection and medical disease screening. This project offers insights that can be applied across many domains. 🌐

## 🎯 Objectives:
- Unearth the indicators of equipment failure. 🔍
- Harness these indicators to build a machine learning model that can predict equipment failure. 🏗️

## 🤔 Hypothesis:
I ventured into this project with the hypothesis that speed and torque would be significant indicators of equipment failure. 

<details>
<summary>Data Dictionary 📚</summary>

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

</details>

## 📊 Data:
The data for this project was sourced from Kaggle's Machine Predictive Maintenance Classification dataset, originally from the UCI Machine Learning Repository. The dataset comprises 10,000 rows, each representing a piece of equipment, and 10 columns, each representing a feature associated with that equipment. 

## 🧹 Data Preparation:
The data was prepared by handling null values, creating dummy columns for categorical fields, and splitting the data into Train, Validate, and Test sets. 

## 🔍 Key Findings:
All measurements had some relationship with equipment failure, with Torque, Rotational speed, and Tool wear showing the strongest correlations.

## 🤖 Modeling:
The baseline was set at 54% using sklearn's DummyClassifier function. Both Logistic Regression and Random Forest models performed well, with the latter emerging as the best model with a recall of 0.8971 on the test set.

## 🎯 Conclusion:
Failures occur at a rate of 3.4% in the dataset. The key indicators of failure are Torque, Rotational speed, and Tool wear. The Random Forest Model is recommended for identifying failure cases with the least impact on operations.

## 🚀 Future Directions:
- Explore upsampling to improve model performance.
- Investigate clustering failure groups.
- Attempt to identify failure types.
- Establish a better baseline metric.
- Explore the imbalanced-learn library for better solutions.

## 🔄 Replicate My Success:
Want to dive into the code? Just clone this repo, grab the csv file from Kaggle, ensure the wrangle.py, explore.py, and modeling.py are in the same folder as the final notebook, and run the final notebook. Happy coding! 💻

## 📝 Final Thoughts:
Predictive maintenance is a game-changer in many industries. By leveraging machine learning and tackling the challenge of imbalanced classes, we can make strides in predicting equipment failure, saving time, money, and resources. This project is a testament to the power of data science in transforming the way we approach maintenance and failure prediction. 💡
