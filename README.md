# 🤖 AIML
Public Repo of AI/ML projects and Kaggle Competitions undertaken

## Summary
The work done in this repository is exclusively authored by Ian Feekes, who can be contacted via the following:
* Email: ianfeekes@gmail.com
* Linkedin: https://linkedin.com/in/ianfeekes

## Directories
### 🦪 Abalone
This was done as part of a [Kaggle Playground Series Competition](https://www.kaggle.com/competitions/playground-series-s4e4) running regression on Abalone Shell Rings (Age) given the difficulty in collecting the ring data in the field.

This notebook's GBT mobel scored ```0.14982``` on the **RMSLE** evaluation metric, ranking it within the top 100 resuls. Improvements could be made by training the models in such a way to optimize the scoring metric, training the root of the data as part of preprocessing, or optimizing the data based on underestimations since **RMSLE** punishes overestimations notably more than underestimations.

### Steel Plate
This was done as part of a [Kaggle Playground Series Competition](https://www.kaggle.com/c/playground-series-s4e3) to run multicategorical classification to classify the defect type on steel plates. The dataset was particularly unique because the defect categories were not mutually exclusive (e.g. a steel plate could have both a ```K Scratch``` and ```bumps```), so the classification was really for multiple models running binary classification.

This notebook's GBT model scored ```0.8907``` **Area under ROC Curve** which placed it ranking within the **top 50 of results**. With access to more data, or perhaps synthetic data on some of the severely underrepresented categories (such as **bumps**), this notebook's models could perform with notable improvements.

### 🍄 Mushroom Edible or Poisonous
The programming in this directory looks at an [open source dataset from UCI](https://www.kaggle.com/datasets/uciml/mushroom-classification) categorizing various mushrooms as edible or poisionous. While one can use this dataset to create models (especially Random Tree Forests and other hierarchical classifiers) that are extremely highly performant, the most interesting take-aways from the work performed here was in the data analysis.

There are several categories (such as the diameter of the mushroom cap) in which a specific value of the target variable (e.g. edible) is the only value, meaning that many machine learning algorithms trained on this dataset may poorly generalize and lead users to eating poisonous mushrooms.

This also illustrates some of the potential benfits of clustering algorithms: through building algorithms that would allow identification of mushroom species based on clusters, the accuracy would be extremely-performant and allow leveraging of highly-dimensional data such as this.

### 🌸 Petals
The [classic iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) is examined in this notebook. We train a basic logistic regression algorithm to perfectly classify between the three geni of iris, with the help of some feature engineering, such as creating ratios from continuous features.

### 📚 Courses
This directory contains work done to continue my education in AI/ML and Data Science, primarily through IBM Learning and Harvard courses.

### CO2 Rwanda
This notebook is simnply EDA on the CO2 Emissions in Rwanda.