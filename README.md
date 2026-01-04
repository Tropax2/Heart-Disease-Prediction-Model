This repo contains a prediction model applied that predicts wether or not an individual is likely to have heart disease.

Since it regards a classification problem we use different models and compare them via a validation set approach: Logistic Regression, LDA, QDA, Naive Bayes and K nearest neighbour.

The files it contains are the following:
- The dataset in CSV format from Kaggle (https://www.kaggle.com/datasets/arezaei81/heartcsv);
- Python files with the treatment of the data and the implementation of the model, as well as its performance.

Features:
- Loads and preprocesses the dataset;
- Implements a predictive model for heart disease;
- Evaluates model performance.

Dependencies:
- Python 3;
- NumPy;
- Pandas;
- Scikit-learn;
- Statsmodels.

Usage:
- Copy the repo;
- Ensure the dataset CSV is in the correct location or update the path in the script;
- Run the Python script.

Tests on a Validation Set:
- Splitted the data into 85 % training and 15 % testing with the same seed for shuffling;
- Computed the correct prediction rate on the same validation set;

Results (Validation Set Approach):
- Logistic Regression: 88.4%
- LDA: 84.7%
- Naive Bayes: 80.4%
- KNN for K = 15: 78.9%
- QDA: 78.2%
- KNN for K = 10: 73.9%
- KNN for K = 20: 71%
- KNN for K = 1, or K = 5,: 69.6%


