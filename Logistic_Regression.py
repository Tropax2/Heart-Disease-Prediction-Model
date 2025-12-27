'''
We use logistic regression to predict if a given patient will develop 
a heart disease. The predictors given in the CSV  are:
- Demographic: age, sex 
- Clinical: CP (chest pain type), trestbps (blood pressure), chol (cholestrol)
fbs (fasting blood sugar), restecg (Resting electrocardiographic results), 
thalach (Maximum heart rate achieved), exang (Exercise-induced angina), 
oldpeak (ST depression induced by exercise relative to rest), slope, 
ca (Number of major vessels), thal (Thalassemia)

We use the confusion matrix to verify, by using a threshold of 0.5, to compute 
the ratio to check the results and compare with other models.
'''
import numpy as np
import pandas as pd
import statsmodels.api as sm
from ISLP import confusion_table
from ISLP.models import ModelSpec as MS, summarize

# Import the data set 
Heart_Disease = pd.read_csv(r"C:\Users\Antonio\Desktop\Heart Disease Model\heart.csv")

# Remove rows with null values 
Heart_Disease.dropna()

# Convert the categorical variables into categorical type 
categorical_predictors = ["sex", "cp", "fbs",  "thal", "restecg", "exang", "slope", "ca", "thal"] 
for col in categorical_predictors:
    Heart_Disease[col] = Heart_Disease[col].astype("category")

predictors = Heart_Disease.drop(columns="target")
X = MS(predictors).fit_transform(Heart_Disease)
Y = Heart_Disease.target == 1

# Apply the logistic regression 
model = sm.GLM(Y, X, family=sm.families.Binomial())
results = model.fit()
# print(summarize(results))
'''
We verify that certain predictors are not statistically significant like age, trestbps, chol, fbs, restecg, slope and thal.
'''
# Compute the prediction rate using a threshold of 0.5.
probs = results.predict(X)
labels = np.array(0 * X.shape[0])
labels = probs > 0.5 
#print(confusion_table(labels, Y))
print(np.mean(labels == Y))
'''
The correct prediction rate of the the model is 88.4% 
'''

#######################

'''
We repeat the same procedure but we drop the predictors that are not considered statistically significant.
'''
# Define the predictors 
predictors = Heart_Disease.drop(columns=["age", "trestbps", "chol", "fbs", "restecg", "slope", "thal", "target"])
X = MS(predictors).fit_transform(Heart_Disease)
Y = Heart_Disease.target == 1

# Apply the logistic regression 
model = sm.GLM(Y, X, family=sm.families.Binomial())
results = model.fit()
#print(summarize(results))

# Compute the prediction rate using a threshold of 0.5.
probs = results.predict(X)
labels = np.array(0 * X.shape[0])
labels = probs > 0.5 
#print(confusion_table(labels, Y))
#print(np.mean(labels == Y))
'''
We verify that after removing the predictors that are not statistically significant, there is some loss of 
information and so the prediction rate drops to 83.83%.
'''