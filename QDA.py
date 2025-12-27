'''
We use LDA to predict if a given patient will develop 
a heart disease. The predictors given in the CSV  are:
- Demographic: age, sex 
- Clinical: CP (chest pain type), trestbps (blood pressure), chol (cholestrol)
fbs (fasting blood sugar), restecg (Resting electrocardiographic results), 
thalach (Maximum heart rate achieved), exang (Exercise-induced angina), 
oldpeak (ST depression induced by exercise relative to rest), slope, 
ca (Number of major vessels), thal (Thalassemia)

We use the confusion matrix to verify to compute the ratio of correct predictions  
to compare with other models.
'''
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from ISLP import confusion_table 

# Import and remove rows with empty values
Heart_Disease = pd.read_csv(r"path")
Heart_Disease.dropna()

X = Heart_Disease.drop(columns='target')
Y = Heart_Disease['target']

# Transform categorical predictors into categorical type
categorical_predictors =  ["sex", "cp", "fbs",  "thal", "restecg", "exang", "slope", "ca", "thal"] 
for col in categorical_predictors:
    Heart_Disease[col] = Heart_Disease[col].astype("category")

# Apply One hot encoding and standardize numerical predictors
enc = OneHotEncoder(drop='first')
enc.fit(X[["cp", "thal", "restecg", "sex"]])

# Standardize the numerical predictors
scaler = StandardScaler()
scaler.fit(X[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']])

# Split the data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42, shuffle=True)

# fit the LDA model 
clf = QDA()
clf.fit(X_train, Y_train)

# predictions 
predicted = clf.predict(X_test)

print(confusion_table(predicted, Y_test))
print(np.mean(predicted == Y_test))
'''
We verify that the correct prediction rate of this model, reserving 15% of the data for testing, is of 78.2%
'''

