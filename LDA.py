'''
We use logistic regression to predict if a given patient will develop 
a heart disease. The predictors given in the CSV  are:
- Demographic: age, sex 
- Clinical: CP (chest pain type), trestbps (blood pressure), chol (cholestrol)
fbs (fasting blood sugar), restecg (Resting electrocardiographic results), 
thalach (Maximum heart rate achieved), exang (Exercise-induced angina), 
oldpeak (ST depression induced by exercise relative to rest), slope, 
ca (Number of major vessels), thal (Thalassemia)

We use the confusion matrix to verify to compute the ratio and check the results 
to compare with other models.
'''
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
from ISLP import confusion_table 

