#Import packages
import numpy as np
import pandas as pd

#%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score

#Import data
wells = pd.read_csv('./data/wells.dat', sep=' ')

# Create logistic regression object
wells_regression = LogisticRegression()

# We don't need the id column, so drop it.
X_wells_names = np.array(['arsenic', 'dist', 'assoc', 'educ'])
X_wells = wells[X_wells_names]
# The response is already encoded as 0's, and 1's.
y_wells = wells['switch']
# Fit model
wells_regression.fit(X_wells, y_wells)
#Predict
wells_predictions = wells_regression.predict(X_wells)
print('Final Model Accuracy: ', accuracy_score(y_wells, wells_predictions))
