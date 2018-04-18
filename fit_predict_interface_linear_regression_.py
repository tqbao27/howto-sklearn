#Import package
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

# Import data
insects = pd.read_csv('./data/insects.csv', sep='\t')
#insects.head()

# Create linear regression object
insects_regression = LinearRegression()
# Categorize x and y
X_insects = insects[['continent', 'latitude', 'sex']]
y_insects = insects['wingsize']
#Fit linear regression
insects_regression.fit(X_insects, y_insects)

#Predict
wing_size_predictions = insects_regression.predict(X_insects)

#Plot
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

axs[0].scatter(insects.latitude, insects.wingsize, s=40)
axs[0].set_title("Actual Data")
axs[0].set_xlabel("Latitude")
axs[0].set_ylabel("Actual Wing Span")


axs[1].scatter(insects.latitude, wing_size_predictions, s=40)
axs[1].set_title("Predicted Data")
axs[1].set_xlabel("Latitude")
axs[1].set_ylabel("Predicted Wing Span")

plt.show()
