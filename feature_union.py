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
from transformer import ColumnSelector
#Import data
wells = pd.read_csv('./data/wells.dat', sep=' ')

# Create logistic regression object
wells_regression = LogisticRegression()

# We don't need the id column, so drop it.
X_wells_names = np.array(['arsenic', 'dist', 'assoc', 'educ'])
X_wells = wells[X_wells_names]

two_columns = FeatureUnion([
    ('arsenic_selector', ColumnSelector([0])),
    ('distance_selector', ColumnSelector([1]))
])
two_columns.fit(X_wells)
print(two_columns.transform(X_wells))
