#Import packages
import numpy as np
import pandas as pd

class PolynomialExpansion(object):

    def __init__(self, degree):
        self.degree = degree

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        # Initialize our return value as a matrix of all zeros.
        # We are going to overwrite all of these zeros in the code below.
        X_poly = np.zeros((X.shape[0], self.degree))
        # The first column in our transformed matrix is just the vector we started with.
        X_poly[:, 0] = X.squeeze()
        # Cleverness Alert:
        # We create the subsequent columns by multiplying the most recently created column
        # by X.  This creates the sequence X -> X^2 -> X^3 -> etc...
        for i in range(2, self.degree + 1):
            X_poly[:, i-1] = X_poly[:, i-2] * X.squeeze()
        return X_poly

class ColumnSelector(object):

    def __init__(self, idxs):
        self.idxs = np.asarray(idxs)

    # Fit here doesn't need to do anything.  We already know the indices of the columns
    # we want to keep.
    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        # Need to teat pandas data frames and numpy arrays slightly differently.
        if isinstance(X_wells, pd.DataFrame):
            return X.iloc[:, self.idxs]
        return X[:, self.idxs]
