import os
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

import rampwf as rw

problem_title = "Shapes and patterns classification"

_labels = ['circle', 'cross', 'heptagon', 'hexagon', 'octagon', 'pentagon',
           'quartercircle', 'rectangle', 'semicircle', 'square', 'star', 'trapezoid', 'triangle']

# Correspondence between categories and int8 categories
# Mapping int to categories
int_to_cat = {
    0: "circle",
    1: "cross",
    2: "heptagon",
    3: "hexagon",
    4: "octagon",
    5: 'pentagon',
    6: 'quartercircle',
    7: 'rectangle',
    8 : 'semicircle',
    9 : 'square',
    10 : 'star',
    11 : 'trapezoid',
    12 : 'triangle',
}

# Mapping categories to int
cat_to_int = {v: k for k, v in int_to_cat.items()}

_labels = list(int_to_cat)

Predictions = rw.prediction_types.make_multiclass(label_names=_labels)
workflow = rw.workflows.Classifier()

score_types = [
    rw.score_types.BalancedAccuracy(
        name="bal_acc", precision=3, adjusted=False
    ),
    rw.score_types.Accuracy(name="acc", precision=3),
]

# Create global variable to use in LOGO CV strategy
groups = None


def _get_data(path=".", split="train"):
    # Load data from CSV files.
    # Data: CSV files containing information
    # Labels: Extracted from the CSV files
    #
    # returns X (input) and y (output) arrays
    # X: array-like (features)
    # y: array-like (labels)

    # data
    X = np.load(os.path.join(path, "data", "X_" + split + ".npy"))

    # labels
    y = pd.read_csv(os.path.join(path, "data", "y_" + split + ".csv"))

    return X, y

def get_train_data(path="."):
    # Load y_df from the training CSV file
    # Return data
    y_df = pd.read_csv(os.path.join(path, "data/y_train.csv"))
    
    # Global variable groups
    global groups
    groups = y_df.to_numpy().ravel()

    return _get_data(path, "train")

def get_test_data(path="."):
    # Load test data
    return _get_data(path, "test")

def get_cv(X, y):
    # Perform cross-validation using LeaveOneGroupOut strategy
    cv = LeaveOneGroupOut()
    return cv.split(X, y, groups=groups)
