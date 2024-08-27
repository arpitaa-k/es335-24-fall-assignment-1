"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(X)

def check_ifreal(y: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(y)

def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy without using any additional libraries.
    """
    probs = Y.value_counts(normalize=True)  # Calculate the probability of each class
    return -sum(probs * probs.apply(lambda p: (0 if p == 0 else log2(p))))

def log2(x):
    """
    Manually compute the log base 2 of x without importing any libraries.
    """
    from math import log
    return log(x) / log(2)

def gini_index(Y: pd.Series) -> float:
    probs = Y.value_counts(normalize=True)
    return 1 - sum(probs ** 2)

def mse(Y: pd.Series) -> float:
    mean_y = Y.mean()
    return ((Y - mean_y) ** 2).mean()

def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    if criterion == "information_gain":
        base_criterion = entropy(Y)
    elif criterion == "gini_index":
        base_criterion = gini_index(Y)
    else:
        base_criterion = mse(Y)

    values = attr.unique()
    weighted_sum = 0
    for val in values:
        subset = Y[attr == val]
        weight = len(subset) / len(Y)
        if criterion == "information_gain":
            weighted_sum += weight * entropy(subset)
        elif criterion == "gini_index":
            weighted_sum += weight * gini_index(subset)
        else:
            weighted_sum += weight * mse(subset)
    
    if criterion in ["information_gain", "gini_index"]:
        return base_criterion - weighted_sum
    else:
        return weighted_sum

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    best_attr = None
    best_gain = -float('inf')
    
    for feature in features:
        gain = information_gain(y, X[feature], criterion)
        if gain > best_gain:
            best_gain = gain
            best_attr = feature
            
    return best_attr

def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    if pd.api.types.is_numeric_dtype(X[attribute]):
        left_indices = X[attribute] <= value
        right_indices = X[attribute] > value
    else:
        left_indices = X[attribute] == value
        right_indices = X[attribute] != value
        
    return X[left_indices], y[left_indices], X[right_indices], y[right_indices]