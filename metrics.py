from typing import Union
import pandas as pd

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    assert y_hat.size == y.size
    return (y_hat == y).mean()

def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    assert y_hat.size == y.size
    tp = ((y_hat == cls) & (y == cls)).sum()
    fp = ((y_hat == cls) & (y != cls)).sum()
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    assert y_hat.size == y.size
    tp = ((y_hat == cls) & (y == cls)).sum()
    fn = ((y_hat != cls) & (y == cls)).sum()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    assert y_hat.size == y.size
    return ((y_hat - y) ** 2).mean() ** 0.5

def mae(y_hat: pd.Series, y: pd.Series) -> float:
    assert y_hat.size == y.size
    return (y_hat - y).abs().mean()