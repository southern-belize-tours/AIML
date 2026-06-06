import numpy as np


def mse(y_pred, y_true):
    return np.square(y_pred - y_true)

def mse_d(y_pred, y_true):
    return 2 * (y_pred - y_true)
