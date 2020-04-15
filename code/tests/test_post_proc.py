import numpy as np
import pandas as pd
from postproc_utils import rmse_masked


def test_rmse_masked():
    y_true = pd.Series([1, 5, 3, 4, 2])
    y_pred = y_true.copy()
    err = rmse_masked(y_true, y_pred)
    assert err == 0

    y_pred = pd.Series([0, 0, 0, 0, 0])
    err = rmse_masked(y_true, y_pred)
    assert round(err, 2) == 3.32

    y_true = pd.Series([1, np.nan, 3, 4, 2])
    err = rmse_masked(y_true, y_pred)
    assert round(err, 2) == 2.74
