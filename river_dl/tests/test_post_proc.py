import numpy as np
import pandas as pd
from postproc_utils import rmse_masked, nse


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


def test_nse():
    y_true = pd.Series([1, 5, 3, 4, 2])
    y_pred = y_true.copy()
    nse_samp = nse(y_true, y_pred)
    assert nse_samp == 1

    y_pred = pd.Series([1, 4, 0, 4, 2])
    nse_samp = nse(y_true, y_pred)
    assert nse_samp == 0

    y_pred = pd.Series([2, 4, 0, 4, 2])
    nse_samp = nse(y_true, y_pred)
    assert round(nse_samp, 1) == -0.1

    y_pred = pd.Series([1, 4, 2, 4, 2])
    nse_samp = nse(y_true, y_pred)
    assert round(nse_samp, 1) == 0.8

    y_true = pd.Series([1, np.nan, 3, 4, 2])
    y_pred = pd.Series([1, 4, 2, 4, 2])
    nse_samp = nse(y_true, y_pred)
    assert round(nse_samp, 1) == 0.8

    y_true = pd.Series([1, np.nan, 3, 4, np.nan])
    y_pred = pd.Series([1, 4, 2, 4, 2])
    nse_samp = nse(y_true, y_pred)
    assert round(nse_samp, 2) == 0.79

    y_true = pd.Series([1, np.nan, 2, 4, np.nan])
    y_pred = pd.Series([1, 4, 2, 4, 2])
    nse_samp = nse(y_true, y_pred)
    assert nse_samp == 1
