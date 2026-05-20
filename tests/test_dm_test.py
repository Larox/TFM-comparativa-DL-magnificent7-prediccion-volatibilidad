import numpy as np
import pandas as pd
import pytest

from tfm_volatility.eval.dm_test import diebold_mariano


def test_dm_zero_when_errors_identical():
    rng = np.random.default_rng(0)
    target = pd.Series(rng.normal(0, 0.01, 100))
    pred_a = target + rng.normal(0, 0.001, 100)
    pred_b = pred_a.copy()
    res = diebold_mariano(target, pred_a, pred_b, horizon=1)
    assert res["dm_stat"] == pytest.approx(0.0, abs=1e-9)
    assert res["p_value"] > 0.5


def test_dm_negative_when_model_a_strictly_better():
    rng = np.random.default_rng(0)
    target = pd.Series(rng.normal(0, 0.01, 200))
    pred_a = target + rng.normal(0, 0.0005, 200)
    pred_b = target + rng.normal(0, 0.005, 200)
    res = diebold_mariano(target, pred_a, pred_b, horizon=1)
    assert res["dm_stat"] < 0
    assert res["p_value"] < 0.05


def test_hln_correction_applied_when_horizon_gt_1():
    rng = np.random.default_rng(0)
    target = pd.Series(rng.normal(0, 0.01, 300))
    pred_a = target + rng.normal(0, 0.001, 300)
    pred_b = target + rng.normal(0, 0.002, 300)
    res_h1 = diebold_mariano(target, pred_a, pred_b, horizon=1)
    res_h5 = diebold_mariano(target, pred_a, pred_b, horizon=5)
    assert abs(res_h5["dm_stat_hln"]) <= abs(res_h1["dm_stat_hln"]) + 1e-9
    assert res_h5["correction_factor"] < 1.0
