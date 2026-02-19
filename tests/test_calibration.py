import numpy as np
import pytest

from glidertools.calibration import (
    bottle_matchup,
    model_figs,
    model_metrics,
    robust_linear_fit,
)
from glidertools.helpers import GliderToolsError


def test_bottle_matchup_match():
    # one dive, 5 depth points, bottle sample close in time and depth
    gld_dives = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    gld_depth = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    base = np.datetime64("2020-01-01T12:00")
    gld_time = np.array([base + np.timedelta64(i, "m") for i in range(5)])

    # bottle at ~30m, taken 10 min after glider start
    btl_depth = np.array([30.0])
    btl_time = np.array([base + np.timedelta64(10, "m")])
    btl_values = np.array([99.9])

    result = bottle_matchup(
        gld_dives,
        gld_depth,
        gld_time,
        btl_depth,
        btl_time,
        btl_values,
    )

    # should match at index 2 (depth=30)
    assert result[2] == 99.9
    # everything else should be nan
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    assert np.isnan(result[3])
    assert np.isnan(result[4])


def test_bottle_matchup_no_match_time():
    # bottle sample too far in time (>120 min default)
    gld_dives = np.array([1.0, 1.0, 1.0])
    gld_depth = np.array([10.0, 20.0, 30.0])
    base = np.datetime64("2020-01-01T12:00")
    gld_time = np.array([base + np.timedelta64(i, "m") for i in range(3)])

    btl_depth = np.array([20.0])
    btl_time = np.array([base + np.timedelta64(200, "m")])  # 200 min away
    btl_values = np.array([50.0])

    result = bottle_matchup(
        gld_dives,
        gld_depth,
        gld_time,
        btl_depth,
        btl_time,
        btl_values,
    )
    # nothing should match
    assert np.all(np.isnan(result))


def test_bottle_matchup_no_match_depth():
    # bottle close in time but depth diff > 5m threshold
    gld_dives = np.array([1.0, 1.0, 1.0])
    gld_depth = np.array([10.0, 20.0, 30.0])
    base = np.datetime64("2020-01-01T12:00")
    gld_time = np.array([base + np.timedelta64(i, "m") for i in range(3)])

    btl_depth = np.array([100.0])  # way deeper than any glider point
    btl_time = np.array([base + np.timedelta64(1, "m")])
    btl_values = np.array([50.0])

    result = bottle_matchup(
        gld_dives,
        gld_depth,
        gld_time,
        btl_depth,
        btl_time,
        btl_values,
    )
    assert np.all(np.isnan(result))


def _fit_huber(x, y):
    """quick helper so we dont repeat the fitting boilerplate"""
    from sklearn.linear_model import HuberRegressor

    m = HuberRegressor(fit_intercept=False)
    m.fit(x.reshape(-1, 1), y)
    return m


def test_model_metrics_keys():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2.0 * x
    model = _fit_huber(x, y)
    result = model_metrics(x, y, model)

    # check that all the keys we expect are present
    for k in (
        "model_type",
        "model_slope",
        "model_intercept",
        "r2_all",
        "r2_robust",
        "rmse_all",
        "rmse_robust",
    ):
        assert k in result


def test_model_metrics_perfect_fit():
    # perfect y = 2x, so r2 should be 1 and rmse 0
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2.0 * x
    model = _fit_huber(x, y)
    result = model_metrics(x, y, model)

    assert result["r2_all"] == pytest.approx(1.0, abs=1e-6)
    assert result["rmse_all"] == pytest.approx(0.0, abs=1e-6)
    assert result["model_slope"] == pytest.approx(2.0, abs=1e-4)


def test_model_figs_returns_axes():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.axes import Axes

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2.0 * x
    model = _fit_huber(x, y)

    ax = model_figs(y, x, model)
    assert isinstance(ax, Axes)


def test_robust_linear_fit_slope():
    gld_var = np.arange(1.0, 11.0)

    # simulate bottle matchup output - mostly nans with a few real values
    gld_var_cal = np.full(10, np.nan)
    gld_var_cal[0] = 2.0
    gld_var_cal[4] = 10.0
    gld_var_cal[9] = 20.0

    model = robust_linear_fit(gld_var, gld_var_cal, return_figures=False)
    assert model.coef_[0] == pytest.approx(2.0, abs=0.1)


def test_robust_linear_fit_nan_predict():
    gld_var = np.arange(1.0, 11.0)
    gld_var_cal = np.full(10, np.nan)
    gld_var_cal[0] = 2.0
    gld_var_cal[4] = 10.0
    gld_var_cal[9] = 20.0

    model = robust_linear_fit(gld_var, gld_var_cal, return_figures=False)

    test_input = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    out = model.predict(test_input)

    # nans in, nans out
    assert np.isnan(out[1])
    assert np.isnan(out[3])
    assert not np.isnan(out[0])
    assert not np.isnan(out[2])
    assert not np.isnan(out[4])


def test_robust_linear_fit_all_nan_raises():
    gld_var = np.arange(1.0, 6.0)
    gld_var_cal = np.full(5, np.nan)

    with pytest.raises(GliderToolsError):
        robust_linear_fit(gld_var, gld_var_cal, return_figures=False)
