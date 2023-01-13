import numpy as np
import pytest


def test_sunrise_sunset():
    """
    Tests if sunrise/sunset:
        1. can run
        2. output is the right shape
        3. if the output is correct-ish
    """
    import numpy as np
    import pandas as pd

    from glidertools.optics import sunset_sunrise

    time = [
        np.datetime64("2000-01-01"),
        np.datetime64("2000-01-02"),
        np.datetime64("2000-01-03"),
    ]
    lat = -35, 45, 80
    lon = 0, 0, 0
    sunrise, sunset = sunset_sunrise(time, lat, lon)

    # Three entries, there should be three outputs
    assert len(sunrise) == len(lat)

    # sunrise will be earlier in the SH in January
    assert sunrise[0] < sunrise[2]

    # expect sunrise at the 4am, 7am and 11am for these times and latitudes
    # high latitude should output polar night default 11:59 for sunrise and 12:01 for sunset
    assert pd.to_datetime(sunrise[0]).hour == 4
    assert pd.to_datetime(sunrise[1]).hour == 7
    assert pd.to_datetime(sunrise[2]).hour == 11

    # high latitude should output polar night default 11:59 for sunrise and 12:01 for sunset
    assert pd.to_datetime(sunrise[2]).hour == 11
    assert pd.to_datetime(sunrise[2]).minute == 59

    assert pd.to_datetime(sunset[2]).hour == 12
    assert pd.to_datetime(sunset[2]).minute == 1


@pytest.mark.parametrize("percentile", [5, 50, 95])
def test_backscatter_dark_count(percentile):
    from glidertools.optics import backscatter_dark_count

    # create some synthetic data
    bbp = np.array([0.002, 0.0006, 0.0005, 0.0005, 0.0005])
    depth = np.array([50, 150, 210, 310, 350])
    # select only depths between 200 and 400
    mask = (depth > 200) & (depth < 400)
    # expected output
    expected_bbp_dark = bbp - np.nanpercentile(bbp[mask], percentile)
    bbp_dark = backscatter_dark_count(bbp, depth, percentile)
    np.testing.assert_allclose(expected_bbp_dark, bbp_dark)


@pytest.mark.parametrize("percentile", [5, 50, 95])
def test_backscatter_dark_count_negative(percentile):
    from glidertools.optics import backscatter_dark_count

    # create some synthetic data
    bbp = np.array(
        [0.002, 0.0006, 0.005, 0.005, 0.0004]
    )  # this will result in negative values that should be zeroed out
    depth = np.array([50, 150, 210, 310, 350])
    bbp_dark = backscatter_dark_count(bbp, depth, percentile)
    # in this case we just want to check if none of the values is negative!
    assert np.all(bbp_dark >= 0)


def test_backscatter_dark_count_warning():
    from glidertools.optics import backscatter_dark_count

    # create some synthetic data
    percentile = 50
    bbp = np.array([0.002, 0.0006, 0.005, 0.005])
    depth = np.array(
        [50, 60, 70, 110]
    )  # this will trigger the warning  (no values between 200 and 400m)
    with pytest.warns(
        UserWarning
    ):  # this line will fail if the command below does not actually raise a warning!
        backscatter_dark_count(bbp, depth, percentile)


@pytest.mark.parametrize("percentile", [5, 50, 95])
def test_flr_dark_count(percentile):
    from glidertools.optics import fluorescence_dark_count

    # create some synthetic data
    flr = np.array([200.0, 100.0, 52.0, 52.0])
    depth = np.array([20, 50, 310, 350])
    # select only depths between 200 and 400
    mask = (depth > 300) & (depth < 400)
    # expected output
    expected_flr_dark = flr - np.nanpercentile(flr[mask], percentile)
    flr_dark = fluorescence_dark_count(flr, depth, percentile)
    np.testing.assert_allclose(expected_flr_dark, flr_dark)


@pytest.mark.parametrize("percentile", [5, 50, 95])
def test_flr_dark_count_negative(percentile):
    from glidertools.optics import fluorescence_dark_count

    # create some synthetic data
    flr = np.array([200.0, 100.0, 152.0, 151.0])
    # this will result in negative values that should be zeroed out
    depth = np.array([20, 50, 310, 350])
    flr_dark = fluorescence_dark_count(flr, depth, percentile)
    # in this case we just want to check if none of the values is negative!
    assert np.all(flr_dark >= 0)


def test_flr_dark_count_warning():
    from glidertools.optics import fluorescence_dark_count

    # create some synthetic data
    percentile = 50
    flr = np.array([200.0, 100.0, 52.0, 52.0])
    depth = np.array([20, 50, 210, 250])

    with pytest.warns(
        UserWarning
    ):  # this line will fail if the command below does not actually raise a warning!
        fluorescence_dark_count(flr, depth, percentile)


@pytest.mark.parametrize("percentile", [90])
def test_par_dark_count(percentile):
    from pandas import date_range

    from glidertools.optics import par_dark_count

    # create some synthetic data
    par = np.array([34, 23.0, 0.89, 0.89])
    depth = np.array([10, 20, 310, 350])
    time = date_range("2018-12-01 10:00", "2018-12-03 00:00", 4)
    # expected output
    expected_par_dark = par - np.nanmedian(
        np.nanpercentile(par[-1], percentile)
    )  # only use values in the 90% percentile of depths and between 23:00 and 01:00
    par_dark = par_dark_count(par, depth, time, percentile)
    np.testing.assert_allclose(expected_par_dark, par_dark)


def test_par_dark_count_warning():
    from pandas import date_range

    from glidertools.optics import par_dark_count

    # create some synthetic data
    percentile = 90
    par = np.array([34, 23.0, 0.89, 0.89])
    depth = np.array([10, 20, 310, 350])
    time = date_range("2018-12-01 10:00", "2018-12-03 20:00", 4)
    # this will trigger the warning  (no values between 200 and 400m)
    with pytest.warns(
        UserWarning
    ):  # this line will fail if the command below does not actually raise a warning!
        par_dark_count(par, depth, time, percentile)
