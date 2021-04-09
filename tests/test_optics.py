import pytest


def test_sunrise_sunset():
    """
    Tests if sunrise/sunset:
        1. can run
        2. output is the right shape
        3. if the output is correct-ish
    """
    import numpy as np

    import glidertools as gt

    time = [
        np.datetime64("2000-01-01"),
        np.datetime64("2000-01-02"),
        np.datetime64("2000-01-03"),
    ]
    lat = -35, 35, 45
    lon = 0, 0, 0
    sunrise, sunset = gt.optics.sunset_sunrise(time, lat, lon)

    # Two entries, there should be two outputs
    assert len(sunrise) == len(lat)

    # sunrise will be earlier for the SH in January
    assert sunrise[0] < sunrise[1]


def test_sunrise_sunset_fail():
    """
    This is a test to make us aware that the astropy will fail if
    the latitude is beyond where the sun sets or rises.
    Perhaps we should add a length of day catch?
    """
    import numpy as np

    import glidertools as gt

    time = [
        np.datetime64("2000-12-21"),
        np.datetime64("2000-06-21"),
    ]
    lat = (
        -80,
        80,
    )
    lon = (
        0,
        0,
    )

    with pytest.raises(ValueError):
        sunrise, sunset = gt.optics.sunset_sunrise(time, lat, lon)


@pytest.mark.parametrize("percentile", [95, 85, 50])
def test_backscatter_dark_count(percentile):
    # create some synthetic data
    bbp = np.array([0.002, 0.0006, 0.0005, 0.0005, 0.0005])
    depth = np.array([50, 150, 210, 310, 350])
    # select only depths between 200 and 400
    mask = (depth > 200) & (depth < 400)
    # expected output
    expected_bbp_dark = bbp - np.nanpercentile(bbp[mask], percentile)
    bbp_dark = gt.optics.backscatter_dark_count(bbp, depth, percentile)
    np.testing.assert_allclose(expected_bbp_dark, bbp_dark)


@pytest.mark.parametrize("percentile", [95, 75, 50])
def test_backscatter_dark_count_negative(percentile):
    # create some synthetic data
    bbp = np.array(
        [0.002, 0.0006, 0.005, 0.005, 0.0004]
    )  # this will result in negative values that should be zeroed out
    depth = np.array([50, 150, 210, 310, 350])
    mask = (depth > 200) & (depth < 400)

    bbp_dark = gt.optics.backscatter_dark_count(bbp, depth, percentile)
    # in this case we just want to check if none of the values is negative!
    assert np.all(bbp_dark >= 0)


@pytest.mark.parametrize("percentile", [95, 75, 50])
def test_flr_dark_count(percentile):
    # create some synthetic data
    flr = np.array([200.0, 100.0, 52.0, 52.0])
    depth = np.array([20, 50, 310, 350])
    # select only depths between 200 and 400
    mask = (depth > 300) & (depth < 400)
    # expected output
    expected_flr_dark = flr - np.nanpercentile(flr[mask], percentile)
    flr_dark = gt.optics.fluorescence_dark_count(flr, depth, percentile)
    np.testing.assert_allclose(expected_flr_dark, flr_dark)


@pytest.mark.parametrize("percentile", [95, 75, 50])
def test_flr_dark_count_negative(percentile):
    # create some synthetic data
    flr = np.array(
        [200.0, 100.0, 152.0, 151.0]
    )  # this will result in negative values that should be zeroed out
    depth = np.array([20, 50, 310, 350])
    mask = (depth > 300) & (depth < 400)
    expected_flr_dark = flr - np.nanpercentile(flr[mask], percentile)
    flr_dark = gt.optics.fluorescence_dark_count(flr, depth, percentile)
    # in this case we just want to check if none of the values is negative!
    assert np.all(flr_dark >= 0)
