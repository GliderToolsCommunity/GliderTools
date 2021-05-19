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
    lat = -35, 75, 45
    lon = 0, 0, 0
    sunrise, sunset = sunset_sunrise(time, lat, lon)

    # Three entries, there should be three outputs
    assert len(sunrise) == len(lat)

    # sunrise will be earlier for the SH in January
    assert sunrise[0] < sunrise[1]

    # high latitude in NH winter should output polar night
    assert pd.to_datetime(sunrise[1]).hour == 11
