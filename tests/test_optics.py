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