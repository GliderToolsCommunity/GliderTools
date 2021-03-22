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

    import glidertools as gt

    time = pd.date_range('2018-01-01','2018-01-03')

    lat = -75, 75, 36
    lon = 0, 0, 0
    sunrise, sunset = gt.optics.sunset_sunrise(time, lat, lon)
    # Two entries, there should be two outputs
    assert len(sunrise) == len(lat)

    # sunrise will be earlier for the SH in January
    assert sunrise[0][0] < sunrise[1][0]


def test_sunrise_sunset_fail():
    """
    This is a test to make us aware that the astropy will fail if
    the latitude is beyond where the sun sets or rises.
    Perhaps we should add a length of day catch? edit (I.G) --> This is no longer applicable
    """
    import numpy as np

    import glidertools as gt
    time = pd.date_range('2018-01-01','2018-01-03')

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
