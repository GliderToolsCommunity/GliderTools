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
    lat = -35, 35, 80
    lon = 0, 0, 0
    sunrise, sunset = gt.optics.sunset_sunrise(time, lat, lon)

    # Two entries, there should be two outputs
    assert len(sunrise) == len(lat)

    # sunrise will be earlier for the SH in January
    assert sunrise[0] < sunrise[1]
