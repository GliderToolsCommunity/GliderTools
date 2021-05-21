import numpy as np
import pytest

from glidertools.processing import (
    calc_backscatter,
    calc_fluorescence,
    calc_oxygen,
    calc_par,
    calc_physics,
)
from tests.test_physics import dat


def test_calc_oxygen():
    o2ml, o2pc, o2au = calc_oxygen(
        dat.oxy_raw, dat.pressure, dat.salt_raw, dat.temp_raw
    )
    assert np.nanmin(o2ml) == pytest.approx(3.7152995, 0.0001)
    assert np.nanmax(o2ml) == pytest.approx(11.460690, 0.0001)
    assert np.nanmin(o2pc) == pytest.approx(49.677466, 0.01)
    assert np.nanmax(o2pc) == pytest.approx(182.91453, 0.01)
    assert np.nanmin(o2au) == pytest.approx(-5.195040, 0.0001)
    assert np.nanmax(o2au) == pytest.approx(3.7637133, 0.0001)
