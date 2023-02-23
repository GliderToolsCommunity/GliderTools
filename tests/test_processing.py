import numpy as np
import pytest

from glidertools.processing import (  # noqa
    calc_backscatter,
    calc_fluorescence,
    calc_oxygen,
    calc_par,
    calc_physics,
    oxygen_ml_per_l_to_umol_per_kg,
)
from tests.test_physics import dat


def test_calc_oxygen():
    dat.oxy_raw.values[dat.oxy_raw.values < 0] = np.nan
    dat.oxy_raw.values[dat.oxy_raw.values > 500] = np.nan
    o2ml, o2pc, o2aou = calc_oxygen(
        dat.oxy_raw,
        dat.pressure,
        dat.salt_raw,
        dat.temp_raw,
        dat.latitude,
        dat.longitude,
    )
    assert np.nanmean(o2ml) == pytest.approx(5.22, 0.001)
    assert np.nanmean(o2pc) == pytest.approx(75.857, 0.001)
    assert np.nanmean(o2aou) == pytest.approx(75.351, 0.001)
