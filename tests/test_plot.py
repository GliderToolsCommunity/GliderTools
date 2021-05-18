import pytest

import glidertools.plot as gt_plt

from glidertools.load import seaglider_basestation_netCDFs


# load some data
filenames = "./tests/data/p542*.nc"

names = ["ctd_depth", "ctd_time", "ctd_pressure", "salinity", "temperature"]
ds_dict = seaglider_basestation_netCDFs(filenames, names, keep_global_attrs=False)

dat = ds_dict["sg_data_point"]


def test_no_warns():
    """Check gt_plt() raises no warnings in pcolormesh."""
    with pytest.warns(None) as warnings:
        import warnings
        warnings.warn('Just Testing', UserWarning)
        gt_plt(dat.dives, dat.ctd_pressure, dat.salinity)
    
    if len(record) > 0:
        raise AssertionError(
                "Warnings were raised: " + ", ".join([str(w) for w in warnings])
            )
