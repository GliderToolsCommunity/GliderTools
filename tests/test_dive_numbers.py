import pytest

import glidertools.utils as gt_util

from glidertools.load import seaglider_basestation_netCDFs


# load some data
filenames = "./tests/data/p542*.nc"

names = ["ctd_depth", "ctd_time"]
ds_dict = seaglider_basestation_netCDFs(filenames, names, keep_global_attrs=False)

dat = ds_dict["sg_data_point"]
depth = dat["ctd_depth"]
time = dat["ctd_time"]


def test_find_correct_number_dives():
    # using default values
    dives = gt_util.calc_dive_number(depth, time)
    assert dives.max() == 599.5
