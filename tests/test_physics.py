import xarray as xr

from glidertools.load import seaglider_basestation_netCDFs
from glidertools.physics import (
    brunt_vaisala,
    mixed_layer_depth,
    potential_density,
    spice0,
)


filenames = "./tests/data/p542*.nc"

names = [
    "ctd_depth",
    "ctd_time",
    "ctd_pressure",
    "salinity",
    "temperature",
    "eng_wlbb2flvmt_Chlsig",
    "eng_wlbb2flvmt_wl470sig",
    "eng_wlbb2flvmt_wl700sig",
    "aanderaa4330_dissolved_oxygen",
    "eng_qsp_PARuV",
]

ds_dict = seaglider_basestation_netCDFs(
    filenames, names, return_merged=True, keep_global_attrs=False
)

merged = ds_dict["merged"]
if "time" in merged:
    merged = merged.drop_vars(["time", "time_dt64"])
dat = merged.rename(
    {
        "salinity": "salt_raw",
        "temperature": "temp_raw",
        "ctd_pressure": "pressure",
        "ctd_depth": "depth",
        "ctd_time_dt64": "time",
        "ctd_time": "time_raw",
        "eng_wlbb2flvmt_wl700sig": "bb700_raw",
        "eng_wlbb2flvmt_wl470sig": "bb470_raw",
        "eng_wlbb2flvmt_Chlsig": "flr_raw",
        "eng_qsp_PARuV": "par_raw",
        "aanderaa4330_dissolved_oxygen": "oxy_raw",
    }
)


def test_is_dataset():
    assert isinstance(dat, xr.core.dataset.Dataset)


def test_mixed_layer_depth():
    mld = mixed_layer_depth(dat.dives, dat.depth, dat.temp_raw)
    assert mld.min() > 10
    assert mld.max() < 40


def test_potential_density():
    pot_den = potential_density(
        dat.salt_raw, dat.temp_raw, dat.pressure, dat.latitude, dat.longitude
    )
    assert pot_den.min() > 1020
    assert pot_den.max() < 1040


def test_brunt_vaisala():
    brunt_val = brunt_vaisala(dat.salt_raw, dat.temp_raw, dat.pressure)
    assert brunt_val.min() > -0.002
    assert brunt_val.max() < 0.002


def test_spice0():
    spice = spice0(
        dat.salt_raw, dat.temp_raw, dat.pressure, dat.latitude, dat.longitude
    )
    assert spice.min() > -1
    assert spice.max() < 1
