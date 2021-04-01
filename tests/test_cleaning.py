from glidertools.cleaning import horizontal_diff_outliers, outlier_bounds_iqr
from glidertools.load import seaglider_basestation_netCDFs


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

sg542 = seaglider_basestation_netCDFs(
    filenames, names, return_merged=True, keep_global_attrs=False
)

sg542_dat = sg542["sg_data_point"]


def test_outlier_bounds():
    # does not test for soft bugs
    salt = sg542_dat["salinity"]
    outlier_bounds_iqr(salt, multiplier=1.5)


def test_horizontal_outliers():
    # does not test for soft bugs
    horizontal_diff_outliers(
        sg542_dat["dives"],
        sg542_dat["ctd_depth"],
        sg542_dat["salinity"],
        multiplier=3,
        depth_threshold=400,
        mask_frac=0.1,
    )
