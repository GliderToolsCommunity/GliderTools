import warnings

import glidertools.plot as gt_plt

from glidertools.load import seaglider_basestation_netCDFs


# load some data
filenames = "./tests/data/p542*.nc"

names = ["ctd_depth", "ctd_time", "ctd_pressure", "salinity", "temperature"]
ds_dict = seaglider_basestation_netCDFs(filenames, names, keep_global_attrs=False)

dat = ds_dict["sg_data_point"]


def test_no_warns():
    """Check gt_plt() raises no warnings in pcolormesh."""
    with warnings.catch_warnings() as record:
        gt_plt(dat.dives, dat.ctd_pressure, dat.salinity)

    # print warnings that were captured
    if record:
        print("Warnings were raised: " + ", ".join([str(w) for w in record]))

        # Check the warning messages for statements we do not want to see
        fail_message = (
            "shading='flat' when X and Y have the same dimensions as C is deprecated"
        )
        assert not any([fail_message in str(r) for r in record])
