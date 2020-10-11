def test_import():
    import glidertools

    print(glidertools)


def test_import_data_seaglider():
    import glidertools as gt

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

    ds_dict = gt.load.seaglider_basestation_netCDFs(
        filenames, names, return_merged=True, keep_global_attrs=False
    )

    assert isinstance(ds_dict, dict)
