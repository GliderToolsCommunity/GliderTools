#!/usr/bin/env python
from __future__ import absolute_import as _ai
from __future__ import print_function as _pf
from __future__ import unicode_literals as _ul


def slocum_geomar_matfile(filename, verbose=True):
    """
    Load .mat file generated with the geomar MATLAB script for Slocum data.

    A dive column is generated on importing the data. When single values per
    dive (e.g. u/v), the value is set for the entire dive.

    Parameters
    ----------
    filename : str
        path of .mat file.
    verbose : bool, optional
        defaults to True

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the all columns in the `.mat` file
    """

    import numpy as np
    import pandas as pd

    from scipy.io import loadmat

    mat = loadmat(filename)

    df = pd.DataFrame()

    if verbose:
        print("Loading variables: \n\t[", end="")
    for key in mat.keys():
        if key.startswith("_"):
            continue

        if verbose:
            print(" " + key, end=",")
        var = mat[key]
        col, dives = [], []
        for i, dat in enumerate(var.squeeze()):
            col += (dat.squeeze(),)
            dives += (np.ones(dat.squeeze().size) * i,)

        try:
            df[key] = np.concatenate(col)
            df["dives"] = np.concatenate(dives)
        except ValueError:
            ser = pd.Series(col, index=np.array(dives).squeeze())
            df[key] = ser.reindex(df.dives).values

    df["dives"] /= 2.0
    if "time_datenum" in df.columns:
        df["time"] = convert_matlab_datenum_to_datetime64(df.time_datenum)

    print("]")
    return df


def convert_matlab_datenum_to_datetime64(datenum):
    from numpy import datetime64, timedelta64

    time_epoch = datetime64("1970-01-01 00:00:00.000")
    time_matlab = timedelta64(-367, "D")
    time_ordinal = datetime64("0001-01-01 00:00:00", "D").astype("timedelta64")
    time_measurements = (datenum * 86400).astype("timedelta64[s]")

    datetime = (time_epoch + time_matlab) + (time_ordinal + time_measurements)

    return datetime
