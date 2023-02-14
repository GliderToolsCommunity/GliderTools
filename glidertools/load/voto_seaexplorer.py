#!/usr/bin/env python
import numpy as np
import xarray as xr


def voto_seaexplorer_nc(filename):
    """
    Load .nc file downloaded from https://observations.voiceoftheocean.org/.
    A dives column is generated on importing the data.

    Parameters
    ----------
    filename : str
        path of .nc file.

    Returns
    -------
    xarray.Dataset
        Dataset containing the all columns in the source file and dives column
    """
    ds = xr.open_dataset(filename)
    ds = voto_seaexplorer_dataset(ds)
    return ds


def voto_seaexplorer_dataset(ds):
    """
    Adapts a VOTO xarray dataset, for example downloaded from the VOTO ERDAP
    server (https://erddap.observations.voiceoftheocean.org/erddap/index.html)
    to be used in GliderTools

    Parameters
    ----------
    ds : xarray.Dataset

    Returns
    -------
    xarray.Dataset
        Dataset containing the all columns in the source file and dives column
    """
    ds = add_dive_column(ds)
    return ds


def add_dive_column(ds):
    """add dive column to dataset

    Parameters:
    -----------
    ds: xarray.Dataset

    Returns:
    --------
    xarray.Dataset
        Dataset containing a dives column
    """
    ds["dives"] = (
        ["time"],
        np.where(ds.profile_direction == 1, ds.profile_num, ds.profile_num + 0.5),
    )
    return ds
