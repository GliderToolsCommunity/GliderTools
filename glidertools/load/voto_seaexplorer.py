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


def voto_concat_datasets(datasets):
    """
    Concatenates multiple datasets along the time dimensions, profile_num
    and dives variable(s) are adapted so that they start counting from one
    for the first dataset and monotonically increase.

    Parameters
    ----------
    datasets : list of xarray.Datasets

    Returns
    -------
    xarray.Dataset
        concatenated Dataset containing all the data from the list of datasets
    """
    # in case the datasets have a different set of variables, emtpy variables are created
    # to allow for concatenation (concat with different set of variables leads to error)
    mlist = [set(dataset.variables.keys()) for dataset in datasets]
    allvariables = set.union(*mlist)
    for dataset in datasets:
        missing_vars = allvariables - set(dataset.variables.keys())
        for missing_var in missing_vars:
            dataset[missing_var] = np.nan

    # renumber profiles, so that profile_num still is unique in concat-dataset
    for index in range(1, len(datasets)):
        datasets[index]["profile_num"] += (
            datasets[index - 1].copy()["profile_num"].max()
        )
    ds = xr.concat(datasets, dim="time")
    ds = add_dive_column(ds)

    return ds
