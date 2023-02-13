#!/usr/bin/env python
from __future__ import absolute_import as _ai
from __future__ import print_function as _pf
from __future__ import unicode_literals as _ul

import numpy as np

from netCDF4 import Dataset

from ..helpers import GliderToolsWarning


# TODO: fix dives indexing (merge dim if same size as other more populated dim)
# TODO: when dims merge dives are sometimes taken from the wrong dataframe


def process_files(file_str):
    from glob import glob

    if isinstance(file_str, str):
        files = np.sort(glob(file_str))

    if len(files) < 1:
        raise FileNotFoundError("The provided string is not a file path")
    return files


def show_variables(files):
    from pandas import DataFrame

    files = process_files(files)

    i = len(files) // 2

    file = files[i]
    print("information is based on file: {}".format(file))

    variables = Dataset(file).variables
    info = {}
    for i, key in enumerate(variables):
        var = variables[key]
        info[i] = {
            "name": key,
            "dims": var.dimensions[0] if len(var.dimensions) == 1 else "string",
            "units": "" if not hasattr(var, "units") else var.units,
            "comment": "" if not hasattr(var, "comment") else var.comment,
        }

    vars = DataFrame(info).T

    dim = vars.dims
    dim[dim.str.startswith("str")] = "string"
    vars["dims"] = dim

    vars = (
        vars.sort_values(["dims", "name"])
        .reset_index(drop=True)
        .loc[:, ["dims", "name", "units", "comment"]]
        .set_index("name")
        .style
    )

    return vars


def check_var_in_ncfiles(files, key):

    is_in_files = []
    for file in files:
        vardict = Dataset(file).variables
        if key in vardict:
            is_in_files += (True,)
        else:
            is_in_files += (False,)

    return any(is_in_files)


def get_var_dim(files, key):
    dims = []
    for file in files:
        variables = Dataset(file).variables
        if key in variables:
            var = variables[key]
            dims += var.dimensions
    unique_dims = list(set(dims))
    if len(unique_dims) > 1:
        return False
    elif len(unique_dims) == 1:
        return unique_dims[0]
    else:
        return "string"


def get_var_units(files, key):
    from numpy import nanargmax, unique

    units = [get_var_attrs(file, key, "units") for file in files]
    units, counts = unique(units, return_counts=True)
    imax = nanargmax(counts)

    return units[imax]


def get_var_attrs(file, key, attr=None):
    vars = Dataset(file).variables
    if key not in vars:
        return
    var = Dataset(file).variables[key]

    if attr is None:
        return {k: var.getncattr(k) for k in var.ncattrs()}
    else:
        if hasattr(var, attr):
            return var.getncattr(attr)


def get_var_coords(files, key):
    """
    Finds the coordinates of the variable for the given netCDF files.

    Parameters
    ----------
    files : list
        a list of netCDF glider files
    key : str
        must be a variable in the netCDF

    Returns
    -------
    coords : list
        a list of coordinates from a subset of files
    """
    from numpy import concatenate

    coords = set([get_var_attrs(f, key, "coordinates") for f in files])
    if None in coords:
        coords.remove(None)
    coords = [c.split() for c in coords]
    if coords != []:
        coords = concatenate(coords).tolist()

    return coords


def get_dim_nobs(files, dim):
    for file in files:
        dimensions = Dataset(file).dimensions
        if dim in dimensions:
            return dimensions[dim].size


def get_dim_vars(files, dim):
    """
    Returns all the variable names that belong to a dimension
    """

    dim_vars = set()  # avoid duplication with a set
    for file in files:  # go through files to ensure all vars are included
        variables = Dataset(file).variables
        for key, var in variables.items():
            # test if the variable belongs to the dimension
            belongs_to_dim = any([dim in d for d in var.dimensions])
            if belongs_to_dim:
                dim_vars.update([key])
    # return a numpy array of the dimension variables (useful for indexing)
    return np.array(list(dim_vars))


def get_dim_same_size(files, dim):
    """
    Get dimension with the same size as the given dimension.
    If more than one is found, return the with most variables.
    """

    def sub_dim_same_size(file, dim):
        dimensions = Dataset(file).dimensions
        # make sure that the given dimension is in the file
        same_size = set()
        if dim in dimensions:
            n = dimensions[dim].size
            dimensions.pop(dim)

            for d in dimensions:
                if n == dimensions[d].size:
                    same_size.update([d])
        return list(same_size)

    # PART 1 get all dimensions with the same size
    same_size = set(sub_dim_same_size(files[0], dim))
    for file in files[1:]:
        same_size = same_size.intersection(sub_dim_same_size(file, dim))

    # if there is only one dimension of the same length return it
    return list(same_size)


def get_dim_coord(files, dim_name, coord_name, niter=0):
    # ensure time dim for each dimension for merging data
    # 1. search for 'coord' in for the same dimension
    # 2. search for coord in other dimension of same length

    dim_vars = get_dim_vars(files, dim_name)
    is_coord = [coord_name in key for key in dim_vars]
    same_size_dims = get_dim_same_size(files, dim_name)

    if any(is_coord) and (niter < 2):
        return dim_vars[is_coord][0]
    elif (same_size_dims != []) and (niter < 2):
        for d in same_size_dims:
            return get_dim_coord(files, d, coord_name, niter=niter + 1)
    else:
        return


def make_variable_dimension_dict(files, variable_names, n_check_files=3):
    import warnings

    step_size = len(files) // n_check_files
    step_size = 1 if step_size == 0 else step_size
    files_checklist = files[::step_size]

    dims = {}
    for key in variable_names:
        if not check_var_in_ncfiles(files_checklist, key):
            msg = key + " was not found in the files"
            warnings.warn(msg, GliderToolsWarning)
            continue
        single_dim = get_var_dim(files_checklist, key)
        if not single_dim:
            continue
        else:
            dim = single_dim

        if dim not in dims:
            dims[dim] = set()

        dims[dim].update([key])
        dims[dim].update(get_var_coords(files_checklist, key))

    # get compulsory time and depth variables (if present)
    for d in dims:
        dim = dims[d]
        if d == "string":
            continue
        has_coord = any(["time" in v for v in dim])

        if not has_coord:
            coord = get_dim_coord(files_checklist, d, "time")
            if coord:
                dims[d].update([coord])
            else:
                msg = "Could not find a time coordinate for dim: {}".format(d)
                warnings.warn(msg, GliderToolsWarning)
    return dims


def read_nc_files_divevars(files, keys, verbose=True, return_skipped=False):
    from os import path

    from numpy.ma import row_stack
    from pandas import DataFrame, concat

    if not verbose:
        from numpy import arange as trange
    else:
        from tqdm import trange

    data = []
    error = ""
    skipped_files = []
    progress_bar = trange(len(files))
    d = 0
    for i in progress_bar:
        fname = files[i]
        nc = Dataset(fname)

        d = nc.dive_number if hasattr(nc, "dive_number") else d + 1

        nc_keys = [k for k in filter(lambda k: k in nc.variables, keys)]
        if nc_keys:
            skipped = set(keys) - set(nc_keys)
            if skipped:
                error += "{} not in {}\n".format(str(skipped), path.split(fname)[1])
            arr = row_stack([nc.variables[k][:] for k in nc_keys])
            nc.close()

            df = DataFrame(arr.T, columns=nc_keys)
            df["dives"] = d
            data += (df,)
        else:
            skipped_files += (fname,)
            error += "{} was skipped\n".format(fname)

    if len(error) > 0:
        print(error)
    data = concat(data, ignore_index=True)

    if return_skipped:
        return data, skipped_files
    else:
        return data


def read_nc_files_strings(files, keys, verbose=True):
    from numpy import array, r_
    from pandas import DataFrame

    if not verbose:
        from numpy import arange as trange
    else:
        from tqdm import trange

    data = []
    idx = []
    d = 0
    for i in trange(files.size):
        fname = files[i]
        nc = Dataset(fname)
        d = nc.dive_number if hasattr(nc, "dive_number") else d + 1
        arr = r_[[nc.variables[k][:].squeeze() for k in keys]]
        nc.close()
        data += (arr,)
        idx += (d,)
    df = DataFrame(array(data, dtype=str), columns=keys)
    for col in df:
        df[col] = df[col].str.encode("ascii", "ignore").str.decode("ascii")
        try:
            df[col] = df[col].values.astype(float)
        except ValueError:
            pass
    df["dives"] = idx

    return df


def process_time(files, df):
    def decode_times_1970(series):
        # DECODING TIMES IF PRESENT
        t0 = np.datetime64("1970-01-01 00:00:00", "s")

        # realistic upper and lower limits since 1970
        tmin = np.datetime64("2000-01-01 00:00:00", "s")
        tmax = np.datetime64("2025-01-01 00:00:00", "s")
        lo_lim = (tmin - t0).astype(int)
        up_lim = (tmax - t0).astype(int)

        series_masked = series[series.notnull()]
        since1970 = ((series_masked > lo_lim) & (series_masked < up_lim)).all()

        if since1970:
            dt = series.values.astype("timedelta64[s]")
            return (t0 + dt).astype("datetime64[ns]")

    time_cols = df.columns[["time" in col for col in df]].values.tolist()
    if isinstance(files, str):
        file = [files]
    else:
        file = [files[len(files) // 2]]

    if len(time_cols) > 0:
        for col in time_cols:
            units = get_var_units(file, col)
            if units.startswith("seconds since 1970"):
                df[col + "_dt64"] = decode_times_1970(df[col])
                df = df.set_index(col + "_dt64", drop=False)
    return df


def process_dives(df):
    def get_dives(time, depth, dives=None):
        from ..utils import calc_dive_number

        if dives is None:
            return calc_dive_number(time, depth)
        else:
            # INDEX UP AND DOWN DIVES
            depth = np.array(depth)
            dives = np.array(dives)

            updive = np.ndarray(dives.size, dtype=bool) * False
            for d in np.unique(dives):
                i = d == dives
                j = np.argmax(depth[i])
                # bool slice of the dive
                k = i[i]
                # make False until the maximum depth
                k[:j] = False
                # assign the bool slice to the updive
                updive[i] = k

            dives = dives + (updive / 2)
            return dives

    depth_cols = df.columns[["depth" in col for col in df]].values.tolist()
    time_cols = df.columns[["time" in col for col in df]].values.tolist()
    if (len(depth_cols) > 0) & ("dives" in df):
        depth = df[depth_cols[0]]
        time = df[time_cols[0]]
        df["dives"] = get_dives(time, depth, df.dives)

    return df


def load_multiple_vars(
    files,
    variable_names,
    return_merged=False,
    verbose=True,
    keep_global_attrs=False,
    netcdf_attrs={},
    keep_variable_attrs=True,
):
    """
    Load a list of variables from the SeaGlider object as a
    ``pandas.DataFrame``.

    Parameters
    ----------
    variable_names : list
        a list of strings representing the keys you would like to load.

    Returns
    -------
    pandas.DataFrame
        Will always have coordinate dimensions loaded (even if not
        specified). These can then be accessed either by the variable
        objects or by .data[<dimension_name>].

    Note
    ----
        Using this method resets all previously loaded and stored data (data
        is stored under ``SeaGlider.data={dim: pandas.DataFrame}``).
        This is done to avoid erroneous coordinate matchup with sometimes
        missing data).
    """
    import time

    from pandas import DataFrame, to_numeric

    from ..utils import merge_dimensions

    # create a dictionary with dims as keys and variables as keys
    files = process_files(files)

    dims_dict = make_variable_dimension_dict(files, variable_names)
    data = {dim_name: DataFrame() for dim_name in dims_dict}
    merge_list = []  # list of mergable dataframes with longest at the front
    max_len = 0

    # LOADING EACH DIMENSION
    for dim_name, var_names in dims_dict.items():

        print("\nDIMENSION: {}\n{}".format(dim_name, str(var_names)).replace("'", ""))
        time.sleep(0.2)  # to prevent progress bar interruption
        skipped_files = []
        if dim_name == "string":
            df = read_nc_files_strings(files, var_names, verbose)
        else:
            df, skipped_files = read_nc_files_divevars(
                files, var_names, verbose, return_skipped=True
            )
        for col in df:
            df[col] = to_numeric(df[col], errors="coerce")

        # converting times that have 'seconds since 1970' units
        dim_files = list(set(files.tolist()) - set(skipped_files))
        df = process_time(dim_files, df)
        # splitting up and down if dives present otherwise calc from depth
        df = process_dives(df)

        # to make the merge list (with time idx) and longest index at the front
        if np.issubdtype(df.index.dtype, np.datetime64):
            if len(df) > max_len:
                merge_list.insert(0, dim_name)
                max_len = len(df)
            else:
                merge_list.append(dim_name)

        # adding columns to dimension based dataframes one by one
        for col in df:
            col = str(col)
            data[dim_name][col] = df[col]

    # MERGING DATA IF POSSIBLE
    can_merge = len(merge_list) > 1
    if return_merged and can_merge:
        print(
            "\nMerging dimensions on time indicies: {}, ".format(merge_list[0]),
            end="",
        )
        df_merged = data[merge_list.pop(0)]
        for other in merge_list:
            if "dives" in data[other]:
                df_other = data[other].drop(columns="dives")
            else:
                df_other = data[other]
            print(other, end=", ")
            df_merged = merge_dimensions(df_merged, df_other, interp_lim=1)
        data["merged"] = df_merged
        drop_names = list(data["merged"].filter(regex="_drop").columns)
        data["merged"] = data["merged"].drop(columns=drop_names)

    elif return_merged and (not can_merge):
        print(
            "\nCannot merge data - not enough time indexed DataFrames"
            "\nReturning unmerged dataframes"
        )

    # MAKING NETCDFS
    for key in data:
        data[key] = make_xr_dataset(
            data[key],
            files,
            keep_global_attrs=keep_global_attrs,
            keep_variable_attrs=keep_variable_attrs,
            index_name=key,
            attrs=netcdf_attrs,
        )
        if "dives" in data:
            data = data.set_coords("dives")

    return data


def make_xr_dataset(
    df,
    files,
    index_name="index",
    attrs={},
    keep_variable_attrs=True,
    keep_global_attrs=False,
):
    import re

    from pandas import Timestamp
    from xarray import open_dataset

    first = list(open_dataset(files[0]).attrs.items())
    final = list(open_dataset(files[-1]).attrs.items())

    if keep_global_attrs:
        global_attrs = dict(list(set(first).intersection(final)))
    else:
        global_attrs = {}

    lons = df.filter(regex=re.compile("lon", re.IGNORECASE))
    lats = df.filter(regex=re.compile("lat", re.IGNORECASE))
    depths = df.filter(regex=re.compile("depth", re.IGNORECASE))
    times = df.filter(regex=re.compile("time_dt64", re.IGNORECASE))
    dives = df.filter(regex=re.compile("dive", re.IGNORECASE))

    now = str(Timestamp("today"))[:19]
    history = (
        "[{}] imported data with GliderTools.load.seaglider_" "basestation_netCDFs;\n"
    ).format(now)

    global_attrs.update(attrs)
    global_attrs.update(
        {
            "date_created": now,
            "number_of_dives": dives.max().max() // 1,
            "files": str([f.split("/")[-1] for f in files]),
            "time_coverage_start": str(times.min().min()),
            "time_coverage_end": str(times.max().max()),
            "geospatial_vertical_min": depths.min().min(),
            "geospatial_vertical_max": depths.max().max(),
            "geospatial_lat_min": lats.min().min(),
            "geospatial_lat_max": lats.max().max(),
            "geospatial_lon_min": lons.min().min(),
            "geospatial_lon_max": lons.max().max(),
            "processing": history,
        }
    )

    coords = set()
    for key in df:
        check_files = files[[0, files.size // 2, -1]]
        coords.update(get_var_coords(check_files, key))
    coords = list(coords)

    for i, coord in enumerate(coords):
        if "time" in coord:
            coords[i] = coord + "_dt64"

    xds = (
        df.to_xarray()
        .drop_indexes(df.index.name)
        .reset_coords()
        .set_coords(coords)
        .rename_dims({df.index.name: index_name})
        .assign_attrs(global_attrs)
    )

    if keep_variable_attrs:
        mid = len(files) // 2
        for key in xds.variables:
            attrs = get_var_attrs(files[mid], key)
            if attrs is not None:
                attrs.pop("coordinates", None)
                xds[key].attrs = attrs

    return xds
