#!/usr/bin/env python
from __future__ import absolute_import as _ai
from __future__ import print_function as _pf
from __future__ import unicode_literals as _ul

from inspect import currentframe as getframe

from .helpers import transfer_nc_attrs


def time_average_per_dive(dives, time):
    """
    Gets the average time stamp per dive. This is used to create psuedo
    discrete time steps per dive for plotting data (using time as x-axis
    variable).

    Parameters
    ----------
    dives : np.array, dtype=float, shape=[n, ]
        discrete dive numbers (down = d.0; up = d.5) that matches time length
    time : np.array, dtype=datetime64, shape=[n, ]
        time stamp for each observed measurement

    Returns
    -------
    time_average_per_dive : np.array, dtype=datetime64, shape=[n, ]
        each dive will have the average time stamp of that dive. Can be used
        for plotting where time_average_per_dive is set as the x-axis.
    """
    from numpy import array, datetime64, nanmean
    from pandas import Series

    atime = array(time)
    dives = array(dives)
    if isinstance(atime[0], datetime64):
        t = atime.astype("datetime64[s]").astype(float)
    else:
        t = atime

    t_grp = Series(t).groupby(dives)
    t_mid = nanmean([t_grp.max(), t_grp.min()], axis=0)
    t_ser = Series(t_mid, index=t_grp.mean().index.values)
    diveavg = t_ser.reindex(index=dives).values
    diveavg = diveavg.astype("datetime64[s]")

    diveavg = transfer_nc_attrs(getframe(), time, diveavg, "_diveavg")

    return diveavg


def mask_to_depth_array(dives, depth, var):
    """
    Use when function returns a boolean section (as a mask) and you would
    like to return the depth of the positive mask (True) for each dive. This is
    useful for cases like MLD which returns a mask. Note that this is for
    ungridded data in "series" format.

    Parameters
    ----------
    dives : np.array, dtype=float, shape=[n, ]
        discrete dive numbers (down = d.0; up = d.5) that matches depth and var
        length
    depth : np.array, dtype=float, shape=[n, ]
        depth of each measurement
    var : np.array, dtype=bool, shape=[n,]
        mask array

    """

    from numpy import array, diff, r_
    from pandas import Series

    i = r_[False, diff(var)].astype(bool)
    idx_depth = Series(array(depth)[i], index=array(dives)[i])

    return idx_depth


def merge_dimensions(df1, df2, interp_lim=3):
    """
    Merges variables measured at different time intervals. Glider data may be
    sampled at different time intervals, as is the case for primary CTD and
    SciCon data.

    Parameters
    ----------
    df1 : pandas.DataFrame
        A dataframe indexed by datetime64 sampling times. Can have multiple
        columns. The index of this first dataframe will be preserved.
    df2 : pandas.DataFrame
        A dataframe indexed by datetime64 sampling times. Can have multiple
        columns. This second dataframe will be interpolated linearly onto the
        first dataframe.

    Returns
    -------
    merged_df : pandas.DataFrame
        The combined arrays interpolated onto the index of the first axis

    Raises
    ------
    Userwarning
        If either one of the indicies are not datetime64 dtypes

    Example
    -------
    You can use the following code and alter it if you want more control

    >>> df = pd.concat([df1, df2], sort=True, join='outer')  # doctest: +SKIP
    >>> df = (df
              .sort_index()
              .interpolate(limit=interp_lim)
              .bfill(limit=interp_lim)
              .loc[df1.index]
        )
    """

    import numpy as np
    import xarray as xr

    from .helpers import GliderToolsError

    is_xds = isinstance(df1, xr.Dataset) | isinstance(df2, xr.Dataset)

    if is_xds:
        msg = "One of your input objects is xr.Dataset, please define "
        raise GliderToolsError(msg)

    same_type = type(df1.index) == type(df2.index)  # noqa: E721
    # turning datetime64[ns] to int64 first,
    # because interpolate doesn't work on datetime-objects

    if same_type:
        df = df1.join(df2, sort=True, how="outer", rsuffix="_drop")
        df.index = df1.index.astype(np.int64)
        keys = df.select_dtypes(include=["datetime64[ns]"]).keys()
        for key in keys:
            df[key] = df[key].astype(np.int64)
        df = df.interpolate(limit=interp_lim).bfill(limit=interp_lim)
        df.index = df1.index.astype("datetime64[ns]")
        for key in keys:
            df[key] = df[key].astype("datetime64[ns]")
        return df.loc[df1.index.astype("datetime64[ns]")]
    else:
        raise UserWarning("Both dataframe indicies need to be same dtype")


def calc_glider_vert_velocity(time, depth):
    """
    Calculate glider vertical velocity in cm/s

    Parameters
    ----------
    time : np.array [datetime64]
        glider time dimension
    depth : np.array [float]
        depth (m) or pressure (dbar) if depth not avail

    Returns
    -------
    velocity : np.array
        vertical velocity in cm/s
    """
    from numpy import array
    from pandas import Series

    # Converting time from datetime 64 to seconds since deployment
    t_ns = array(time).astype("datetime64[ns]").astype(float)
    t_s = Series((t_ns - t_ns.min()) / 1e9)

    # converting pressure from dbar/m to cm
    p_m = array(depth).astype(float)
    p_cm = Series(p_m * 100)

    # velocity in cm/s
    velocity = p_cm.diff() / t_s.diff()

    return velocity


def calc_dive_phase(time, depth):
    """
    Determine the glider dive phase

    Parameters
    ----------
    time : np.array [datetime64]
        glider time dimension
    depth : np.array [float]
        depth (m) or pressure (dbar) if depth not avail

    Returns
    -------
    phase : np.array [int]
        phase according to the EGO dive phases
    """
    from numpy import array, isnan, ndarray

    time = array(time)
    depth = array(depth)

    velocity = calc_glider_vert_velocity(time, depth)  # cm/s

    phase = ndarray(time.size)

    phase[velocity > 0.5] = 1  # down dive
    phase[velocity < -0.5] = 4  # up dive
    phase[(depth > 200) & (velocity >= -0.5) & (velocity <= 0.5)] = 3  # inflexion
    phase[depth <= 200] = 0  # surface drift
    phase[isnan(phase)] = 6
    phase = phase.astype(int)

    return phase


def calc_dive_number(time, depth):
    """
    Determine the glider dive number (based on dive phase)

    Parameters
    ----------
    time : np.array [datetime64]
        glider time dimension
    depth : np.array [float]
        depth (m) or pressure (dbar) if depth not avail

    Returns
    -------
    dive_number : np.ndarray [float]
        the dive number where down dives are x.0 and up dives are x.5
    """

    phase = calc_dive_phase(time, depth)

    dive = dive_phase_to_number(phase)

    return dive


def dive_phase_to_number(phase):
    from pandas import Series

    phase = Series(phase)

    u_dive = ((phase == 4).astype(int).diff() == 1).astype(int).cumsum()
    d_dive = ((phase == 1).astype(int).diff() == 1).astype(int).cumsum()

    dive = (u_dive + d_dive) / 2

    return dive


def distance(lon, lat, ref_idx=None):
    """
    Great-circle distance in m between lon, lat points.

    Parameters
    ----------
    lon, lat : array-like, 1-D (size must match)
        Longitude, latitude, in degrees.
    ref_idx : None, int
        Defaults to None, which gives adjacent distances.
        If set to positive or negative integer, distances
        will be calculated from that point

    Returns
    -------
    distance : array-like
        distance in meters between adjacent points
        or distance from reference point

    """
    import numpy as np

    lon = np.array(lon)
    lat = np.array(lat)

    earth_radius = 6371e3

    if not lon.size == lat.size:
        raise ValueError(
            "lon, lat size must match; found %s, %s" % (lon.size, lat.size)
        )
    if not len(lon.shape) == 1:
        raise ValueError("lon, lat must be flat arrays")

    lon = np.radians(lon)
    lat = np.radians(lat)

    if ref_idx is None:
        i1 = slice(0, -1)
        i2 = slice(1, None)
        dlon = np.diff(lon)
        dlat = np.diff(lat)
    else:
        ref_idx = int(ref_idx)
        i1 = ref_idx
        i2 = slice(0, None)
        dlon = lon[ref_idx] - lon
        dlat = lat[ref_idx] - lat

    a = np.sin(dlat / 2) ** 2 + np.sin(dlon / 2) ** 2 * np.cos(lat[i1]) * np.cos(
        lat[i2]
    )

    angles = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = earth_radius * angles
    d = np.r_[0, distance]

    return d


if __name__ == "__main__":

    pass
