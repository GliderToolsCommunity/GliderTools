#!/usr/bin/env python
from __future__ import absolute_import as _ai
from __future__ import print_function as _pf
from __future__ import unicode_literals as _ul

from inspect import currentframe as getframe

from .helpers import transfer_nc_attrs


def find_bad_profiles(
    dives, depth, var, ref_depth=None, stdev_multiplier=1, method="median"
):
    """
    Find profiles that exceed a threshold at a reference depth.

    This function masks bad dives based on
        mean + std x [1] or
        median + std x [1] at a reference depth.
    Function is typically used to clean raw fluorescence and backscatter data.

    Parameters
    ----------
    dives: numpy.ndarray or pandas.Series
        The dive count (round is down dives, 0.5 is up dives).
    depth: numpy.ndarray or pandas.Series
        The depth array in metres.
    var: numpy.ndarray or pandas.Series
        Array of data variable for function to be performed on.
    ref_depth: int
        The depth threshold for optics.find_bad_profiles below which the
        median or mean is calculated for identifying outliers.
    stdev_multiplier: int
        The standard deviation multiplier for calculating outliers,
        i.e. mean +- std x [1].
    method: str
        Whether to use the deep median or deep mean to determine bad profiles
        for optics.find_bad_profiles.

    Returns
    -------

    bad_dive_idx: numpy.ndarray or pandas.Series
        The index of dives where the deep mean/median is greater than the limit
    bad_dive: mask
        If True, the dive deep mean/median is greater than the limit.

    """
    from numpy import array, c_
    from pandas import DataFrame

    stdev_multiplier = stdev_multiplier

    dives = array(dives)

    df = DataFrame(c_[depth, dives, var], columns=["depth", "dives", "dat"])

    if not ref_depth:
        # reference depth is found by finding the average maximum
        # depth of the variable. The max depth is multiplied by 3
        # this reference depth can be set
        ref_depth = df.depth[df.dat.groupby(df.dives).idxmax().values].mean() * 3

    # find the median below the reference depth
    deep_avg = df[df.depth > ref_depth].groupby("dives").dat.median()

    if method.startswith("med"):
        # if the deep_avg is larger than the median
        bad_dive = deep_avg > (deep_avg.median() + (deep_avg.std() * stdev_multiplier))
        bad_dive = bad_dive.index.values[bad_dive]
        bad_dive_idx = array([(dives == d) for d in bad_dive]).any(0)
        return bad_dive_idx, bad_dive
    else:
        # if the deep_avg is larger than the mean
        bad_dive = deep_avg > (deep_avg.mean() + (deep_avg.std() * stdev_multiplier))
        bad_dive = bad_dive.index.values[bad_dive]
        bad_dive_idx = array([(dives == d) for d in bad_dive]).any(0)
        return bad_dive_idx, bad_dive


def par_dark_count(par, depth, time, depth_percentile=90):
    """
    Calculates an in situ dark count from the PAR sensor.

    The in situ dark count for the PAR sensor is calculated from the median,
    selecting only observations in the nighttime and in the 90th percentile of
    the depth sampled (i.e. the deepest depths measured)

    Parameters
    ----------

    par: numpy.ndarray or pandas.Series
        The par array after factory calibration in units uE/m2/sec.
    depth: numpy.ndarray or pandas.Series
        The depth array in metres.
    time: numpy.ndarray or pandas.Series
        The date & time array in a numpy.datetime64 format.
    depth_percentile: int
        User defined percentile for minimum dark depth. Defaults to 90
        so that samples from deepest 10 % of profile are used in correction

    Returns
    -------

    par_dark: numpy.ndarray or pandas.Series
        The par data corrected for the in situ dark value in units uE/m2/sec.
    """
    import warnings

    from numpy import array, isnan, ma, nanmedian, nanpercentile

    par_arr = array(par)
    depth = array(depth)
    time = array(time)

    # DARK CORRECTION FOR PAR
    hrs = time.astype("datetime64[h]") - time.astype("datetime64[D]")
    xi = ma.masked_inside(hrs.astype(int), 21, 5)  # find hours between 22:00 and 3:00
    if ma.sum(xi) < 1:
        warnings.warn(
            "There are no reliable night time measurements. This dark count correction cannot be "
            "cannot be trusted",
            UserWarning,
        )

    yi = ma.masked_outside(
        depth, *nanpercentile(depth[~isnan(par_arr)], [depth_percentile, 100])
    )  # pctl of depth
    i = ~(xi.mask | yi.mask)
    dark = nanmedian(par_arr[i])
    par_dark = par_arr - dark
    par_dark[par_dark < 0] = 0

    return par_dark


def backscatter_dark_count(bbp, depth, percentile=5):
    """
    Calculates an in situ dark count from the backscatter sensor.

    The in situ dark count for the backscatter sensor is calculated from the
    user-defined percentile between 200 and 400m.

    Parameters
    ----------

    bbp: numpy.ndarray or pandas.Series
        The total backscatter array after factory calibration in m-1.
    depth: numpy.ndarray or pandas.Series
        The depth array in metres.

    Returns
    -------

    bbp: numpy.ndarray or pandas.Series
        The total backscatter data corrected for the in situ dark value.
    """
    import warnings

    from numpy import array, isnan, nanpercentile

    bbp_dark = array(bbp)
    mask = (depth > 200) & (depth < 400)
    if (~isnan(bbp[mask])).sum() == 0:
        warnings.warn(
            "There are no backscatter measurements between 200 "
            "and 400 metres.The dark count correction cannot be "
            "made and backscatter data can't be processed.",
            UserWarning,
        )

    dark_pctl = nanpercentile(bbp_dark[mask], percentile)
    bbp_dark -= dark_pctl
    bbp_dark[bbp_dark < 0] = 0

    bbp_dark = transfer_nc_attrs(getframe(), bbp, bbp_dark, "_dark")

    return bbp_dark


def fluorescence_dark_count(flr, depth, percentile=5):
    """
    Calculates an in situ dark count from the fluorescence sensor.

    The in situ dark count for the fluorescence sensor is calculated from the
    user-defined percentile between 300 and 400m.

    Parameters
    ----------

    flr: numpy.ndarray or pandas.Series
        The fluorescence array after factory calibration.
    depth: numpy.ndarray or pandas.Series
        The depth array in metres.

    Returns
    -------

    flr: numpy.ndarray or pandas.Series
        The fluorescence data corrected for the in situ dark value.

    """
    import warnings

    from numpy import array, isnan, nanpercentile

    mask = (depth > 300) & (depth < 400)
    flr_dark = array(flr)

    if (~isnan(flr_dark[mask])).sum() == 0:
        warnings.warn(
            "\nThere are no fluorescence measurements between "
            "300 and 400 metres.\nThe dark count correction "
            "cannot be made and fluorescence data can't be processed.",
            UserWarning,
        )
    dark_pctl = nanpercentile(flr_dark[mask], percentile)
    flr_dark -= dark_pctl
    flr_dark[flr_dark < 0] = 0

    flr_dark = transfer_nc_attrs(getframe(), flr, flr_dark, "_dark")

    return flr_dark


def par_scaling(par_uV, scale_factor_wet_uEm2s, sensor_output_mV):
    """
    Scaling correction for par with factory calibration coefficients.

    The function subtracts the sensor output from the raw counts and divides
    with the scale factor. The factory calibrations are unique for each
    deployment and should be taken from the calibration file for that
    deployment.

    Parameters
    ----------
    par_uV: numpy.ndarray or pandas.Series
        The raw par data with units uV.
    scale_factor_wet_uEm2s: float
        The scale factor from the factory calibration file in units uE/m2/sec.
    sensor_output_mV: float
        The sensor output in the dark from the factory calibration file in
        units mV.

    Returns
    par_uEm2s: numpy.ndarray or pandas.Series
        The par data corrected for the sensor output and scale factor from the
        factory calibration file in units uE/m2/sec.

    """
    sensor_output_uV = sensor_output_mV / 1000.0

    par_uEm2s = (par_uV - sensor_output_uV) / scale_factor_wet_uEm2s

    par_uEm2s = transfer_nc_attrs(getframe(), par_uV, par_uEm2s, "par_uEm2s")

    return par_uEm2s


def par_fill_surface(par, dives, depth, max_curve_depth=100):
    """
    Algebraically calculates the top 5 metres of the par profile.

    The function removes the top 5 metres of par data, and then using an
    exponential equation calculates the complete profile.

    Parameters
    ----------
    par: numpy.ndarray or pandas.Series
        The par data with units uE/m2/sec.
    dives: numpy.ndarray or pandas.Series
        The dive count (round is down dives, 0.5 is up dives).
    depth: numpy.ndarray or pandas.Series
        The depth array in metres.
    max_curve_depth: int
        The maximum depth of which to fit the exponential function.


    Returns
    -------
    par_filled: numpy.ndarray or pandas.Series
        The par data with the algebraically calculated top 5 metres.

    """
    import numpy as np

    from scipy.optimize import curve_fit

    def dive_par_fit(depth, par):
        def exp_func(x, a, b):
            return a * np.exp(b * x)

        xj, yj = depth, par
        mask = ~(np.isnan(xj) | np.isnan(yj)) & (xj < max_curve_depth)
        xm, ym = xj[mask], yj[mask]

        if all(ym == 0) | (mask.sum() <= 2):
            yj_hat = np.ones_like(depth) * np.nan
        else:
            try:
                [a, b], _ = curve_fit(exp_func, xm, ym, p0=(500, -0.03), maxfev=1000)
                yj_hat = exp_func(xj, a, b)
            except RuntimeError:
                yj_hat = np.ones_like(depth) * np.nan

        return yj_hat

    var = par.copy()
    par = np.array(par)
    dives = np.array(dives)
    depth = np.array(depth)

    par_filled = np.ones_like(depth) * np.nan
    for d in np.unique(dives):
        i = dives == d
        par_fit = dive_par_fit(depth[i], par[i])
        par_filled[i] = par_fit

    par_filled = transfer_nc_attrs(getframe(), var, par_filled, "par_expfill")

    return par_filled


def photic_depth(par, dives, depth, return_mask=False, ref_percentage=1):
    """
    Algebraically calculates the euphotic depth.

    The function calculates the euphotic depth and attenuation coefficient (Kd)
    based upon the linear fit of the natural log of par with depth.

    Parameters
    ----------
    par: numpy.ndarray or pandas.Series
        The par data with units uE/m2/sec.
    dives: numpy.ndarray or pandas.Series
        The dive count (round is down dives, 0.5 is up dives).
    depth: numpy.ndarray or pandas.Series
        The depth array in metres.
    return_mask: bool
        If True, will return a mask for the photic layer
        (depth < euphotic depth).
    ref_percentage: int
        The percentage light depth to calculate the euphotic layer, typically
        assumed to be 1% of surface par.

    Returns
    -------
    light_depths: numpy.ndarray
        An array of the euphotic depths in metres.
    slopes: numpy.ndarray
        An array of the par attenuation coefficient (Kd).
    """
    import numpy as np
    import pandas as pd

    from scipy.stats import linregress

    def dive_slope(par, depth):
        mask = ~(np.isnan(par) | np.isnan(depth))
        x, y = depth[mask], par[mask]

        y = np.log(y)
        slope = linregress(x, y).slope

        return slope

    # Precentage light depth
    def dive_light_depth(depth, slope):

        if np.isnan(slope):
            euph_depth = np.nan
        else:
            light_depth = np.exp((depth * -1) / (-1 / slope)) * 100.0
            ind = abs(light_depth - ref_percentage).argmin()
            euph_depth = depth[ind]

        if return_mask:
            return depth < euph_depth
        else:
            return [euph_depth]

    #########################################################
    assert np.array(par).any(), "PAR does not contain data"

    slopes = []
    light_depths = []
    udives = np.unique(dives)
    for d in udives:
        i = dives == d
        zj = np.array(par[i])
        yj = np.array(depth[i])
        # xj = np.array(dives[i])

        if all(np.isnan(zj)):
            slope = np.nan
        else:
            slope = dive_slope(zj, yj)
        light_depth = dive_light_depth(yj, slope)

        slopes += (slope,)
        light_depths += (light_depth,)

    slopes = pd.Series(slopes, index=udives)
    light_depths = np.concatenate(light_depths)

    if not return_mask:
        light_depths = pd.Series(light_depths, index=udives)

    return light_depths, slopes


def sunset_sunrise(time, lat, lon):
    """
    Calculates the local sunrise/sunset of the glider location.

    The function uses the Skyfield package to calculate the sunrise and sunset
    times using the date, latitude and longitude. The times are returned
    rather than day or night indices, as it is more flexible for the quenching
    correction.


    Parameters
    ----------
    time: numpy.ndarray or pandas.Series
        The date & time array in a numpy.datetime64 format.
    lat: numpy.ndarray or pandas.Series
        The latitude of the glider position.
    lon: numpy.ndarray or pandas.Series
        The longitude of the glider position.

    Returns
    -------
    sunrise: numpy.ndarray
        An array of the sunrise times.
    sunset: numpy.ndarray
        An array of the sunset times.

    """
    import numpy as np
    import pandas as pd

    from pandas import DataFrame
    from skyfield import api

    ts = api.load.timescale()
    eph = api.load("de421.bsp")
    from skyfield import almanac

    df = DataFrame.from_dict(dict([("time", time), ("lat", lat), ("lon", lon)]))

    # set days as index
    df = df.set_index(df.time.values.astype("datetime64[D]"))

    # groupby days and find sunrise for unique days
    # groupby days and find sunrise/sunset for unique days
    grp_avg = df.groupby(df.index).mean(numeric_only=False)
    date = grp_avg.index.to_pydatetime()
    date = grp_avg.index

    time_utc = ts.utc(date.year, date.month, date.day, date.hour)
    time_utc_offset = ts.utc(
        date.year, date.month, date.day + 1, date.hour
    )  # add one day for each unique day to compute sunrise and sunset pairs

    bluffton = []
    for i in range(len(grp_avg.lat)):
        bluffton.append(api.wgs84.latlon(grp_avg.lat[i], grp_avg.lon[i]))
    bluffton = np.array(bluffton)

    sunrise = []
    sunset = []
    for n in range(len(bluffton)):

        f = almanac.sunrise_sunset(eph, bluffton[n])
        t, y = almanac.find_discrete(time_utc[n], time_utc_offset[n], f)

        if not t:
            if f(time_utc[n]):  # polar day
                sunrise.append(
                    pd.Timestamp(
                        date[n].year, date[n].month, date[n].day, 0, 1
                    ).to_datetime64()
                )
                sunset.append(
                    pd.Timestamp(
                        date[n].year, date[n].month, date[n].day, 23, 59
                    ).to_datetime64()
                )
            else:  # polar night
                sunrise.append(
                    pd.Timestamp(
                        date[n].year, date[n].month, date[n].day, 11, 59
                    ).to_datetime64()
                )
                sunset.append(
                    pd.Timestamp(
                        date[n].year, date[n].month, date[n].day, 12, 1
                    ).to_datetime64()
                )

        else:
            sr = t[y == 1]  # y=1 sunrise
            sn = t[y == 0]  # y=0 sunset

            sunup = pd.to_datetime(sr.utc_iso()).tz_localize(None)
            sundown = pd.to_datetime(sn.utc_iso()).tz_localize(None)

            # this doesnt look very efficient at the moment, but I was having issues with getting the datetime64
            # to be compatible with the above code to handle polar day and polar night

            su = pd.Timestamp(
                sunup.year[0],
                sunup.month[0],
                sunup.day[0],
                sunup.hour[0],
                sunup.minute[0],
            ).to_datetime64()

            sd = pd.Timestamp(
                sundown.year[0],
                sundown.month[0],
                sundown.day[0],
                sundown.hour[0],
                sundown.minute[0],
            ).to_datetime64()

            sunrise.append(su)
            sunset.append(sd)

    sunrise = np.array(sunrise).squeeze()
    sunset = np.array(sunset).squeeze()

    grp_avg["sunrise"] = sunrise
    grp_avg["sunset"] = sunset

    # reindex days to original dataframe as night
    df_reidx = grp_avg.reindex(df.index)
    sunrise, sunset = df_reidx[["sunrise", "sunset"]].values.T

    return sunrise, sunset


def quenching_correction(
    flr,
    bbp,
    dives,
    depth,
    time,
    lat,
    lon,
    max_photic_depth=100,
    night_day_group=True,
    surface_layer=5,
    sunrise_sunset_offset=1,
):
    """
    Corrects the fluorescence data based upon Thomalla et al. (2017).

    The function calculates the quenching depth and performs the quenching
    correction based on the fluorescence to backscatter ratio. The quenching
    depth is calculated based upon the different between night and daytime
    fluorescence. The default setting is for the preceding night to be used to
    correct the following day's quenching (`night_day_group=True`). This can
    be changed so that the following night is used to correct the preceding
    day. The quenching depth is then found from the difference between the
    night and daytime fluorescence, using the steepest gradient of the {5
    minimum differences and the points the difference changes sign (+ve/-ve)}.
    The function gets the backscatter/fluorescence ratio between from the
    quenching depth to the surface, and then calculates a mean nighttime
    ratio for each night. The quenching ratio is calculated from the nighttime
    ratio and the daytime ratio, which is then applied to fluorescence to
    correct for quenching. If the corrected value is less than raw, then the
    function will return the original raw data.

    Parameters
    ----------
    flr: numpy.ndarray or pandas.Series
        fluorescence data after cleaning and factory calibration conversion
    bbp: numpy.ndarray or pandas.Series
        Total backscatter after cleaning and factory calibration conversion
    dives: numpy.ndarray or pandas.Series
        The dive count (round is down dives, 0.5 is up dives).
    depth: numpy.ndarray or pandas.Series
        The depth array in metres.
    time: numpy.ndarray or pandas.Series
        The date & time array in a numpy.datetime64 format.
    lat: numpy.ndarray or pandas.Series
        The latitude of the glider position.
    lon: numpy.ndarray or pandas.Series
        The longitude of the glider position.
    max_photic_depth: int
        Limit the quenching correction to depth less than a given value [100].
    night_day_group: bool
        If True, use preceding night otherwise use following night for
        calculating the flr:bbp ratio.
    surface_layer: int
        The surface depth that is omitted from the correction calculations
        (metres)
    sunrise_sunset_offset: int
        The delayed onset and recovery of quenching in hours [1]
        (assumes symmetrical).

    Returns
    -------
    flr_corrected: numpy.ndarray or pandas.Series
        The fluorescence data corrected for quenching.
    quenching layer: bool
        A boolean mask of where the fluorescence is quenched.

    """

    import numpy as np
    import pandas as pd

    from scipy.interpolate import Rbf

    from .cleaning import rolling_window

    def grad_min(depth, fluor_diff, surface_layer=5):
        """
        TODO:   need to refine this function. Doesn't always correct to
                the deepest quenching point
        Quenching depth for a day/night fluorescence difference

        INPUT:   depth and fluorescence as pd.Series or np.ndarray
                 surface_layer [5] is the depth to search for the
                     reference in the gradient
        OUPUT:   Quenching layer as a boolean mask
        """
        if depth.size <= surface_layer:
            return np.zeros(depth.size).astype(bool)

        x = np.array(depth)
        y = rolling_window(np.array(fluor_diff), np.nanmean, 5)
        s = x < surface_layer  # surface data to the top 5 metres
        mask = np.zeros(depth.size).astype(bool)

        # get the smallest 5 points and where the difference crosses 0
        small5 = np.argsort(np.abs(y))[:5]
        cross0 = np.where(np.r_[False, np.diff((y) > 0)])[0]
        # combine the indicies
        i = np.unique(np.r_[small5, cross0])
        # the max in the surface as a reference
        if not s.sum():
            return mask
        j = y[s].argmax()

        # calculate the gradient of the selected points to the reference
        grad = (y[s][j] - y[i]) / (x[s][j] - x[i])
        # If there are only nans in the gradient return only nans
        if np.isnan(grad).all():
            return mask
        # get the index of the steepest gradient (min)
        grad_min_i = i[np.nanargmin(grad)]

        # fill the mask with True values above the quenching depth
        mask[0:grad_min_i] = True
        # on up dives the array is backwards so reverse the mask
        if x[-1] < x[0]:
            mask = ~mask
        # If the majority of the points in the selected region are
        # negative (night < day) then return an empty mask
        return mask

    var = flr.copy()  # create a copy for netCDF attrs preservation

    flr = np.array(flr)
    bbp = np.array(bbp)
    dives = np.array(dives)
    depth = np.array(depth)
    time = np.array(time)
    lat = np.array(lat)
    lon = np.array(lon)

    # ############################ #
    #  GENERATE DAY/NIGHT BATCHES  #
    # ############################ #
    sunrise, sunset = sunset_sunrise(time, lat, lon)
    offset = np.timedelta64(sunrise_sunset_offset, "h")
    # creating quenching correction batches, where a batch is a night and the
    # following day
    day = (time > (sunrise + offset)) & (time < (sunset + offset))
    # find day and night transitions
    daynight_transitions = np.abs(np.diff(day.astype(int)))
    # get the cumulative sum of daynight to generate separate batches for day
    # and night
    daynight_batches = daynight_transitions.cumsum()
    # now get the batches with padded 0 to account for the diff
    # also add a bool that makes night_day or day_night batches
    batch = np.r_[0, (daynight_batches + night_day_group) // 2]
    isday = (np.r_[0, daynight_batches] / 2 % 1) == 0

    # ######################## #
    #  GET NIGHTTIME AVERAGES  #
    # ######################## #
    # blank arrays to be filled
    flr_night, bbp_night = flr.copy(), bbp.copy()

    # create a dataframe with fluorescence and backscatter
    df = pd.DataFrame(np.c_[flr, bbp], columns=["flr", "bbp"])
    # get the binned averages for each batch and select the night
    night_ave = df.groupby([day, batch, np.around(depth)]).mean()
    night_ave = night_ave.dropna().loc[False]
    # A second group where only batches are grouped
    grp_batch = df.groupby(batch)

    # GETTING NIGHTTIME AVERAGE FOR NONGRIDDED DATA - USE RBF INTERPOLATION
    for b in np.unique(night_ave.index.codes[0]):
        i = grp_batch.groups[b].values  # batch index
        j = i[~np.isnan(flr[i]) & (depth[i] < 400)]  # index without nans
        x = night_ave.loc[b].index.values  # batch depth
        y = night_ave.loc[b]  # batch flr and bbp

        if y.flr.isna().all() | y.bbp.isna().all():
            continue
        elif y.flr.size <= 2:
            continue
        # radial basis functions with a smoothing factor
        f1 = Rbf(x, y.flr.values, function="linear", smooth=20)
        f2 = Rbf(x, y.bbp.values, function="linear", smooth=20)
        # interpolation function is used to find flr and bbp for all
        # nighttime fluorescence
        flr_night[j] = f1(depth[j])
        bbp_night[j] = f2(depth[j])

    # calculate the difference between average nighttime - and fluorescence
    fluor_diff = flr_night - flr

    # ################################ #
    #  FIND THE QUENCHING DEPTH LAYER  #
    # ################################ #
    # create a "photic layer" mask to which calc will be limited daytime,
    # shalower than [100m] and fluoresence is quenched relative to night
    photic_layer = isday & (depth < max_photic_depth) & (fluor_diff > 0)
    # blank array to be filled
    quenching_layer = np.zeros(depth.size).astype(bool)
    # create a grouped dataset by dives to find the depth of quenching
    cols = np.c_[depth, fluor_diff, dives][photic_layer]
    grp = pd.DataFrame(cols, columns=["depth", "flr_dif", "dives"])
    grp = grp.groupby("dives")
    # apply the minimum gradient algorithm to each dive
    quench_mask = grp.apply(lambda df: grad_min(df.depth, df.flr_dif))
    # fill the quench_layer subscripted to the photic layer
    quenching_layer[photic_layer] = np.concatenate([el for el in quench_mask])

    # ################################### #
    #  DO THE QUENCHING CORRECTION MAGIC  #
    # ################################### #
    # a copy of fluorescence to be filled with quenching corrected data
    flr_corrected = flr.copy()
    # nighttime backscatter to fluorescence ratio
    flr_bb_night = flr_night / bbp_night
    # quenching ratio for nighttime
    quench_ratio = flr_bb_night * bbp / flr
    # apply the quenching ratio to the fluorescence
    quench_corrected = flr * quench_ratio
    # if unquenched data is corrected return the original data
    mask = quench_corrected < flr
    quench_corrected[mask] = flr[mask]
    # fill the array with queching corrected data in the quenching layer only
    flr_corrected[quenching_layer] = quench_corrected[quenching_layer]

    flr_corrected = transfer_nc_attrs(
        getframe(), var, flr_corrected, "flr_quench_corrected", units="RFU"
    )
    quenching_layer = transfer_nc_attrs(
        getframe(), var, quenching_layer, "quench_layer", units=""
    )

    return flr_corrected, quenching_layer


def quenching_report(
    flr, flr_corrected, quenching_layer, dives, depth, pcolor_kwargs={}
):
    """
    A report for the results of optics.quenching_correction.

    The function creates a figure object of 3 subplots containing a pcolormesh
    of the original fluorescence data, the quenching corrected fluorescence
    data and the quenching layer calculated from the
    optics.quenching_correction function.

    Parameters
    ----------
    flr: numpy.ndarray or pandas.Series
        Fluorescence data after cleaning and factory calibration conversion.
    flr_corrected: numpy.ndarray or pandas.Series
        The fluorescence data corrected for quenching.
    quenching layer: bool
        A boolean mask of where the fluorescence is quenched.
    dives: numpy.ndarray or pandas.Series
        The dive count (round is down dives, 0.5 is up dives).
    depth: numpy.ndarray or pandas.Series
        The depth array in metres.
    pcolor_kwargs: dict
        A dictionary of keyword arguements passed to pcolormesh.

    Returns
    -------
    figure: object
        Creates a figure object with 3 subplots containing a pcolormesh of the
        original fluorescence data, the quenching corrected fluorescence data
        and the quenching layer calculated from the optics.quenching_correction
        function.
    """
    from matplotlib.pyplot import cm, subplots
    from numpy import array, nanpercentile

    from . import plot

    y = array(depth)
    i = y < 183
    y = y[i]
    x = array(dives)[i]
    z = [array(flr)[i], array(flr_corrected)[i], array(quenching_layer)[i]]

    fig, ax = subplots(3, 1, figsize=[10, 11], dpi=90)
    title = "Quenching correction with Thomalla et al. (2017)"

    bmin, bmax = nanpercentile(z[1], [2, 98])
    smin, smax = nanpercentile(z[2], [2, 98])
    props = dict(cmap=cm.YlGnBu_r)
    props.update(pcolor_kwargs)

    plot.pcolormesh(x, y, z[0], ax=ax[0], vmin=bmin, vmax=bmax, **props),
    plot.pcolormesh(x, y, z[1], ax=ax[1], vmin=bmin, vmax=bmax, **props),
    plot.pcolormesh(x, y, z[2], ax=ax[2], vmin=smin, vmax=smax, **props),

    for i in range(0, 3):
        ax[i].set_ylim(180, 0)
        ax[i].set_xlim(x.min(), x.max())
        ax[i].set_ylabel("Depth (m)")
        ax[i].set_xlabel("Dive number")

        if i != 2:
            ax[i].set_xticklabels([])
            ax[i].cb.set_label("Relative Units")
        else:
            ax[i].set_xlabel("Dive number")
            ax[i].cb.set_label("Boolean mask")

    ax[0].set_title("Original fluorescence")
    ax[1].set_title("Quenching corrected fluorescence")
    ax[2].set_title("Quenching layer")

    fig.tight_layout()
    fig.text(0.47, 1.02, title, va="center", ha="center", size=14)

    return fig


if __name__ == "__main__":
    pass
