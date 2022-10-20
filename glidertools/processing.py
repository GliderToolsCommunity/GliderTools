#!/usr/bin/env python
from __future__ import absolute_import as _ai
from __future__ import print_function as _pf
from __future__ import unicode_literals as _ul

from inspect import currentframe as getframe

from .helpers import printv, transfer_nc_attrs


def calc_physics(
    variable,
    dives,
    depth,
    spike_window=3,
    spike_method="minmax",
    iqr=1.5,
    depth_threshold=400,
    mask_frac=0.2,
    savitzky_golay_window=11,
    savitzky_golay_order=2,
    verbose=True,
    name="Physics Variable",
):
    """
    A standard setup for processing physics variables (temperature, salinity).

    The function applies a neighbourhood interquartile range (IQR)
    outlier filter, the Briggs et al. (2011) spike filter
    followed by a Savitzky-Golay smoothing function.

    The Savitzky-Golay filter is demonstrated well on wikipedia:
    https://en.wikipedia.org/wiki/Savitzky-Golay_filter
    """

    from numpy import array, isnan

    from .cleaning import (
        despike,
        horizontal_diff_outliers,
        outlier_bounds_iqr,
        savitzky_golay,
    )

    # an interpolation step is added so that no nans are created.
    # Note that this interpolates on the flattened series
    var = variable.copy()  # attribute preservation

    x = array(dives)
    y = array(depth)
    z = array(variable)
    printv(verbose, "\n" + "=" * 50 + "\n{}:".format(name))

    if iqr:
        nans_before = isnan(z).sum()
        z = outlier_bounds_iqr(z, multiplier=iqr)
        nans_after = isnan(z).sum()
        n_masked = nans_after - nans_before
        printv(
            verbose,
            "\tRemoving outliers with IQR * {}: {} obs".format(iqr, n_masked),
        )

    if spike_window:
        z = despike(z, spike_window, spike_method)[0]
        printv(
            verbose,
            "\tRemoving spikes with rolling median (spike window={})".format(
                spike_window
            ),
        )

    if depth_threshold:
        z = horizontal_diff_outliers(x, y, z, iqr, depth_threshold, mask_frac)
        printv(
            verbose,
            ("\tRemoving horizontal outliers " "(fraction={}, multiplier={})").format(
                mask_frac, iqr
            ),
        )

    if savitzky_golay_window:
        printv(
            verbose,
            ("\tSmoothing with Savitzky-Golay filter " "(window={}, order={})").format(
                savitzky_golay_window, savitzky_golay_order
            ),
        )
        z = savitzky_golay(z, savitzky_golay_window, savitzky_golay_order)

    z = transfer_nc_attrs(getframe(), var, z, "_processed")

    return z


def calc_oxygen(
    o2raw,
    pressure,
    salinity,
    temperature,
    auto_conversion=True,
    spike_window=7,
    spike_method="median",
    savitzky_golay_window=0,
    savitzky_golay_order=2,
    verbose=True,
):
    """
    This function processes oxygen.

    It is assumed that either mL/L or umol/kg are passed as input.
    The units are automatically detected by looking at the mean ratio.
    Below are some conversions to help with the Oxygen units:

    >>> µmol/l > µmol/kg * 1.025
        µmol/l > ml/l * 44.66
        µmol/l > mg/l * 31.25

    Parameters
    ----------
    o2raw : array, dtype=float, shape=[n, ]
        raw oxygen without unit conversion
    pressure : array, dtype=float, shape=[n, ]
    salinity : array, dtype=float, shape=[n, ]
    temperature : array, dtype=float, shape=[n, ]
    conversion : bool=True
        tries to determine the unit of oxygen based on ``o2raw`` values.
        The user needs to do a manual conversion if False
    spike_window : int=7
        rolling window size to apply for the ``cleaning.despike`` function.
    spike_method : string='median'
        can be 'median' or 'minmax'. see ``cleaning.despike`` for more info.
    savitzky_golay_window : int=0
        rolling window size for ``cleaning.savitzky_golay`` function
    savitzky_golay_order : int=2
        polynomial order for ``cleaning.savitzky_golay`` function
    verbose : bool=True

    Returns
    -------
    o2mll : array, dtype=float, shape=[n, ]
        oxygen concentration in mL/L (if unit auto_conversion is set True)
    o2pct : array, dtype=float, shape=[n, ]
        theoretical oxygen saturation percentage
    o2aou : array, dtype=float, shape=[n, ]
        aparent oxygen utilisation based on measured oxygen and oxygen
        saturation.

    Note
    ----
    To Do: Oxygen processing should have its own section to be consistent

    """

    import seawater as sw

    from numpy import abs, array, c_, isnan, median, ones
    from pandas import Series

    from .cleaning import despike, outlier_bounds_iqr, savitzky_golay

    var = o2raw.copy()  # metdata preservation
    if isinstance(o2raw, Series):
        name = o2raw.name
    else:
        name = "Oxygen"
    o2raw = array(o2raw)
    pressure = array(pressure)
    temperature = array(temperature)
    salinity = array(salinity)

    if spike_window:
        o2raw, _ = despike(o2raw, spike_window, spike_method)
        printv(
            verbose,
            "\n" + "=" * 50 + "\n{}:\n"
            "\tSmoothing data with despiking algorithm:\n\t"
            "    spike identification (spike window={})"
            "".format(name, spike_window),
        )

    if savitzky_golay_window:
        printv(
            verbose,
            ("\tSmoothing with Savitzky-Golay filter " "(window={}, order={})").format(
                savitzky_golay_window, savitzky_golay_order
            ),
        )
        o2raw = savitzky_golay(o2raw, savitzky_golay_window, savitzky_golay_order)

    o2sat = sw.satO2(salinity, temperature)
    density = sw.dens(salinity, temperature, pressure)

    if auto_conversion:
        # use linear regression to determine the oxygen unit
        # raw surface (<10m) O2 is regressed theoretical saturation
        # the slope of the regression will be indicative of the
        # units as theoretical saturation is always in mL/L
        # Use the min difference between the slope and known
        # conversion factors to estimate the appropriate conversion.

        # clean the data first with basic cleaning
        surf = (pressure < 20) & ~isnan(o2raw) & ~isnan(o2sat)
        # prepare the data for linear regression
        Y = o2raw[surf].copy()
        X = c_[ones(surf.sum()), o2sat[surf]]
        # removing outliers accodring to IQR
        ll, ul = outlier_bounds_iqr(Y, multiplier=1.5)
        m = (Y > ll) & (Y < ul)
        ratios = Y[m] / X[m, 1]

        # compare the slopes
        observed_ratio = median(ratios)
        # the theoretical values have been divided by 1.025 to account for
        # the density of seawater
        theoretic_ratio = array([1, 43.5])
        ratio_diffs = abs(observed_ratio - theoretic_ratio)
        # catch if the difference is too big
        if ratio_diffs.min() > 10:
            printv(
                verbose,
                (
                    "Oxygen unit could not be estimated automatically. "
                    "Do the unit conversion on the raw data before "
                    "passing it to the function. \n"
                    "Below is some info to help you\n"
                    "    µmol/l > µmol/kg * 1.025\n"
                    "    µmol/l > ml/l * 44.66\n"
                    "    µmol/l > mg/l * 31.25"
                ),
            )
        # otherwise do the conversion
        else:
            unit_idx = ratio_diffs.argmin()
            if unit_idx == 0:
                unit = "mL/L"
                o2mll = array(o2raw)
            elif unit_idx == 2:
                unit = "mg/L"
                o2mll = array(o2raw) / 31.25 * (density / 1000)
            elif unit_idx == 1:
                unit = "umol/kg"
                o2mll = array(o2raw) / 44.66 * (density / 1000)
            else:
                printv(verbose, "Difference is {}".format(ratio_diffs))
            printv(verbose, "\tUnits automatically detected {}".format(unit))
            if ratio_diffs.min() > 5:
                print(
                    "\tWARNING: Confirm units mannually as near the "
                    "confidence threshold"
                )
        o2aou = o2sat - o2mll
        o2pct = o2mll / o2sat * 100

        o2mll = transfer_nc_attrs(
            getframe(),
            var,
            o2mll,
            "o2mll",
            units="mL/L",
            comment="",
            standard_name="dissolved_oxygen",
        )
        o2aou = transfer_nc_attrs(
            getframe(),
            var,
            o2mll,
            "o2aou",
            units="mL/L",
            comment="",
            standard_name="aparent_oxygen_utilisation",
        )
        o2pct = transfer_nc_attrs(
            getframe(),
            var,
            o2mll,
            "o2pct",
            units="percent",
            comment="",
            standard_name="theoretical_oxgen_saturation",
        )

        return o2mll, o2pct, o2aou

    else:
        print(
            "No oxygen conversion applied - user "
            "must impliment before or after running "
            "the cleaning functions."
        )


def calc_backscatter(
    bb_raw,
    tempC,
    salt,
    dives,
    depth,
    wavelength,
    dark_count,
    scale_factor,
    spike_window=7,
    spike_method="median",
    iqr=3,
    profiles_ref_depth=300,
    deep_multiplier=1,
    deep_method="median",
    return_figure=False,
    verbose=True,
):
    r"""
    The function processes the raw backscattering data in counts into total
    backscatter (bbp) in metres.

    The function uses a series of steps to clean the data before applying the
    Zhang et al. (2009) functions to convert the data into total backscatter
    (bbp/m)). The function uses functions from the flo_functions toolkit [1]_.
    The theta angle of sensors (124deg) and xfactor for theta 124 (1.076) are
    set values that should be updated if you are not using a WetLabs ECO BB2FL

    The following standard sequence is applied:

    1. find IQR outliers  (i.e. data values outside of the lower and upper
       limits calculated by cleaning.outlier_bounds_iqr)
    2. find_bad_profiles  (e.g. high values below 300 m are counted as bad
       profiles)
    3. flo_scale_and_offset (factory scale and offset)
    4. flo_bback_total  (total backscatter based on Zhang et al. 2009) [2]_
    5. backscatter_dark_count  (based on Briggs et al. 2011) [3]_
    6. despike  (using Briggs et al. 2011 - rolling min--max) [3]_

    Parameters
    ----------

    bb_raw: np.array / pd.Series, dtype=float, shape=[n, ]
        The raw output from the backscatter channel in counts.
    tempC: np.array / pd.Series, dtype=float, shape=[n, ]
        The QC'd temperature data in degC.
    salt: np.array / pd.Series, dtype=float, shape=[n, ]
        The QC'd salinity in PSU.
    dives: np.array / pd.Series, dtype=float, shape=[n, ]
        The dive count (round is down dives, 0.5 is up dives).
    depth: np.array / pd.Series, dtype=float, shape=[n, ]
        The depth array in metres.
    wavelength: int
        The wavelength of the backscatter channel, e.g. 700 nm.
    dark_count: float
        The dark count factory values from the calibration sheet.
    scale_factor: float
        The scale factor factory values from the calibration sheet.
    spike_window: int
        The window size over which to run the despiking method.
    spike_method: str
        Whether to use a rolling median or combination of min+max filter as
        the despiking method.
    iqr: int
        Multiplier to determine the lower and upper limits of the
        interquartile range for outlier detection.
    profiles_ref_depth: int
        The depth threshold for optics.find_bad_profiles below which the
        median or mean is calculated for identifying outliers.
    deep_multiplier: int=1
        The standard deviation multiplier for calculating outliers,
        i.e. :math:`\mu \pm \sigma \cdot[1]`.
    deep_method: str
        Whether to use the deep median or deep mean to determine bad profiles
        for optics.find_bad_profiles.
    return_figure: bool
        If True, will return a figure object that shows before and after the
        quenching correction was applied.
    verbose: bool
        If True, will print the progress of the processing function.

    Returns
    -------
    baseline: numpy.ma.masked_array
        The despiked + bad profile identified backscatter with the mask
        denoting the filtered values of the backscatter baseline as
        defined in Briggs et al. (2011).
    quench_corrected: np.array / pd.Series, dtype=float, shape=[n, ]
        The backscatter spikes as defined in Briggs et al. (2011).
    figs: object
        The figures reporting the despiking, bad profiles and quenching
        correction.

    References
    ----------
    .. [1] https://github.com/ooici/ion-functions Copyright (c) 2010, 2011 The
           Regents of the University of California
    .. [2] Zhang, X., Hu, L., & He, M. (2009). Scattering by pure seawater:
           Effect of salinity. Optics Express, 17(7), 5698.
           https://doi.org/10.1364/OE.17.005698
    .. [3] Briggs, N., Perry, M. J., Cetinic, I., Lee, C., D'Asaro, E., Gray,
           A. M., & Rehm, E. (2011). High-resolution observations of aggregate
           flux during a sub-polar North Atlantic spring bloom. Deep-Sea
           Research Part I: Oceanographic Research Papers, 58(10), 1031–1039.
           https://doi.org/10.1016/j.dsr.2011.07.007


    """
    from numpy import array, count_nonzero, isnan, nan, unique
    from pandas import Series

    from . import flo_functions as ff
    from . import optics as op
    from .cleaning import despike, despiking_report, outlier_bounds_iqr

    var = bb_raw.copy()  # metadata preservation
    bb_raw = Series(bb_raw.copy())
    dives = array(dives)
    depth = array(depth)
    tempC = array(tempC)
    salt = array(salt)

    name = "bb{:.0f}".format(wavelength)
    theta = 124  # factory set angle of optical sensors
    xfactor = 1.076  # for theta 124
    # Values taken from Sullivan et al. (2013) & Slade and Boss (2015)

    ref_depth = profiles_ref_depth
    stdev_multiplier = deep_multiplier
    method = deep_method

    dive_count = count_nonzero(unique(dives))

    printv(verbose, "\n" + "=" * 50 + "\n{}:".format(name))

    if iqr:

        nans_before = isnan(bb_raw).sum()
        bb_raw = outlier_bounds_iqr(bb_raw, multiplier=iqr)
        nans_after = isnan(bb_raw).sum()
        n_masked = nans_after - nans_before
        printv(
            verbose,
            "\tRemoving outliers with IQR * {}: {} obs".format(iqr, n_masked),
        )

    printv(
        verbose,
        "\tMask bad profiles based on deep values (depth={}m)".format(ref_depth),
    )
    bad_profiles = op.find_bad_profiles(
        dives, depth, bb_raw, ref_depth, stdev_multiplier, method
    )
    bb_raw[bad_profiles[0]] = nan

    bad_count = count_nonzero(bad_profiles[1])

    printv(
        verbose,
        "\tNumber of bad profiles = {}/{}".format(bad_count, dive_count),
    )
    printv(verbose, "\tZhang et al. (2009) correction")
    beta = ff.flo_scale_and_offset(bb_raw, dark_count, scale_factor)
    bbp = ff.flo_bback_total(beta, tempC, salt, theta, wavelength, xfactor)

    # This is from .Briggs et al. (2011)
    printv(verbose, "\tDark count correction")
    bbp = op.backscatter_dark_count(bbp, depth)

    printv(
        verbose,
        "\tSpike identification (spike window={})".format(spike_window),
    )
    baseline, spikes = despike(bbp, spike_window, spike_method="median")
    baseline = Series(baseline, name="bb{:.0f}".format(wavelength))

    baseline = transfer_nc_attrs(
        getframe(),
        var,
        baseline,
        name + "_baseline",
        units="units",
        standard_name="backscatter",
    )
    spikes = transfer_nc_attrs(
        getframe(),
        var,
        spikes,
        name + "_spikes",
        units="units",
        standard_name="backscatter",
    )

    if not return_figure:
        return baseline, spikes
    else:
        printv(verbose, "\tGenerating figure for despiking report")
        fig = despiking_report(dives, depth, bbp, baseline, spikes, name=name)

        return baseline, spikes, fig


def calc_fluorescence(
    flr_raw,
    bbp,
    dives,
    depth,
    time,
    lat,
    lon,
    dark_count,
    scale_factor,
    spike_window=7,
    spike_method="median",
    night_day_group=True,
    sunrise_sunset_offset=1,
    profiles_ref_depth=300,
    deep_multiplier=1,
    deep_method="median",
    return_figure=False,
    verbose=True,
):
    r"""
    This function processes raw fluorescence and corrects for quenching using
    the Thomalla et al. (2018) approach [1]_.

    The following standard sequence is applied:

    1. find_bad_profiles  (e.g. high Fluorescence in > 300 m water signals
       bad profile)
    2. fluorescence_dark_count & scale factor  (i.e. factory correction)
    3. despike  (using Briggs et al. 2011 - rolling min--max)
    4. quenching_correction  (corrects for quenching with Thomalla et al. 2017)

    Parameters
    ----------
    flr_raw: np.array / pd.Series, dtype=float, shape=[n, ]
        The raw output of fluorescence data in instrument counts.
    bbp: np.array / pd.Series, dtype=float, shape=[n, ]
        The processed backscatter data from the less noisy channel, i.e. the
        one dataset with less spikes or bad profiles.
    dives: np.array / pd.Series, dtype=float, shape=[n, ]
        The dive count (round is down dives, 0.5 is up dives).
    depth: np.array / pd.Series, dtype=float, shape=[n, ]
        The depth array in metres.
    time: np.array / pd.Series, dtype=float, shape=[n, ]
        The date & time array in a numpy.datetime64 format.
    lat: np.array / pd.Series, dtype=float, shape=[n, ]
        The latitude of the glider position.
    lon: np.array / pd.Series, dtype=float, shape=[n, ]
        The longitude of the glider position.
    dark_count: float
        The dark count factory values from the calibration sheet.
    scale_factor: float
        The scale factor factory values from the calibration sheet.
    spike_window: int=7
        The window size over which to run the despiking method.
    spike_method: str=median
        Whether to use a rolling median or combination of min+max filter as
        the despiking method.
    night_day_group: bool=True
        If True, use preceding night otherwise use following night for
        calculating the flr:bbp ratio.
    sunrise_sunset_offset: int=1
        The delayed onset and recovery of quenching in hours [1]
        (assumes symmetrical).
    profiles_ref_depth: int=300
        The depth threshold for optics.find_bad_profiles below which the
        median or mean is calculated for identifying outliers.
    deep_multiplier: int=1
        The standard deviation multiplier for calculating outliers,
        i.e. mean ± std x [1].
    deep_method: str='median'
        Whether to use the deep median or deep mean to determine bad profiles
        for optics.find_bad_profiles.
    return_figure: bool=False
        If True, will return a figure object that shows before and after the
        quenching correction was applied.
    verbose: bool=True
        If True, will print the progress of the processing function.

    Returns
    -------
    baseline: array, dtype=float, shape=[n, ]
        The despiked + bad profile identified fluorescence that has not had
        the quenching correction applied.
    quench_corrected: array, dtype=float, shape=[n, ]
        The fluorescence data corrected for quenching.
    quench_layer: array, dtype=bool, shape=[n, ]
        The quenching layer as a mask.
    figs: object
        The figures reporting the despiking, bad profiles and quenching
        correction.

    References
    ----------
    .. [1] Thomalla, S. J., Moutier, W., Ryan-Keogh, T. J., Gregor, L.,
           & Schutt, J. (2018). An optimized method for correcting fluorescence
           quenching using optical backscattering on autonomous platforms.
           Limnology and Oceanography: Methods, 16(2), 132–144.
           https://doi.org/10.1002/lom3.10234

    """

    from numpy import array, count_nonzero, nan, unique

    from . import optics as op
    from .cleaning import despike, despiking_report

    var = flr_raw.copy()  # metdata preservation
    flr_raw = array(flr_raw)
    bbp = array(bbp)
    dives = array(dives)
    depth = array(depth)
    time = array(time)
    lat = array(lat)
    lon = array(lon)
    ref_depth = profiles_ref_depth
    stdev_multiplier = deep_multiplier
    method = deep_method

    printv(
        verbose,
        (
            "\n" + "=" * 50 + "\nFluorescence\n\tMask bad profiles based on "
            "deep values (ref depth={}m)"
        ).format(ref_depth),
    )
    bad_profiles = op.find_bad_profiles(
        dives, depth, flr_raw, ref_depth, stdev_multiplier, method
    )
    flr_raw[bad_profiles[0]] = nan

    bad_count = count_nonzero(bad_profiles[1])
    dive_count = count_nonzero(unique(dives))
    printv(
        verbose,
        "\tNumber of bad profiles = {}/{}".format(bad_count, dive_count),
    )

    printv(verbose, "\tDark count correction")
    flr_raw -= dark_count
    flr_dark = op.fluorescence_dark_count(flr_raw, depth)
    flr_dark[flr_dark < 0] = nan

    baseline, spikes = despike(flr_dark, spike_window, spike_method="median")

    printv(verbose, "\tQuenching correction")
    quench_corrected, quench_layer = op.quenching_correction(
        baseline,
        bbp,
        dives,
        depth,
        time,
        lat,
        lon,
        sunrise_sunset_offset=1,
        night_day_group=True,
    )

    printv(
        verbose,
        "\tSpike identification (spike window={})".format(spike_window),
    )

    baseline = transfer_nc_attrs(
        getframe(),
        var,
        baseline,
        "FLR_baseline",
        units="RFU",
        standard_name="",
    )
    quench_corrected = transfer_nc_attrs(
        getframe(),
        var,
        quench_corrected,
        "FLR_quench_corrected",
        units="RFU",
        standard_name="fluorescence",
    )
    quench_layer = transfer_nc_attrs(
        getframe(),
        var,
        quench_layer,
        "quenching_layer",
        units="",
        standard_name="",
        comment="",
    )

    if return_figure:
        printv(verbose, "\tGenerating figures for despiking and quenching report")
        figs = (
            despiking_report(
                dives,
                depth,
                flr_raw,
                baseline.data,
                spikes,
                name="Fluorescence",
            ),
        )
        figs += (
            op.quenching_report(
                baseline.data,
                quench_corrected.data,
                quench_layer,
                dives,
                depth,
            ),
        )
        return baseline, quench_corrected, quench_layer, figs
    else:
        return baseline, quench_corrected, quench_layer


def calc_par(
    par_raw,
    dives,
    depth,
    time,
    scale_factor_wet_uEm2s,
    sensor_output_mV,
    curve_max_depth=80,
    verbose=True,
):
    """
    Calculates the theoretical PAR based on an exponential curve fit.

    The processing steps are:

    1. par_scaling  (factory cal sheet scaling)
    2. par_dark_count  (correct deep par values to 0 using 5th %)
    3. par_fill_surface  (return the theoretical curve of par based
       exponential fit)

    Parameters
    ----------
    All inputs must be ungridded np.ndarray or pd.Series data
    par_raw : array, dtype=float, shape=[n, ]
        raw PAR
    dives : array, dtype=float, shape=[n, ]
        the dive count (round is down dives, 0.5 up dives)
    depth : array, dtype=float, shape=[n, ]
        in metres
    time : array, dtype=float, shape=[n, ]
        as a np.datetime64 array

    Returns
    -------
    par_filled : array, dtype=float, shape=[n, ]
        PAR with filled surface values.
    """

    from numpy import array

    from . import optics as op

    var = par_raw.copy()  # metdata presrevation
    par_raw = array(par_raw)
    dives = array(dives)
    depth = array(depth)
    time = array(time)

    printv(verbose, "\n" + "=" * 50 + "\nPAR\n\tDark correction")

    # dark correction for par
    par_scaled = op.par_scaling(par_raw, scale_factor_wet_uEm2s, sensor_output_mV)
    par_dark = op.par_dark_count(par_scaled, dives, depth, time)
    printv(verbose, "\tFitting exponential curve to data")
    par_filled = op.par_fill_surface(
        par_dark, dives, depth, max_curve_depth=curve_max_depth
    )
    par_filled[par_filled < 0] = 0

    attrs = dict(
        standard_name="photosynthetically_available_radiation",
        units="uE/m2/s2",
        comment="",
    )
    par_filled = transfer_nc_attrs(
        getframe(), var, par_filled, "PAR_processed", **attrs
    )
    par_filled = par_filled.fillna(0)

    return par_filled


if __name__ == "__main__":
    pass
