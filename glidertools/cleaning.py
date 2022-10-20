#!/usr/bin/env python
from __future__ import absolute_import as _ai
from __future__ import print_function as _pf
from __future__ import unicode_literals as _ul

from inspect import currentframe as getframe

from .helpers import transfer_nc_attrs


def outlier_bounds_std(arr, multiplier=3):
    r"""
    Mask values outside the upper and lower outlier limits by standard
    deviation

        :math:`\mu \pm 3\sigma`

    the multiplier [3] can be adjusted by the user
    returns the lower_limit, upper_limit

    Parameters
    ----------
    arr : np.array|xr.DataArray, dtype=float, shape=[n, ]
        the full timeseries of the entire dataset
    multiplier : float=1.5
        sets the standard deviation multiplier

    Returns
    -------
    arr : array | xarray.DataArray
        A data object where values outside the limits are masked.
        Metdata will be preserved if the original input array is xr.DataArray

    """

    from numpy import array, nan, nanmean, nanstd

    var = arr.copy()
    arr = array(arr)

    mean = nanmean(arr)
    std = nanstd(arr)

    ll = mean - std * multiplier
    ul = mean + std * multiplier

    mask = (arr < ll) | (arr > ul)
    arr[mask] = nan

    attrs = dict(outlier_lims=[ll, ul])

    out = transfer_nc_attrs(getframe(), var, arr, "_outlierSTD", **attrs)

    return out


def outlier_bounds_iqr(arr, multiplier=1.5):
    r"""
    Mask values outside the upper/lower outlier limits by interquartile range:

    .. math::

        lim_{low} = Q_1 - 1.5\cdot(Q_3 - Q_1)\\
        lim_{up} = Q_3 + 1.5\cdot(Q_3 - Q_1)

    the multiplier [1.5] can be adjusted by the user
    returns the lower_limit, upper_limit

    Parameters
    ----------
    arr : np.array|xr.DataArray, dtype=float, shape=[n, ]
        the full timeseries of the entire dataset
    multiplier : float=1.5
        sets the interquartile range

    Returns
    -------
    arr : array | xarray.DataArray
        A data object where values outside the limits are masked.
        Metdata will be preserved if the original input array is xr.DataArray


    """
    from numpy import array, nan, nanpercentile

    var = arr.copy()
    arr = array(arr)

    q1, q3 = nanpercentile(arr, [25, 75])
    iqr = q3 - q1

    ll = q1 - iqr * multiplier
    ul = q3 + iqr * multiplier

    mask = (arr < ll) | (arr > ul)
    arr[mask] = nan

    attrs = dict(outlier_lims=[ll, ul])

    out = transfer_nc_attrs(getframe(), var, arr, "_outlierIQR", **attrs)
    return out


def horizontal_diff_outliers(
    dives, depth, arr, multiplier=1.5, depth_threshold=450, mask_frac=0.2
):
    """
    Find Z-score outliers (> 3) on the horizontal. Can be limited below a
    certain depth.

    The function uses the horizontal gradient as a threshold, below a defined
    depth threshold to find outliers. Useful to identify when a variable at
    depth is not the same as neighbouring values.

    Parameters
    ----------

    dives: numpy.ndarray or pandas.Series
        The dive count (round is down dives, 0.5 is up dives)
    depth: numpy.ndarray or pandas.Series
        The depth array in metres
    arr: numpy.ndarray or pandas.Series
        Array of data variable for cleaning to be performed on.
    multiplier: float
        A z-score threshold
    depth_threshold: int
        Outliers will be identified below this depth value to the max depth
        value of the dive.
    mask_frac: float
        When the ratio of bad values per dive is greater than this value, then
        the dive will be masked.

    Returns
    -------
    mask
        A mask of dives where the bad values per dive ratio is greater than
        mask_frac.
    """
    from numpy import abs, arange, array, inf, nanmean, nanstd

    from .mapping import grid_data

    var = arr.copy()
    dives = array(dives)
    depth = array(depth)
    arr = array(arr)

    # grid data so that the horizontal rolling median can be calculated
    # we use a window of 3 to find only "horizonal spikes"
    gridded = grid_data(
        dives,
        depth,
        array(arr),
        bins=arange(0, depth.max(), 1),
        verbose=False,
        return_xarray=False,
    )
    median = gridded.rolling(3, axis=1, center=True, min_periods=2).median()
    # get zscore of the difference between the median and the raw data
    diff = gridded - median
    zdiff = abs(diff - nanmean(diff)) / nanstd(diff)

    # this finds the 99.7th percentile outliers
    # note that this is based on the global horizonal diff
    # but is only applied below the depth threshold
    # this means that the surface data sets a higher limit
    deep_outlier = zdiff.loc[depth_threshold:] >= multiplier

    # get the ratio of bad values per dive and mask if it
    # exceeds a user defined fraction
    deep_outlier_count = deep_outlier.sum()
    deep_obs_num = gridded.shape[0] - depth_threshold  # assumes bin of 1m
    deep_outlier_ratio = deep_outlier_count / deep_obs_num
    # finds the index where dives exceed the mask_frac threshold
    i = deep_outlier_ratio > mask_frac
    deep_outlier_dives = i[i].index.values

    mask = arr < -inf  # create a dummy mask
    for d in deep_outlier_dives:
        i = dives == d
        mask[i] = True

    baddives = mask_bad_dive_fraction(mask, dives, arr, mask_frac=mask_frac)[0]
    out = transfer_nc_attrs(getframe(), var, baddives, "_horzOutlierSTD")

    return out


def mask_bad_dive_fraction(mask, dives, var, mask_frac=0.2):
    """
    Find bad dives - where more than a fraction of the dive is masked

    Parameters
    ----------
    mask : array, dtype=bool, shape=[n, ]
        boolean 1D array with masked values
    dives : array, dtype=float, shape=[n, ]
        discrete dive numbers (down round, up n.5)
    var : array, dtype=float, shape=[n, ]
        series or array containing data that will be masked with NaNs
    mask_frac : int=0.2
        fraction of the dive that is masked for the whole dive to be bad

    Returns
    -------
    var : array, dtype=float, shape=[n, ]
        the same as the input, but has been masked
    mask_dives : array, dtype=bool
        a mask array that has full dives that are deemed "bad" masked out

    """
    from numpy import NaN, array
    from pandas import Series

    # catch dives where the marjority of the data is masked
    # and return a fully masked dive
    dives = array(dives)
    arr = array(var)

    grp = Series(mask).groupby(dives)
    masked_frac_per_dive = grp.sum() / grp.count() > mask_frac
    majority_masked = masked_frac_per_dive[masked_frac_per_dive].index.values

    # create a mask that masks ungridded data
    mask_dives = mask.copy()
    for d in majority_masked:
        i = array(dives) == d
        mask_dives[i] = True

    arr[mask_dives] = NaN
    baddive = arr

    baddive = transfer_nc_attrs(getframe(), var, baddive, None)

    return baddive, mask_dives


def data_density_filter(x, y, conv_matrix=None, min_count=5, return_figures=True):
    """
    Use the 2D density cloud of observations to find outliers for any variables

    The data density filter needs tuning to work well.
    This uses convolution to create the density cloud - you can specify
    the exact convolution matrix, or its shape

    Parameters
    ----------
    x : np.array / pd.Series, shape=[n, ]
        e.g. temperature
    y : np.array / pd.Series, shape=[n, ]
        e.g. salinity
    conv_matrix : int, list, np.array, optional
        int = size of the isotropic round convolution window.
        [int, int] = anisotropic (oval) convoltion window.
        2d array is a weighted convolution window;
        rectangle = np.ones([int, int]);
        more advanced anisotropic windows can also be created
    min_count : int, default=5, optional
        masks the 2d histogram counts smaller than this limit when performing
        the convolution
    return_figures : bool, default=True, optional
        returns figures of the data plotted for blob detection...

    Returns
    -------
    mask : np.array, shape=[n, ]
        a mask that returns only values
    figure :
        only returned if return_figure is True

    """
    from numpy import array, c_, inf, isnan, linspace, where
    from pandas import Series, cut
    from scipy.signal import convolve2d

    def gaussian_kernel(*shape):
        """
        Returns a 2D array with gaussian values to be used in the blob_outliers
        detection function. Can be anisotripic (oblong). Scaling is determined
        automatically.

        Parameters
        ----------
        shape : int, int
            if one integer is passed the kernel will be isotropic
            if two integers are passed the kernel will be anisotropic

        Returns
        -------
        array  (float)
            The 2D representation of the kernel
        """
        from matplotlib.cbook import flatten
        from numpy import exp, mgrid

        # make shape a list regardless of input
        shape = [int(a // 2) for a in flatten([shape])]
        # anisotropic if len(2) else isotropic
        if len(shape) == 1:
            sx, sy = shape[0], shape[0]
        elif len(shape) == 2:
            sx, sy = shape

        # create the x and y grid
        x, y = mgrid[-sx : sx + 1, -sy : sy + 1]
        sigma = [sx / 8, sy / 8]  # sigma scaled by shape
        c = tuple([sx, sy])  # centre index of x and y
        g = 1 * exp(
            -(
                (x - x[c]) ** 2 / (2 * sigma[0]) ** 2
                + (y - y[c]) ** 2 / (2 * sigma[1]) ** 2
            )
        )
        return g

    # turning input into pandas.Series
    x = Series(x, name="X" if not isinstance(x, Series) else x.name)
    y = Series(y, name="Y" if not isinstance(y, Series) else y.name)

    ###############
    #   BINNING   #
    ###############
    # create bins for the data - equal bins
    xbins = linspace(x.min(), x.max(), 250)
    ybins = linspace(y.min(), y.max(), 250)
    # binning the data with pandas. This is quick to find outliers at the end
    xcut = cut(x, xbins, labels=c_[xbins[:-1], xbins[1:]].mean(1), right=False)
    ycut = cut(y, ybins, labels=c_[ybins[:-1], ybins[1:]].mean(1), right=False)

    # binning the data and returning as a 2D array (pandas.DataFrame)
    count = x.groupby([xcut, ycut]).count()
    count.name = "count"  # to avoid an error when unstacking
    count = count.unstack()
    count = count.sort_index().sort_index(axis=1)

    ###################
    #   CONVOLUTION   #
    ###################
    # make convolution matrix if not given
    if conv_matrix is None:
        conv_matrix = (gaussian_kernel(21) > 1e-5).astype(int)
    elif isinstance(conv_matrix, (list, int, float)):
        conv_matrix = (gaussian_kernel(conv_matrix) > 1e-5).astype(int)
    else:
        ndim = array(conv_matrix).ndim
        if ndim != 2:
            raise UserWarning("conv_matrix must have 2 dimensions")
    # An array with which the convolution is done
    # use a threshold to mask out bins with low counts
    # thus only dense regions of data are considered
    count0 = count.fillna(0).values
    count0[count0 < min_count] = 0
    # 2d convolution with the input matrix
    convolved_count = convolve2d(count0, conv_matrix, mode="same")
    outliers = (convolved_count == 0) & ~isnan(count)

    cols = count.index
    rows = count.columns

    ########################################
    #   FINDING OUTLIERS AND CREATE MASK   #
    ########################################
    # find indicies of of the where there is no convolution,
    # but there are data. Then get the x and y values of these
    # points. Turn these into pairs for pandas multi-indexing.
    i, j = where(outliers)
    xi = cols[i].values
    yj = rows[j].values
    ij = list(zip(xi, yj))
    # Create a pandas dataframe with the pd.cut data as indicies
    # with a column for a numerical index.
    if len(ij) > 0:
        idx = x.to_frame().reset_index().drop(x.name, axis=1)
        idx = idx.set_axis([xcut, ycut], inplace=False)
        idx = idx.loc[ij]["index"].values
    else:
        idx = None
    # create a placeholder mask and fill outliers with True
    mask = (x > inf).values
    mask[idx] = True

    ###############
    #   FIGURES   #
    ###############
    if return_figures:
        from matplotlib import cm, colors
        from matplotlib import pyplot as plt
        from numpy import ma, r_

        # x and y plotting coordinates
        xp = cols.values.astype(float)
        yp = rows.values.astype(float)
        # plotting variables a, b, c
        a = ma.masked_invalid(count.T, 0)
        b = convolved_count.T
        c = ma.masked_where(a.mask, ~outliers.T)

        # create the figure
        fig, ax = plt.subplots(1, 2, figsize=[10, 5], dpi=90, sharey=True)
        # properties for the pcolormesh and contours
        pn = colors.PowerNorm(0.3)
        mesh_props = dict(cmap=cm.Spectral_r, norm=pn)
        # create the pcolormesh plots
        im = (
            ax[0].pcolormesh(xp, yp, a, vmax=a.max() / 2, **mesh_props),
            ax[1].pcolormesh(xp, yp, c, vmin=0, vmax=1),
        )
        ax[1].contour(xp, yp, b, levels=[0.5], linestyles="-", colors="r", linewidths=2)

        # change figure parameters
        ax[0].set_title("Histogram of data (min_count = {})".format(min_count))
        ax[1].set_title(
            "{} Outliers found using\n{} convolution decision boundary".format(
                mask.sum(), str(conv_matrix.shape)
            )
        )
        ax[0].set_xticks([])
        ax[0].set_ylabel(y.name)
        ax[1].set_xlabel(x.name)

        # tight layout before creating the axes for pcolomesh plots
        fig.tight_layout()

        # make colorbar axes based on axes [0, 1]
        p = ax[0].get_position()
        cax = fig.add_axes([p.x0, p.y0 - 0.05, p.width, 0.04])
        cb = plt.colorbar(im[0], cax=cax, orientation="horizontal")
        cb.set_label("Count")
        cb.set_ticks([1, 2, 3, 5, 10, 30, 80, 200])
        # plot the min_count on the colorbar
        cx = pn(r_[cb.get_clim(), min_count])[-1]
        cb.ax.plot(cx, 0, marker="^", color="k", markersize=8)
        cb.ax.plot(cx, 1, marker="v", color="k", markersize=8)
        return mask, fig

    return mask


def despike(var, window_size, spike_method="median"):
    """
    Return a smooth baseline of data and the anomalous spikes

    This script is copied from Nathan Briggs' MATLAB script as described in
    Briggs et al (2011). It returns the baseline of the data using either a
    rolling window method and the residuals of [measurements - baseline].

    Parameters
    ----------
    arr: numpy.ndarray or pandas.Series
        Array of data variable for cleaning to be performed on.
    window_size: int
        the length of the rolling window size
    method: str
        A string with `minmax` or `median`. 'minmax' first applies a rolling
        minimum to the dataset thereafter a rolling maximum is applied. This
        forms the baseline, where the spikes are the difference from the
        baseline. 'median' first applies a rolling median to the dataset, which
        forms the baseline. The spikes are the difference between median and
        baseline, and thus are more likely to be negative.

    Returns
    -------
    baseline: numpy.ndarray or pandas.Series
        The baseline from which outliers are determined.
    spikes: numpy.ndarray or pandas.Series
        Spikes are the residual of [measurements - baseline].


    """
    from numpy import array, isnan, nan, nanmax, nanmedian, nanmin, ndarray

    # convert to array
    arr = array(var)
    # create empty array for baseline
    baseline = ndarray(arr.shape) * nan
    # mask with exisiting nans masked out
    mask = ~isnan(arr)

    # if min-max method then get the rolling minimum and
    # then the rolling maximum
    if spike_method.startswith("min"):
        base_min = rolling_window(arr[mask], nanmin, window_size)
        base = rolling_window(base_min, nanmax, window_size)
    else:
        base = rolling_window(arr[mask], nanmedian, window_size)

    baseline[mask] = base
    spikes = arr - baseline

    baseline = transfer_nc_attrs(getframe(), var, baseline, "_baseline")
    spikes = transfer_nc_attrs(getframe(), var, spikes, "_spikes")

    return baseline, spikes


def despiking_report(dives, depth, raw, baseline, spikes, name=None, pcolor_kwargs={}):
    """
    A report for the results of cleaning.despike.

    The function creates a figure object of 3 subplots containing a pcolormesh
    of the original data, the despiked data and the spikes calculated from the
    cleaning.despike function.

    Parameters
    ----------
    dives: numpy.ndarray or pandas.Series
        The dive count (round is down dives, 0.5 is up dives)
    depth: numpy.ndarray or pandas.Series
        The depth array in metres
    raw: numpy.ndarray or pandas.Series
        Array of data variable for cleaning to be performed on.
    baseline: numpy.ndarray or pandas.Series
        The baseline from which outliers are determined.
    spikes: numpy.ndarray or pandas.Series
        Spikes are the residual of [measurements - baseline].
    name: str
        String name for the figure header, if None will try to get a pandas.
        Series.name from raw.
    pcolor_kwargs: dict
        A dictionary of keyword arguements passed to pcolormesh.


    Returns
    -------
    figure: object
        Creates a figure object with 3 subplots containing a pcolormesh of the
        original data, the despiked data and the spikes calculated from the
        cleaning.despike function.

    """
    from matplotlib.pyplot import cm, subplots
    from numpy import array, isnan, ma, nanpercentile

    from . import plot

    if name is None:
        name = "Variable" if not hasattr(raw, "name") else raw.name

    x = array(dives)
    y = array(depth)
    z = [array(raw), ma.masked_array(baseline), array(spikes)]

    fig, ax = subplots(3, 1, figsize=[10, 11], dpi=90)
    title = "{}\nDespiking Report".format(name)

    bmin, bmax = nanpercentile(z[1].data, [2, 98])
    smin, smax = nanpercentile(z[2].data, [2, 98])
    props = dict(cmap=cm.Spectral_r)
    props.update(pcolor_kwargs)

    im = []
    im += (plot.pcolormesh(x, y, z[0], ax=ax[0], vmin=bmin, vmax=bmax, **props),)
    im += (plot.pcolormesh(x, y, z[1], ax=ax[1], vmin=bmin, vmax=bmax, **props),)
    im += (plot.pcolormesh(x, y, z[2], ax=ax[2], vmin=smin, vmax=smax, **props),)

    for i in range(0, 3):
        ymax = y[~isnan(baseline)].max()
        ax[i].set_ylim(ymax, 0)
        ax[i].set_xlim(x.min(), x.max())
        ax[i].set_ylabel("Depth (m)")
        if i != 2:
            ax[i].set_xticklabels([])
        else:
            ax[i].set_xlabel("Dive number")
        ax[i].cb.set_label("Units")

    ax[0].set_title("Original")
    ax[1].set_title("Despiked")
    ax[2].set_title("Spikes")

    fig.tight_layout()
    fig.text(0.47, 1.02, title, va="center", ha="center", size=14)

    p0 = ax[0].get_position()
    p1 = ax[1].get_position()
    ax[0].set_position([p0.x0, p0.y0, p1.width, p0.height])

    return fig


def rolling_window(var, func, window):
    """
    A rolling window function that is nan-resiliant

    Parameters
    ----------
    arr:array, dtype=float, shape=[n, ]
        array that you want to pass the rolling window over
    func : callable
        an aggregating function. e.g. mean, std, median
    window : int
        the size of the rolling window that will be applied

    Returns
    -------
    arr : array, dtype=float, shape=[n, ]
        the same as the input array, but the rolling window has been applied
    """
    from numpy import array, nan, ndarray, r_

    n = window
    # create an empty 2D array with shape (window, len(arr))
    arr = array(var)
    mat = ndarray([n, len(arr) - n]) * nan
    # create a vector for each window
    for i in range(n):
        mat[i, :] = arr[i : i - n]
    # get the mean or meidan or any other function of the matrix
    out = func(mat, axis=0)

    # the array will be shorter than the original
    # pad the output with the rolling average of the values left out
    i0 = n // 2
    i1 = n - i0
    seg0 = array([func(arr[: i + 1]) for i in range(i0)])
    seg1 = array([func(arr[-i - 1 :]) for i in range(i1)])
    rolwin = r_[seg0, out, seg1]

    rolwin = transfer_nc_attrs(getframe(), var, rolwin, "_rollwin")

    return rolwin


def savitzky_golay(var, window_size, order, deriv=0, rate=1, interpolate=True):
    """
    Smooth (and optionally differentiate) data with a Savitzky-Golay filter.

    The Savitzky-Golay filter removes high frequency noise from data [1]_.
    It has the advantage of preserving the original shape and features of the
    signal better than other types of filtering approaches, such as moving
    averages techniques. By default, nans in the array are interpolated with a
    limit set to the window size of the dataset before smoothing. The nans are
    inserted back into the dataset after the convolution. This limits the loss
    of data over blocks where there are nans. This can be switched off with the
    `interpolate` keyword arguement.

    Parameters
    ----------
    var : array, dtype=float, shape=[n, ]
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv : int
        the order of the derivative to compute (default = 0 means only
        smoothing)
    interpolate : bool=True
        By default, nans in the array are interpolated with a limit set to
        the window size of the dataset before smoothing. The nans are
        inserted back into the dataset after the convolution. This limits
        the loss of data over blocks where there are nans. This can be
        switched off with the `interpolate` keyword arguement.

    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).

    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point [2]_.

    Examples
    --------
    >>> t = linspace(-4, 4, 500)
        y = exp( -t**2 ) + random.normal(0, 0.05, t.shape)
        ysg = savitzky_golay(y, window_size=31, order=4)
        import matplotlib.pyplot as plt
        plt.plot(t, y, label='Noisy signal')
        plt.plot(t, exp(-t**2), 'k', lw=1.5, label='Original signal')
        plt.plot(t, ysg, 'r', label='Filtered signal')
        plt.legend()
        plt.show()

    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    from math import factorial

    from numpy import abs, array, concatenate, convolve, isnan, linalg, mat, nan
    from pandas import Series

    # sorting out window stuff
    arr = array(var)
    try:
        window_size = abs(int(window_size))
        order = abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomial order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2

    # allow to interpolate for the window size
    if interpolate:
        ser = Series(arr).interpolate()
        y = array(ser)
    else:
        y = array(arr)

    # precompute coefficients
    b = mat(
        [[k**i for i in order_range] for k in range(-half_window, half_window + 1)]
    )
    m = linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - abs(y[1 : half_window + 1][::-1] - y[0])
    lastvals = y[-1] + abs(y[-half_window - 1 : -1][::-1] - y[-1])
    y = concatenate((firstvals, y, lastvals))

    savgol = convolve(m[::-1], y, mode="valid")

    oldnans = isnan(arr)
    savgol[oldnans] = nan

    savgol = transfer_nc_attrs(getframe(), var, savgol, "_savgolay")

    return savgol
