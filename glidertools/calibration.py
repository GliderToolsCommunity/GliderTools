#!/usr/bin/env python
from __future__ import absolute_import as _ai
from __future__ import print_function as _pf
from __future__ import unicode_literals as _ul

from inspect import currentframe as getframe

import numpy as _np

from .helpers import transfer_nc_attrs


def bottle_matchup(
    gld_dives,
    gld_depth,
    gld_time,
    btl_depth,
    btl_time,
    btl_values,
    min_depth_diff_metres=5,
    min_time_diff_minutes=120,
):
    """
    Performs a matchup between glider and bottle samples based on time and
    depth (or density).

    Parameters
    ----------
    gld_depth : np.array, dtype=float
        glider depth at time of measurement
    gld_dives : np.array, dtype=float
        dive index of the glider (given by glider toolbox)
    gld_time : np.array, dtype=datetime64
        glider time that will be used as primary indexing variable
    btl_time: np.array, dtype=datetime64
        in-situ bottle sample's time
    btl_depth : np.array, dtype=float
        depth of in-situ sample
    btl_values : np.array, dtype=float
        the value that will be interpolated onto the glider time and
        depth coordinates (time, depth/dens)
    min_depth_diff_metres : float, default=5
        the minimum allowable depth difference
    min_time_diff_minutes : float, default=120
        the minimum allowable time difference between bottles and glider

    Returns
    -------
    array : float
        Returns the bottle values in the format of the glider
        i.e. the length of the output will be the same as gld_*

    """
    from pandas import Series

    # metadata preservation
    var = gld_depth.copy()
    if isinstance(btl_values, Series):
        var_name = btl_values.name + "_bottle_matchups"
    else:
        var_name = "bottle_matchups"

    # make all input variables np.arrays
    args = gld_time, gld_depth, gld_dives, btl_time, btl_depth, btl_values
    gld_time, gld_depth, gld_dives, btl_time, btl_depth, btl_values = map(
        _np.array, args
    )

    # create a blank array that matches glider data
    # (placeholder for calibration bottle values)
    gld_cal = _np.ones_like(gld_depth) * _np.nan

    # loop through each ship based CTD station
    stations = _np.unique(btl_time)
    for c, t in enumerate(stations):
        # index of station from ship CTD
        btl_idx = t == btl_time
        # number of samples per station
        btl_num = btl_idx.sum()

        # string representation of station time
        t_str = str(t.astype("datetime64[m]")).replace("T", " ")
        t_dif = abs(gld_time - t).astype("timedelta64[m]").astype(float)

        # loop through depths for the station
        if t_dif.min() < min_time_diff_minutes:
            # index of dive where minimum difference occurs
            i = _np.where(gld_dives[_np.nanargmin(t_dif)] == gld_dives)[0]
            n_depths = 0
            for depth in btl_depth[btl_idx]:
                # an index for bottle where depth and station match
                j = btl_idx & (depth == btl_depth)
                # depth difference for glider profile
                d_dif = abs(gld_depth - depth)[i]
                # only match depth if diff is less than given threshold
                if _np.nanmin(d_dif) < min_depth_diff_metres:
                    # index of min diff for this dive
                    k = i[_np.nanargmin(d_dif)]
                    # assign the bottle values to the calibration output
                    gld_cal[k] = btl_values[j]
                    n_depths += 1
            print(
                (
                    "[stn {}/{}] SUCCESS: {} ({} of {} samples) match-up "
                    "within {} minutes"
                ).format(c, stations.size, t_str, n_depths, btl_num, t_dif.min())
            )
        else:
            print(
                (
                    "[stn {}/{}]  FAILED: {} Couldn't find samples within "
                    "constraints"
                ).format(c, stations.size, t_str)
            )

    attrs = dict(units="", positive="", comment="", standard_name="", axis="")
    gld_cal = transfer_nc_attrs(getframe(), var, gld_cal, var_name, **attrs)

    return gld_cal


def model_metrics(x, y, model):
    from numpy import array
    from sklearn import metrics

    x = array(x).reshape(-1, 1)
    y = array(y)

    y_hat = model.predict(x).squeeze()
    ol = (
        model.outliers_
        if hasattr(model, "outliers_")
        else _np.zeros_like(y).astype(bool)
    )

    # formula = '$f(x) = {:.2g}x + {:.2g}$'.format(
    #     model.coef_[0], model.intercept_
    # )

    # metrics calculation
    out = dict(
        model_type=model.__class__.__name__,
        model_slope=model.coef_[0],
        model_intercept=model.intercept_,
    )

    params = {
        "param_" + key: value for key, value in model.__class__().get_params().items()
    }

    results = dict(
        r2_all=metrics.r2_score(y, y_hat),
        r2_robust=metrics.r2_score(y[~ol], y_hat[~ol]),
        rmse_all=metrics.mean_squared_error(y, y_hat) ** 0.5,
        rmse_robust=metrics.mean_squared_error(y[~ol], y_hat[~ol]) ** 0.5,
    )

    out.update(params)
    out.update(results)

    return out


def model_figs(bottle_data, glider_data, model, ax=None):
    """
    Creates the figure for a linear model fit.

    Parameters
    ----------
    bottle_data : np.array, shape=[m, ]
        bottle data with the number of matched bottle/glider samples
    glider_data : np.array, shape[m, ]
        glider data with the number of matched bottle/glider samples
    model : sklearn.linear_model object
        a fitted model that you want to test.

    Returns
    -------
    figure axes : matplotlib.Axes
        A figure showing the fit of the
    """

    from matplotlib.offsetbox import AnchoredText
    from matplotlib.pyplot import subplots
    from numpy import array, isnan, linspace, nanmax, nanmin
    from sklearn import metrics

    y = array(bottle_data)
    x = array(glider_data).reshape(-1, 1)

    assert not any(isnan(x)), "There are nans in glider_data"
    assert not any(isnan(y)), "There are nans in bottle_data"
    assert x.size == y.size, "glider_data and bottle_data are not the same size"
    assert (
        x.size == model.outliers_.size
    ), "model.outliers_ is a different size to bottle_data"

    xf = linspace(nanmin(x), nanmax(x), 100).reshape(-1, 1)
    y_hat = model.predict(x).squeeze()
    ol = (
        model.outliers_
        if hasattr(model, "outliers_")
        else _np.zeros_like(y).astype(bool)
    )
    formula = "$f(x) = {:.2g}x + {:.2g}$".format(model.coef_[0], model.intercept_)
    formula = formula if not formula.endswith("+ 0$") else formula[:-5] + "$"

    print(x.shape, xf.shape)
    # PLOTTING FROM HERE ON #############
    if ax is None:
        _, ax = subplots(1, 1, figsize=[6, 5], dpi=120)
    ax.plot(x, y, "o", c="k", zorder=99, label="Samples ({})".format(x.size))[0]
    ax.plot(xf, model.predict(xf), c="#AAAAAA", label="{}".format(formula))
    ax.plot(
        x[ol],
        y[ol],
        "ow",
        visible=ol.any(),
        mew=1,
        mec="k",
        zorder=100,
        label="Outliers ({})".format(ol.sum()),
    )
    ax.legend(fontsize=10, loc="upper left")

    # Additional info about the model displayed from here on
    params = model.get_params()
    rcModel = model.__class__().get_params()
    for key in rcModel:
        if rcModel[key] == params[key]:
            params.pop(key)

    # metrics calculation
    r2_all = metrics.r2_score(y, y_hat)
    r2_robust = metrics.r2_score(y[~ol], y_hat[~ol])
    rmse_all = metrics.mean_squared_error(y, y_hat) ** 0.5
    rmse_robust = metrics.mean_squared_error(y[~ol], y_hat[~ol]) ** 0.5

    # string formatting
    m_name = "Huber Regresion"
    r2_str = "$r^2$ score: {:.2g} ({:.2g})\n"
    rmse_str = "RMSE: {:.2g} ({:.2g})"
    placeholder = "{}: {}\n"

    # formatting the strings to be displayed
    params_str = "{} Params\n".format(m_name)
    params_str += "".join([placeholder.format(key, params[key]) for key in params])
    params_str += "\nResults (robust)\n"
    params_str += r2_str.format(r2_all, r2_robust)
    params_str += rmse_str.format(rmse_all, rmse_robust)

    # placing the text box
    anchored_text = AnchoredText(
        params_str, loc=4, prop=dict(size=10, family="monospace"), frameon=True
    )
    anchored_text.patch.set_boxstyle("round, pad=0.3, rounding_size=0.2")
    anchored_text.patch.set_linewidth(0.2)
    ax.add_artist(anchored_text)

    # axes labelling
    ax.set_ylabel("Bottle sample")
    ax.set_xlabel("Glider sample")
    ax.set_title("Calibration curve using {}".format(m_name))

    return ax


def robust_linear_fit(
    gld_var, gld_var_cal, interpolate_limit=3, return_figures=True, **kwargs
):
    """
    Perform a robust linear regression using a Huber Loss Function to remove
    outliers. Returns a model object that behaves like a scikit-learn model
    object with a model.predict method.

    Parameters
    ----------
    gld_var : np.array, shape=[n, ]
        glider variable
    gld_var_cal : np.array, shape=[n, ]
        bottle variable on glider indicies
    fit_intercept : bool, default=False
        forces 0 intercept if False
    return_figures : bool, default=True
        create figure with metrics
    interpolate_limit : int, default=3
        glider data may have missing points. The glider data is thus
        interpolated to ensure that as many bottle samples as possible have a
        match-up with the glider.
    **kwargs : keyword=value pairs
        will be passed to the Huber Loss regression to adjust regression

    Returns
    -------
    model : sklearn.linear_model
        A fitted model. Use model.predict(glider_var) to create the calibrated
        output.
    """

    from pandas import Series
    from sklearn import linear_model

    from .helpers import GliderToolsError

    # make all input arguments numpy arrays
    args = gld_var, gld_var_cal
    gld_var, gld_var_cal = map(_np.array, args)

    if _np.isnan(gld_var_cal).all():
        raise GliderToolsError("There are no matches in your bottle data")

    gld_var = Series(gld_var).interpolate(limit=interpolate_limit).values

    # get bottle and glider values for the variables
    i = ~_np.isnan(gld_var_cal) & ~_np.isnan(gld_var)
    y = gld_var_cal[i]
    x = gld_var[i][:, None]

    if "fit_intercept" not in kwargs:
        kwargs["fit_intercept"] = False
    model = linear_model.HuberRegressor(**kwargs)
    model.fit(x, y)

    if return_figures:
        model_figs(x, y, model)

    model._predict = model.predict

    def predict(self, x):
        """
        A wrapper around the normal predict function that takes
        nans into account. An extra dimension is also added if needed.
        """
        from xarray import DataArray

        var = x.copy()
        x = _np.array(x)
        out = _np.ndarray(x.size) * _np.NaN
        i = ~_np.isnan(x)
        x = x[i].reshape(-1, 1)
        out[i.squeeze()] = self._predict(x).squeeze()

        out = transfer_nc_attrs(getframe(), var, out, "_calibrated")
        if hasattr(self, "info") & isinstance(out, DataArray):
            out.attrs["model_info"] = str(self.info)

        return out

    model.predict = predict.__get__(model, linear_model.HuberRegressor)
    model.info = model_metrics(x, y, model)

    return model
