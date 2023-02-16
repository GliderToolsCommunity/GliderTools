#!/usr/bin/env python
from __future__ import absolute_import as _ai
from __future__ import print_function as _pf
from __future__ import unicode_literals as _ul

import warnings

from inspect import currentframe as getframe

import numpy as np

from .helpers import GliderToolsWarning, transfer_nc_attrs


try:
    _gsw_avail = True
    from gsw import alpha as alpha_thermal
    from gsw import beta as beta_saline
except ImportError:
    _gsw_avail = False
    from seawater import alpha as alpha_thermal  # noqa: F401
    from seawater import beta as beta_saline  # noqa: F401

    message = (
        "'gsw' could not be imported (Python 2.x is not compatible "
        "with 'gsw'). Reverting to 'seawater'. You will not be able to "
        "calculate brunt_vaisala and potential_density will not use TEOS-10."
    )
    warnings.warn(message, category=GliderToolsWarning)


def mixed_layer_depth(ds, variable, thresh=0.01, ref_depth=10, return_as_mask=False):
    """
    Calculates the MLD for ungridded glider array.

    You can provide density or temperature.
    The default threshold is set for density (0.01).

    Parameters
    ----------
    ds : xarray.Dataset Glider dataset
    variable : str
         variable that will be used for the threshold criteria
    thresh : float=0.01 threshold for difference of variable
    ref_depth : float=10 reference depth for difference

    Return
    ------
    mld : array
        will be an array of depths the length of the
        number of unique dives.
    """
    ds = ds.reset_coords().to_pandas().set_index("dives")
    mld = (
        ds[[variable, "depth"]]
        .groupby("dives")
        .apply(mld_profile, variable, thresh, ref_depth)
    )

    if return_as_mask:
        return np.concatenate([el for el in mld])
    else:
        return mld


def mld_profile(df, variable, thresh, ref_depth, mask=False):
    df = df.dropna(subset=[variable, "depth"])
    if len(df) == 0:
        mld = np.nan
    elif np.nanmin(np.abs(df.depth.values - ref_depth)) > 5:
        message = """no observations within 5 m of ref_depth for dive {}
                """.format(
            df.index[0]
        )
        warnings.warn(message, category=GliderToolsWarning)
        mld = np.nan
    else:
        direction = 1 if np.unique(df.index % 1 == 0) else -1
        # create arrays in order of increasing depth
        var_arr = df[variable].values[:: int(direction)]
        depth = df.depth.values[:: int(direction)]
        # get index closest to ref_depth
        i = np.nanargmin(np.abs(depth - ref_depth))
        # create difference array for threshold variable
        dd = var_arr - var_arr[i]
        # mask out all values that are shallower then ref_depth
        dd[depth < ref_depth] = np.nan
        # get all values in difference array within treshold range
        mixed = dd[abs(dd) > thresh]
        if len(mixed) > 0:
            idx_mld = np.argmax(abs(dd) > thresh)
            mld = depth[idx_mld]
        else:
            mld = np.nan
            message = """threshold criterion never true (all mixed or shallow
            profile) for profile {}""".format(
                df.index[0]
            )
            warnings.warn(message, category=GliderToolsWarning)
    if mask:
        return depth <= mld
    else:
        return mld


def potential_density(salt_PSU, temp_C, pres_db, lat, lon, pres_ref=0):
    """
    Calculate density from glider measurements of salinity and temperature.

    The Basestation calculates density from absolute salinity and potential
    temperature. This function is a wrapper for this functionality, where
    potential temperature and absolute salinity are calculated first.
    Note that a reference pressure of 0 is used by default.

    Parameters
    ----------
    salt_PSU : array, dtype=float, shape=[n, ]
        practical salinty
    temp_C : array, dtype=float, shape=[n, ]
    temperature in deg C
    pres_db : array, dtype=float, shape=[n, ]
        pressure in decibar
    lat : array, dtype=float, shape=[n, ]
        latitude in degrees north
    lon : array, dtype=float, shape=[n, ]
        longitude in degrees east

    Returns
    -------
    potential_density : array, dtype=float, shape=[n, ]


    Note
    ----
    Using seawater.dens does not yield the same results as this function. We
    get very close results to what the SeaGlider Basestation returns with this
    function. The difference of this function with the basestation is on
    average ~ 0.003 kg/m3
    """

    try:
        import gsw

        salt_abs = gsw.SA_from_SP(salt_PSU, pres_db, lon, lat)
        pot_dens = gsw.pot_rho_t_exact(salt_abs, temp_C, pres_db, pres_ref)
    except ImportError:
        import seawater as sw

        pot_dens = sw.pden(salt_PSU, temp_C, pres_db, pres_ref)

    pot_dens = transfer_nc_attrs(
        getframe(),
        temp_C,
        pot_dens,
        "potential_density",
        units="kg/m3",
        comment="",
        standard_name="potential_density",
    )
    return pot_dens


if _gsw_avail:

    def brunt_vaisala(salt, temp, pres, lat=None):
        r"""
        Calculate the square of the buoyancy frequency.

        This is a copy from GSW package, with the exception that
        the array maintains the same shape as the input. Note that
        it only works on ungridded data at the moment.

        .. math::

        N^{2} = \frac{-g}{\sigma_{\theta}} \frac{d\sigma_{\theta}}{dz}

        Parameters
        ----------
        SA : array-like
            Absolute Salinity, g/kg
        CT : array-like
            Conservative Temperature (ITS-90), degrees C
        p : array-like
            Sea pressure (absolute pressure minus 10.1325 dbar), dbar
        lat : array-like, 1-D, optional
            Latitude, degrees.
        axis : int, optional
            The dimension along which pressure increases.

        Returns
        -------
        N2 : array
            Buoyancy frequency-squared at pressure midpoints, 1/s.
            The shape along the pressure axis dimension is one
            less than that of the inputs.
        """

        from gsw import Nsquared
        from numpy import nan, r_

        def pad_nan(a):
            return r_[a, nan]

        n2 = pad_nan(Nsquared(salt, temp, pres)[0])

        n2 = transfer_nc_attrs(
            getframe(),
            temp,
            n2,
            "N_squared",
            units="1/s2",
            comment="",
            standard_name="brunt_vaisala_freq",
        )

        return n2

    # compute spice
    def spice0(salt_PSU, temp_C, pres_db, lat, lon):
        """
        Calculate spiciness from glider measurements of salinity and temperature.

        Parameters
        ----------
        salt_PSU : array, dtype=float, shape=[n, ]
            practical salinty
        temp_C : array, dtype=float, shape=[n, ]
        temperature in deg C
        pres_db : array, dtype=float, shape=[n, ]
            pressure in decibar
        lat : array, dtype=float, shape=[n, ]
            latitude in degrees north
        lon : array, dtype=float, shape=[n, ]
            longitude in degrees east

        Returns
        -------
        potential_density : array, dtype=float, shape=[n, ]


        Note
        ----
        Using seawater.dens does not yield the same results as this function. We
        get very close results to what the SeaGlider Basestation returns with this
        function. The difference of this function with the basestation is on
        average ~ 0.003 kg/m3
        """
        import gsw

        salt_abs = gsw.SA_from_SP(salt_PSU, pres_db, lon, lat)
        cons_temp = gsw.CT_from_t(salt_abs, temp_C, pres_db)

        spice0 = gsw.spiciness0(salt_abs, cons_temp)

        spice0 = transfer_nc_attrs(
            getframe(),
            temp_C,
            spice0,
            "spiciness0",
            units=" ",
            comment="",
            standard_name="spiciness0",
        )
        return spice0
