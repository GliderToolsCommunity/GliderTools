#!/usr/bin/env python
from __future__ import absolute_import as _ai
from __future__ import print_function as _pf
from __future__ import unicode_literals as _ul

import numpy as np


def _process_2D_plot_args(args, gridding_dz=1):
    """
    Processes input to the plotting class functions. Allows plots to receive
    one (2D) or three (1D) input arguements.
    """
    from numpy import array, ma, nan, ndarray
    from pandas import DataFrame, Series
    from xarray import DataArray

    from .helpers import GliderToolsError
    from .mapping import grid_data

    name = ""
    if len(args) == 3:
        x = array(args[0])
        y = array(args[1]).astype(float)
        z = args[2].copy()

        if isinstance(z, ma.MaskedArray):
            z[z.mask] = nan
        elif isinstance(z, DataArray):
            name = z.name if z.name is not None else ""
            unit = " [{}]".format(z.units) if "units" in z.attrs else ""
            name = name + unit
            z = ma.masked_invalid(array(z)).astype(float)
        else:
            z = ma.masked_invalid(array(z)).astype(float)

        if (x.size == y.size) & (len(z.shape) == 1):
            df = grid_data(x, y, z, interp_lim=6, verbose=False, return_xarray=False)
            x = df.columns
            y = df.index
            z = ma.masked_invalid(df.values)

        return x, y, z, name

    elif len(args) == 1:
        z = args[0]
        if isinstance(z, DataArray):
            name = z.name if z.name is not None else ""
            unit = " [{}]".format(z.units) if "units" in z.attrs else ""
            name = name + unit
            if z.ndim == 1:
                raise GliderToolsError(
                    "Please provide gridded DataArray or x and y coordinates"
                )
            elif z.ndim == 2:
                z = z.to_series().unstack()
            elif z.ndim > 2:
                raise GliderToolsError(
                    "GliderTools plotting currently only supports 2 "
                    "dimensional plotting"
                )
        elif isinstance(z, (ndarray, Series)):
            if z.ndim == 2:
                z = DataFrame(z)
            else:
                raise IndexError("The input must be a 2D DataFrame or ndarray")

        x = z.columns.values
        y = z.index.values
        z = ma.masked_invalid(z.values).astype(float)

        return x, y, z, name


class plot_functions(object):
    """
    Plot data (gridded or not) as a section and more.

    This function provides several options to plot data as a section. The
    default action when called is to plot data as a ``pcolormesh`` section.

    See the individual method help for more information about each plotting
    method.

    Parameters
    ----------
    args : array_like
        - same length x, y, z. Will be gridded with depth of 1 meter.
        - x(m), y(n), z(n, m) arrays
        - z DataFrame where indicies are depth and columns are dives
        - z DataArray where dim0 is dives and dim1 is depth, or
            contains information about time and depth axes
    kwargs : key-value pairs
        - ax - give an axes to the plotting function
        - robust - use the 0.5 and 99.5 percentile to set color limits
        - gridding_dz - gridding depth [default 1]

    """

    @staticmethod
    def __new__(*args, **kwargs):

        if len(args) > 1:
            args = args[1:]
        return plot_functions.pcolormesh(*args, **kwargs)

    @staticmethod
    def pcolormesh(*args, **kwargs):
        """
        Plot a section plot of the dives with x-time and y-depth and
        z-variable. The data can be linearly interpolated to fill missing
        depth values. The number of points to interpolate can be set with
        interpolate_dist.

        Parameters
        ----------
        args : array_like
            - same length x, y, z. Will be gridded with depth of 1 meter.
            - x(m), y(n), z(n, m) arrays
            - z DataFrame where indicies are depth and columns are dives
            - z DataArray where dim0 is dives and dim1 is depth
        kwargs : key-value pairs
            - ax - give an axes to the plotting function
            - robust - use the 0.5 and 99.5 percentile to set color limits
            - gridding_dz - gridding depth [default 1]

        """
        from datetime import datetime

        from matplotlib.pyplot import colorbar, subplots
        from numpy import datetime64, nanpercentile

        ax = kwargs.pop("ax", None)
        robust = kwargs.pop("robust", True)
        gridding_dz = kwargs.pop("gridding_dz", 1)

        # set default shading and rasterized for pcolormesh (can be overriden by user)
        kwargs.setdefault("shading", "nearest")
        kwargs.setdefault("rasterized", "True")

        x, y, z, name = _process_2D_plot_args(args, gridding_dz=gridding_dz)
        m = (~z.mask).any(axis=1)

        x_time = isinstance(x[0], (datetime, datetime64))

        if robust & (("vmin" not in kwargs) | ("vmax" not in kwargs)):
            kwargs["vmin"] = nanpercentile(z.data, 0.5)
            kwargs["vmax"] = nanpercentile(z.data, 99.5)

        if ax is None:
            fig, ax = subplots(1, 1, figsize=[9, 3.5], dpi=90)
        else:
            fig = ax.get_figure()

        im = ax.pcolormesh(x, y, z, **kwargs)
        ax.cb = colorbar(mappable=im, pad=0.02, ax=ax, fraction=0.05)
        ylim = nanpercentile(y[m], [100, 0])
        ax.set_ylim(ylim)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylabel("Depth (m)")
        ax.set_xlabel("Date" if x_time else "Dives")
        if len(name) < 30:
            ax.cb.set_label(name)
        else:
            ax.set_title(name)

        [tick.set_rotation(45) for tick in ax.get_xticklabels()]
        fig.tight_layout()

        return ax

    @staticmethod
    def contourf(*args, **kwargs):
        """
        Plot a section plot of the dives with x-time and y-depth and
        z-variable. The data can be linearly interpolated to fill missing
        depth values. The number of points to interpolate can be set with
        interpolate_dist.

        Parameters
        ----------
        args :
            - same length x, y, z. Will be gridded with depth of 1 meter.
            - x(m), y(n), z(n, m) arrays
            - z DataFrame where indicies are depth and columns are dives
            - z DataArray where dim0 is dives and dim1 is depth
        kwargs :
            - ax : give an axes to the plotting function
            - robust : use the 0.5 and 99.5 percentile to set color limits
            - gridding_dz : gridding depth [default 1]
            - can also be anything that gets passed to plt.pcolormesh.

        Returns
        -------
        axes
        """

        from datetime import datetime

        from matplotlib.pyplot import colorbar, subplots
        from numpy import datetime64, nanpercentile

        ax = kwargs.pop("ax", None)
        robust = kwargs.pop("robust", False)
        gridding_dz = kwargs.pop("gridding_dz", 1)

        x, y, z, name = _process_2D_plot_args(args, gridding_dz=gridding_dz)

        x_time = isinstance(x[0], (datetime, datetime64))

        if robust & (("vmin" not in kwargs) | ("vmax" not in kwargs)):
            kwargs["vmin"] = nanpercentile(z[~z.mask], 0.5)
            kwargs["vmax"] = nanpercentile(z[~z.mask], 99.5)

        if ax is None:
            fig, ax = subplots(1, 1, figsize=[9, 3.5], dpi=90)
        else:
            fig = ax.get_figure()

        im = ax.contourf(x, y, z, **kwargs)
        ax.cb = colorbar(mappable=im, pad=0.02, ax=ax, fraction=0.05)
        ax.set_xlim(x.min(), x.max())
        m = (~z.mask).any(axis=1)
        ylim = nanpercentile(y[m], [100, 0])
        ax.set_ylim(ylim)
        ax.set_ylabel("Depth (m)")
        ax.set_xlabel("Date" if x_time else "Dives")
        ax.cb.set_label(name)

        [tick.set_rotation(45) for tick in ax.get_xticklabels()]
        fig.tight_layout()

        return ax

    @staticmethod
    def scatter(x, y, z, ax=None, robust=False, **kwargs):
        """
        Plot a scatter section plot of a small dataset (< 10 000 obs)

        Parameters
        ----------
        x : array, dtype=float, shape=[n, ]
            continuous horizontal variable (e.g. time, lat, lon)
        y : array, dtype=float, shape=[n, ]
            continous vertical variable (e.g. depth, density)
        z : array, dtype=float, shape=[n, ]
            ungridded data variable
        ax : matplotlib.axes
            a predefined set of axes to draw on
        robust : bool=False
            if True, uses the 0.5 and 99.5 percentile to set color limits
        kwargs : any key:values pair that gets passed to plt.pcolormesh.

        Returns
        -------
        axes

        Raises
        ------
        will ask if you want to continue if more than 10000 points
        """

        from datetime import datetime

        from matplotlib.pyplot import colorbar, subplots
        from numpy import array, datetime64, isnan, ma, nanmax, nanmin, nanpercentile

        z = ma.masked_invalid(z)
        m = ~(z.mask | isnan(y))
        z = z[m]
        x = array(x)[m]
        y = array(y)[m]

        if y.size >= 1e5:
            carry_on = input(
                "There are a large number of points to plot ({}). "
                "This will take a while to plot.\n"
                'Type "y" to continue or "n" to cancel.\n'.format(y.size)
            )
            if carry_on != "y":
                print("You have aborted the scatter plot")
                return None

        x_time = isinstance(x[0], (datetime, datetime64))

        if robust:
            kwargs["vmin"] = nanpercentile(z, 0.5)
            kwargs["vmax"] = nanpercentile(z, 99.5)

        if ax is None:
            fig, ax = subplots(1, 1, figsize=[9, 3.5], dpi=90)
        else:
            fig = ax.get_figure()
        im = ax.scatter(x, y, c=z, rasterized=True, **kwargs)

        ax.cb = colorbar(mappable=im, pad=0.02, ax=ax, fraction=0.05)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(nanmax(y), nanmin(y))
        ax.set_ylabel("Depth (m)")
        ax.set_xlabel("Date" if x_time else "Dives")

        [tick.set_rotation(45) for tick in ax.get_xticklabels()]
        fig.tight_layout()

        return ax

    @staticmethod
    def bin_size(depth, bins=None, ax=None, add_colorbar=True, **hist_kwargs):
        """
        Plots a 2D histogram of the depth sampling frequency.

        Profiling gliders will often sample at a lower frequency at depth to
        conserve battery. It is useful to know this frequency if you'd like to
        make more informed decisions about binning the data.

        Parameters
        ----------
        depth : array, dtype=float, shape=[n, ]
            the head-to-tail concatenated depth readings
        bins: [array, array]
            a user defined set of delta depth and depth bins. If
            unspecified then these bins are automatically chosen.
        hist_kwargs : key-value pairs
            passed to the 2D histogram function.

        Returns
        -------
        axes
        """
        from matplotlib.colors import LogNorm
        from matplotlib.pyplot import colorbar, subplots
        from numpy import abs, array, diff, isnan, nan, nanmedian, r_

        from .mapping import get_optimal_bins

        depth = array(depth)

        x = abs(diff(depth))
        y = depth[1:]
        m = ~(isnan(x) | isnan(y))
        x, y = x[m], y[m]

        if bins is None:
            binning_freq = 50
            ybins = get_optimal_bins(depth, binning_freq)[0]
            xbins = r_[nan, diff(ybins)]
            bins = binning_freq

        else:
            ybins = bins[1]
            xbins = []
            for k in range(len(ybins) - 1):
                d0 = ybins[k]
                d1 = ybins[k + 1]

                i = (y > d0) & (y < d1)
                xbins += (nanmedian(x[i]),)
            xbins = r_[nan, xbins]

        if ax is None:
            fig, ax = subplots(1, 1, figsize=[4, 6])

        im = ax.hist2d(x, y, bins=bins, norm=LogNorm(), rasterized=True, **hist_kwargs)[
            -1
        ]

        ax.plot(xbins, ybins, lw=4, ls="-", color="k", label="Bins")

        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_ylabel("Depth (m)")
        ax.set_xlabel(r"$\Delta$ Depth (m)")  # noqa: W605
        ax.legend(loc=0)

        if add_colorbar:
            cb = colorbar(mappable=im, ax=ax, fraction=0.1, pad=0.05)
            cb.set_label("Measurement count")

        return ax

    @staticmethod
    def section3D(
        dives,
        depth,
        x,
        y,
        variable,
        zmin=-1000,
        zmax=1,
        vmin=None,
        vmax=None,
        cmap=None,
        aspect_ratio_x=1.5,
        return_plot=True,
    ):
        """
        Returns an interactive 3D plot in an HTML page.

        Parameters
        ----------
        dives : array, dtype=float, shape=[n, ]
            timeseries of dive number (or can be pseudo discrete time)
        depth : array, dtype=float, shape=[n, ]
            head-to-tail concatenated depth readings
        x : array, dtype=float, shape=[n, ]
            the x-coordinate used in the plot (e.g. longitude, time)
        y : array, dtype=float, shape=[n, ]
            the y-coordinate used in the plot (e.g. latitude, time)
        variable : array, dtype=float, shape=[n, ]
            the variable to grid and plot (e.g. temperature salinity)
        zmin : int=-1000
            lower depth limit for the depth axis
        zmax : int=1
            upper depth limit for the depth axis
        vmin : float=None
            lower color limit of variable. Defaults to 1st percentile
        vmax : float=None
            upper color limit of variable. Defaults to 99th percentile
        cmap : cm.colormap=cm.Spectral_r
            colorbar used in the plot
        aspect_ratio : float=1.5
            the ratio of the plot [1.5] (best to use trail and error)

        Returns
        -------
        a plotly figure object that can be adjusted if needed

        Example
        -------
        >>> fig = gt.plot.section3D(df.dives, df.ctd_depth, df.longitude,
                                    df.latitude, df.temperature)
        """

        try:
            from plotly.offline import download_plotlyjs, plot  # noqa: F401
        except ImportError:
            raise ImportError("You need to install plotly for `section3D` to work")
        import numpy as np
        import plotly.graph_objs as go

        from matplotlib import cm
        from pandas import Series

        from .mapping import grid_data

        def matplotlib_to_plotly(cmap, pl_entries=255):
            if cmap is None:
                cmap = cm.Spectral_r
            h = 1.0 / (pl_entries - 1)
            pl_colorscale = []

            for k in range(pl_entries):
                C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
                pl_colorscale.append([k * h, "rgb" + str((C[0], C[1], C[2]))])

            return pl_colorscale

        d1 = depth.max()
        ds = 5
        d1 += ds

        if x.dtype.type == np.datetime64:
            x = x.values.astype(float)  # nanoseconds
        props = dict(bins=np.arange(0, d1, ds), return_xarray=False, verbose=False)
        gx = grid_data(dives, depth, x, **props).values
        gy = grid_data(dives, depth, y, **props).values
        gz = grid_data(dives, depth, depth, **props).values
        gf = grid_data(dives, depth, variable, **props)

        # color range
        lL = 0.01 if vmin is None else vmin
        uL = 0.99 if vmax is None else vmax
        c0, c1 = Series(variable).quantile([lL, uL]).values

        data = [
            go.Surface(
                z=-gz,
                y=gy,
                x=gx,
                name=variable.name.capitalize(),
                cmin=c0,
                cmax=c1,
                colorscale=matplotlib_to_plotly(cmap),
                colorbar=dict(title=variable.name.capitalize()),
                surfacecolor=gf.values,
                text=("c: " + gf.round(2).astype(str)).values,
                hoverinfo="x+y+z+text+name",
            ),
        ]

        layout = go.Layout(
            autosize=True,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=65, r=50, b=65, t=90),
            scene=dict(
                zaxis=dict(range=[zmin, zmax], title="Depth (m)"),
                xaxis=dict(title=x.name.capitalize()),
                yaxis=dict(title=y.name.capitalize()),
                aspectmode="manual",
                aspectratio=dict(y=1, x=aspect_ratio_x, z=0.5),
            ),
        )

        fig = go.Figure(data=data, layout=layout)

        if return_plot:
            plot(fig)
        return fig

    @staticmethod
    def save_figures_to_pdf(fig_list, pdf_name, **savefig_kwargs):
        """
        Saves a list of figure objects to a pdf.

        This function is useful if you'd like to create automatic QC reports in
        PDF format with a plot per page.

        Parameters
        ----------
        fig_list : list
            list of figure objects
        pdf_name : str
            path to save pdf to.
        savefig_kwargs : key-value pairs passed to ``Figure.savefig``
        """
        import matplotlib.backends.backend_pdf

        from matplotlib import pyplot as plt

        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)
        for fig in fig_list:  # will open an empty extra figure :(
            pdf.savefig(fig.number, dpi=120, **savefig_kwargs)
        pdf.close()
        plt.close("all")


class logo:
    @staticmethod
    def run(show_figures=False):
        import matplotlib.pyplot as plt

        x, y = logo.profile_dummy_data()

        fig1, ax1 = logo.logo_with_name()
        fig2, ax2 = logo.logo_wo_name()

        # props = dict(transparent=True, dpi=200, bbox_inches='tight')
        # fig1.savefig('./docs/img/logo_with_name.png', **props)
        # fig2.savefig('./docs/img/logo_wo_name.png', fc='#3a3a3a', **props)

        if show_figures:
            plt.show()

    @staticmethod
    def profile_dummy_data(n_zigzags=3):
        import pandas as pd
        import seaborn as sns

        sns.set_palette("Spectral", int(n_zigzags * 2))

        zigzags = np.r_[
            np.linspace(-1, -0.03, 100), np.linspace(-0.03, -1, 100)
        ].tolist() * (n_zigzags - 1)

        y = np.array(
            np.linspace(0, -1, 100).tolist()
            + zigzags
            + np.linspace(-1, 0, 100).tolist()
        )

        x = np.linspace(0, 5, y.size)

        # noise = np.random.normal(loc=0, scale=0.2, size=y.size)
        # wieght = np.linspace(1, 0, y.size)
        # x += noise * wieght

        x = pd.Series(x).rolling(10, center=True).mean()
        y = pd.Series(y).rolling(10, center=True).mean()

        return x, y

    @staticmethod
    def logo_with_name(n_zigzags=3, ax=None):
        import matplotlib.pyplot as plt

        x, y = logo.profile_dummy_data(n_zigzags)
        interval = int(x.size / (n_zigzags * 2))

        if ax is None:
            fig = plt.figure(figsize=[5, 1.4])
            ax = fig.add_axes([0, 0, 0.45, 1])

        for i in range(0, x.size, interval):
            j = i + interval
            ax.scatter(
                x[i:j],
                y[i:j],
                84,
                np.ones(interval) * i,
                cmap=plt.cm.Spectral_r,
                vmin=0,
                vmax=x.size,
            )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

        fig.text(
            0.45,
            0.46,
            "GLIDER\nTOOLS",
            weight=900,
            size=50,
            color="#606060",
            ha="left",
            va="center",
        )

        return fig, ax

    @staticmethod
    def logo_wo_name(n_zigzags=3, ax=None):
        import matplotlib.pyplot as plt

        x, y = logo.profile_dummy_data(n_zigzags)
        interval = int(x.size / (n_zigzags * 2))
        if ax is None:
            fig = plt.figure(figsize=[2.5, 2.5])
            ax = fig.add_axes([0, 0.2, 1, 0.6], facecolor="#3a3a3a")

        for i in range(0, x.size, interval):
            j = i + interval
            ax.scatter(
                x[i:j],
                y[i:j],
                84,
                np.ones(interval) * i,
                cmap=plt.cm.Spectral_r,
                vmin=0,
                vmax=x.size,
            )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

        c = "#3a3a3a"
        ax.set_fc(c)

        return fig, ax


if __name__ == "__main__":
    pass
    "fun people"
