#!/usr/bin/env python

import warnings as _warnings

from . import (  # NOQA
    calibration,
    cleaning,
    flo_functions,
    load,
    mapping,
    optics,
    physics,
    utils,
)
from .helpers import package_version
from .mapping import grid_data, interp_obj
from .plot import logo as make_logo
from .plot import plot_functions as plot
from .processing import *


# from importlib.metadata import version, PackageNotFoundError
# from pkg_resources import DistributionNotFound, get_distribution


__version__ = package_version()
_warnings.filterwarnings("ignore", category=RuntimeWarning)
