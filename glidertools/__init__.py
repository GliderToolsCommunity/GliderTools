#!/usr/bin/env python
from __future__ import absolute_import as _ai
from __future__ import print_function as _pf
from __future__ import unicode_literals as _ul

import warnings as _warnings

from setuptools_scm import get_version

from . import (calibration, cleaning, flo_functions, load, mapping, optics,
               physics, utils)
from .mapping import grid_data, interp_obj
from .plot import logo as make_logo
from .plot import plot_functions as plot
from .processing import *

__version__ = get_version(root='..', relative_to=__file__)
del get_version

_warnings.filterwarnings('ignore', category=RuntimeWarning)
