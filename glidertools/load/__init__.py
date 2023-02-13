#!/usr/bin/env python
from __future__ import absolute_import as _ai
from __future__ import print_function as _pf
from __future__ import unicode_literals as _ul

from .ego import load_mission_nc as ego_mission_netCDF
from .seaglider import load_multiple_vars as seaglider_basestation_netCDFs
from .seaglider import show_variables as seaglider_show_variables
from .slocum import slocum_geomar_matfile
