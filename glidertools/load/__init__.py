#!/usr/bin/env python
from __future__ import (print_function as _pf,
                        unicode_literals as _ul,
                        absolute_import as _ai)


from .slocum import slocum_geomar_matfile
from .seaglider import load_multiple_vars as seaglider_basestation_netCDFs
from .seaglider import show_variables as seaglider_show_variables
from .ego import load_mission_nc as ego_mission_netCDF
