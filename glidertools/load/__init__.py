#!/usr/bin/env python

from .ego import load_mission_nc as ego_mission_netCDF
from .seaglider import load_multiple_vars as seaglider_basestation_netCDFs
from .seaglider import show_variables as seaglider_show_variables
from .slocum import slocum_geomar_matfile
from .voto_seaexplorer import (
    voto_concat_datasets,
    voto_seaexplorer_dataset,
    voto_seaexplorer_nc,
)
