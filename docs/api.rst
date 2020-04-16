API Reference
=============

The API reference is automatically generated from the function docstrings in
the GliderTools package. Refer to the examples in the sidebar for reference on
how to use the functions.

Loading Data
------------
.. currentmodule:: glidertools
.. autosummary::
   :toctree: ./_generated/

   load.seaglider_basestation_netCDFs
   load.seaglider_show_variables
   load.ego_mission_netCDF
   load.slocum_geomar_matfile


High level processing
---------------------
.. currentmodule:: glidertools
.. autosummary::
   :toctree: ./_generated/

   processing.calc_physics
   processing.calc_oxygen
   processing.calc_backscatter
   processing.calc_fluorescence
   processing.calc_par


Cleaning
--------
.. currentmodule:: glidertools
.. autosummary::
   :toctree: ./_generated/

   cleaning.outlier_bounds_std
   cleaning.outlier_bounds_iqr
   cleaning.horizontal_diff_outliers
   cleaning.mask_bad_dive_fraction
   cleaning.data_density_filter
   cleaning.despike
   cleaning.despiking_report
   cleaning.rolling_window
   cleaning.savitzky_golay


Physics
-------
.. currentmodule:: glidertools
.. autosummary::
   :toctree: ./_generated/

   physics.mixed_layer_depth
   physics.potential_density
   physics.brunt_vaisala



Optics
------
.. currentmodule:: glidertools
.. autosummary::
   :toctree: ./_generated/

   optics.find_bad_profiles
   optics.par_dark_count
   optics.backscatter_dark_count
   optics.fluorescence_dark_count
   optics.par_scaling
   optics.par_fill_surface
   optics.photic_depth
   optics.sunset_sunrise
   optics.quenching_correction
   optics.quenching_report


Calibration
-----------
.. currentmodule:: glidertools
.. autosummary::
   :toctree: ./_generated/

   calibration.bottle_matchup
   calibration.model_figs
   calibration.robust_linear_fit

Gridding and Interpolation
--------------------------
.. currentmodule:: glidertools
.. autosummary::
   :toctree: ./_generated/

   mapping.interp_obj
   mapping.grid_data
   mapping.variogram



Plotting
--------
.. currentmodule:: glidertools
.. autosummary::
   :toctree: ./_generated/

   plot.plot_functions


General Utilities
-----------------
.. currentmodule:: glidertools
.. autosummary::
   :toctree: ./_generated/

   utils.time_average_per_dive
   utils.mask_to_depth_array
   utils.merge_dimensions
   utils.calc_glider_vert_velocity
   utils.calc_dive_phase
   utils.calc_dive_number
   utils.dive_phase_to_number
   utils.distance
