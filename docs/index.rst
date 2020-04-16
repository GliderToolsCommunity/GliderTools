=====================================
Glider Tools: profile data processing
=====================================

Glider tools is a Python 3.6+ package designed to process data from the first level of processing to a science ready dataset.
The package is designed to easily import data to a standard column format:
``numpy.ndarray``, ``pandas.DataFrame`` or ``xarray.DataArray`` (we recommend
the latter which has full support for metadata).
Cleaning and smoothing functions are flexible and can be applied as required by the user.
We provide examples and demonstrate best practices as developed by the `SOCCO Group <http://www.socco.org.za/>`_.

For the original publication of this package see: https://doi.org/10.3389/fmars.2019.00738.

For recommendations or bug reports, please visit https://github.com/GliderToolsCommunity/GliderTools/issues/new

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   cheatsheet

.. toctree::
   :maxdepth: 2
   :caption: Users Guide

   loading
   quality_control
   physics
   optics
   calibration
   mapping
   saving
   other

.. toctree::
   :maxdepth: 2
   :caption: Help and Reference

   GitLab Repo <https://github.com/GliderToolsCommunity/GliderTools>
   api
   package_structure
   citing
   contributing
   history
   wishlist
