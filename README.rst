===============================
glidertools
===============================

.. image:: https://github.com/GliderToolsCommunity/GliderTools/actions/workflows/ci.yml/badge.svg
        :target: https://github.com/GliderToolsCommunity/GliderTools/actions/workflows/ci.yml
.. image:: https://img.shields.io/conda/vn/conda-forge/glidertools.svg
        :target: https://anaconda.org/conda-forge/glidertools
.. image:: https://badgen.net/pypi/v/glidertools
        :target: https://pypi.org/project/glidertools
.. image:: https://pepy.tech/badge/glidertools
        :target: https://pepy.tech/project/glidertools
.. image:: https://readthedocs.org/projects/glidertools/badge/?version=latest
        :target: https://glidertools.readthedocs.io
.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
        :target: https://www.gnu.org/licenses/gpl-3.0
.. image:: https://img.shields.io/badge/Journal-10.3389%2Ffmars.2019.00738-blue
        :target: https://doi.org/10.3389/fmars.2019.00738
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3360280.svg
        :target: https://doi.org/10.5281/zenodo.4075238
.. image:: https://codecov.io/gh/GliderToolsCommunity/GliderTools/branch/master/graph/badge.svg?token=FPUJ29TMSH
        :target: https://codecov.io/gh/GliderToolsCommunity/GliderTools

Glider tools is a Python 3.8+ package designed to process data from the first level of processing to a science ready dataset (delayed mode quality control). The package is designed to easily import data to a standard column format (numpy.ndarray or pandas.DataFrame). Cleaning and smoothing functions are flexible and can be applied as required by the user. We provide examples and demonstrate best practices as developed by the `SOCCO Group <http://socco.org.za/>`_, with early contributions from `Polar Gliders <https://sebswart.com/>`_ at the University of Gothenburg. GliderTools includes contributions from `VOTO <https://voiceoftheocean.org//>`_. We aim to implement Best Practices developed by `OceanGliders <https://www.oceangliders.org/>`_ in the ongoing `discussions <https://github.com/OceanGlidersCommunity>`_.

Please cite the `original publication <https://doi.org/10.3389/fmars.2019.00738>`_ of this package and `the package itself <https://doi.org/10.5281/zenodo.4075238>`_.

Installation
------------
Conda
.....
To install the core package from conda-forge run: ``conda install -c conda-forge glidertools``

PyPI
....
To install the core package run: ``pip install glidertools``.

GitHub
......
1. Clone glidertools to your local machine: ``git clone https://github.com/GliderToolsCommunity/GliderTools``
2. Change to the parent directory of GliderTools
3. Install glidertools with ``pip install -e ./GliderTools``. This will allow
   changes you make locally, to be reflected when you import the package in Python

Recommended, but optional packages
..................................
There are some packages that are not installed by default, as these are large packages or can
result in installation errors, resulting in failure to install GliderTools.
These should install automatically with ``pip install package_name``:

* ``gsw``: accurate density calculation (may fail in some cases)
* ``pykrige``: variogram plotting (installation generally works, except when bundled)
* ``plotly``: interactive 3D plots (large package)


How you can contribute
----------------------
- Join the community `by introducing yourself <https://github.com/GliderToolsCommunity/GliderTools/discussions/47>`_ (no need to be a Python or Git guru! Just say what you are working with and join the discussion)
- If you find an error, please report it on `as a Github issue <https://github.com/GliderToolsCommunity/GliderTools/issues/new>`_. Please copy the entire error message (even if it's long).
- Oxygen processing is rudimentary so far but we are on it and happy to get your support `in this discussion <https://github.com/GliderToolsCommunity/GliderTools/discussions/74>`_

For contributing follow the `instructions <https://glidertools.readthedocs.io/en/latest/contributing.html>`_

Acknowledgements
----------------
- We rely heavily on ``ion_functions.data.flo_functions`` which was
  written by Christopher Wingard, Craig Risien, Russell Desiderio
- This work was initially funded by Pedro M Scheel Monteiro at the
  Council for Scientific and Industrial Research (where Luke was working
  at the time of writing the code).
- Testers for their feedback: SOCCO team at the CSIR and ...
