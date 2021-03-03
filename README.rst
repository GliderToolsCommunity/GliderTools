===============================
glidertools
===============================


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
.. image:: https://zenodo.org/badge/256331120.svg
        :target: https://zenodo.org/badge/latestdoi/256331120

Glider tools is a Python 3.7 package designed to process data from the first level of processing to a science ready dataset. The package is designed to easily import data to a standard column format (numpy.ndarray or pandas.DataFrame). Cleaning and smoothing functions are flexible and can be applied as required by the user. We provide examples and demonstrate best practices as developed by the SOCCO Group (http://socco.org.za/).

Please cite the original publication of this package (https://doi.org/10.3389/fmars.2019.00738) and the software package (https://doi.org/10.5281/zenodo.4075238).

Installation
------------

PyPI
....
To install the core package run: ``pip install glidertools``.

GitHub
......
1. Clone glidertools to your local machine: `git clone https://github.com/GliderToolsCommunity/GliderTools`
2. Change to the parent directory of GliderTools
3. Install glidertools with `pip install -e ./GliderTools`. This will allow
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
- Error reporting with using GitLab (https://github.com/GliderToolsCommunity/GliderTools/issues/new). Please copy the entire error message (even if it's long).
- Join our slack group:  https://join.slack.com/t/glidertools/shared_invite/zt-mz1akvfr-5k5mzuL~Z44JGPo8v~L0qg
- Detailed error reporting so users know where the fault lies.
- Oxygen processing is rudimentary as we do not have the expertise in our group to address this

For contributing follow the instructions here: https://glidertools.readthedocs.io/en/latest/contributing.html

Acknowledgements
----------------
- We rely heavily on ``ion_functions.data.flo_functions`` which was
  written by Christopher Wingard, Craig Risien, Russell Desiderio
- This work was initially funded by Pedro M Scheel Monteiro at the
  Council for Scientific and Industrial Research (where Luke was working
  at the time of writing the code).
- Testers for their feedback: SOCCO team at the CSIR and ...
