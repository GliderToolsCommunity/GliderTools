.. currentmodule:: glidertools

What's New
===========

.. Template (do not remove)
    ------------------------

    Breaking changes
    ~~~~~~~~~~~~~~~~
    Description. (:pull:`ii`, :issue:`ii`). By `Name <https://github.com/github_username>`_.

    New Features
    ~~~~~~~~~~~~

    Documentation
    ~~~~~~~~~~~~~

    Internal Changes
    ~~~~~~~~~~~~~~~~

    Bug fixes
    ~~~~~~~~~
    - Dark count corrections for optical sensors(:pull:'110'). By 'Isabelle Giddy <https://github.com/isgiddy>'_.

v2024.xx (unreleased)
---------------------

.. _whats-new.2024.xx:

Bug fixes
~~~~~~~~~
- fixed quenching correction for eastern longitudes larger than 70E (:issue:`202`). (:pull:`204`) by 'Martin Mohrmann <https://github.com/MartinMohrmann'>`_.


v2023.07.25 (2023/07/25)
------------------------

.. _whats-new.2023.07.25:

New Features
~~~~~~~~~~~~
- added import for VOTO seaexplorer data (:pull:`170`) By `Martin Mohrmann <https://github.com/MartinMohrmann>`_.
- added versatile, depth dependent masking (:pull:`172`) and per profile grouping (:pull:`175`). By `Martin Mohrmann <https://github.com/MartinMohrmann>`_.
- add concatenation of two or more datasets (:pull:`173`), even with different set of variables (:pull:`183`). By `Martin Mohrmann <https://github.com/MartinMohrmann>`_.

Breaking changes
~~~~~~~~~~~~~~~~
- Changed the behavior of `find_dive_phase` and `calc_dive_number` to use a smaller depth threshold when determining a valid dive (15 dbar down from 200 dbar).  this is also now adjusteable. (:pull:`134`) By `Tom Hull <https://github.com/tomhull>`_.
- GliderTools defaults for Figure creation were changed. Automatic application of plt.tight_layout was dropped in favour of more flexible embedding of GliderTools plots into existing layouts/subplots. (:pull:`185`). By `Martin Mohrmann <https://github.com/MartinMohrmann>`_.
- The mixed layer depth algorithm was corrected. (:pull:`169`, :issue:`168`). By `Martin Mohrmann <https://github.com/MartinMohrmann>`_. API change! Existing mixed layer computation code must be adapted.

Internal changes
~~~~~~~~~~~~~~~~
- Removed outdated python-seawater dependency (:pull:`186`). By `Callum Rollo <https://github.com/callumrollo>`_.
- Update documentation of required dependencies (:pull:`174`). By `Sören Thomsen <https://github.com/soerenthomsen>`_.
- Some cleanup of old python2 dependencies (:pull:`166`). By `Martin Mohrmann <https://github.com/MartinMohrmann>`_.
- Replace deprecated pkg_resources with importlib.metadata (:pull:`187`). By `Martin Mohrmann <https://github.com/MartinMohrmann>`_.
- Add release guide to documentation (:pull:`186`). By `Martin Mohrmann <https://github.com/MartinMohrmann>`_.
- Cleanup of unused imports (:pull:`174`). By `Martin Mohrmann <https://github.com/MartinMohrmann>`_.

Bug fixes
~~~~~~~~~
- Adapt demo notebook to updated Glider Tools (:pull:`179`). By `Callum Rollo <https://github.com/callumrollo>`_.
- Fix netCDF attribute handling for non-string attributes (:pull:`194`). By `Martin Mohrmann <https://github.com/MartinMohrmann>`_.
- Adapt quenching_report to modern numpy versions (:pull:`191`) By `Martin Mohrmann <https://github.com/MartinMohrmann>`_.
- Improve error handling for MLD computation (:pull:`190`). By `Martin Mohrmann <https://github.com/MartinMohrmann>`_.

Thanks also to `Julius Busecke <https://github.com/jbusecke>`_ for help with the github CI, `Sam Woodman <https://github.com/smwoodman>`_ for detailed bug reports and everyone else who has contributed.


v2022.12.13 (2022/12/13)
------------------------

.. _whats-new.2022.12.13:

Internal changes
~~~~~~~~~~~~~~~~
- Refactoring and update of testing and development framework, update of flake, black and almost all python dependencies


Breaking changes
~~~~~~~~~~~~~~~~
- Fixed processing/calc_oxygen (:pull: `116`, :issue: `112`) By `Callum Rollo <https://github.com/callumrollo>`_.


Internal Changes
~~~~~~~~~~~~~~~~
- Implemented code linting as part of the CI (:pull:`100`) By `Julius Busecke <https://github.com/jbusecke>`_.

Documentation
~~~~~~~~~~~~~
- Added conda installation instructions + badge. (:pull:`94`) By `Julius Busecke <https://github.com/jbusecke>`_.

Bug fixes
~~~~~~~~~
- Replaced `skyfield` dependency with `astral`, fixing sunrise/sunset problems at high latitudes.  By `Isabelle Sindiswa Giddy <https://github.com/isgiddy>`_.

v2021.03 (2021/3/30)
-------------------------

.. _whats-new.2021.03:

Documentation
~~~~~~~~~~~~~
- Updated contributor guide for conda based workflow. (:pull:`81`) By `Julius Busecke <https://github.com/jbusecke>`_.

Internal Changes
~~~~~~~~~~~~~~~~
- Migration of CI to conda based workflow with multiple python versions. (:pull:`54`)  By `Julius Busecke <https://github.com/jbusecke>`_.
- Revamp distribution actions. (:pull:`82`) By `Julius Busecke <https://github.com/jbusecke>`_.
- Migrate from astral to skyfield (:pull:'121') By 'Isabelle Giddy <https://github.com/isgiddy>'_.
