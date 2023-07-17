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


v2023.xx (unreleased)
------------------------

.. _whats-new.2023.xx:

New Features
~~~~~~~~~~~~
- added import for VOTO seaexplorer data (:pull:`170`) By `Martin Mohrmann <https://github.com/MartinMohrmann>`_.
- added versatile, depth dependent masking (:pull:`172`) By `Martin Mohrmann <https://github.com/MartinMohrmann>`_.

Breaking changes
~~~~~~~~~~~~~~~~
- GliderTools defaults for Figure creation were changed. Automatic application of plt.tight_layout was dropped in favour of more flexible embedding of GliderTools plots into existing layouts/subplots.
- The mixed layer depth algorithm was corrected. (:pull:`169`, :issue:`168`). By `Martin Mohrmann <https://github.com/MartinMohrmann>`_. API change! Existing mixed layer computation code must be adapted.
- Changed the behavior of `find_dive_phase` and `calc_dive_number` to use a smaller depth threshold when determining a valid dive (15 dbar down from 200 dbar).  this is also now adjusteable. (:pull:`134`) By `Tom Hull <https://github.com/tomhull>`_.

Internal changes
~~~~~~~~~~~~~~~~
- Some cleanup of old python2 dependencies (:pull:`166`). By `Martin Mohrmann <https://github.com/MartinMohrmann>`_.


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
