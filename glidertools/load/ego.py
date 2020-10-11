# base module to load ego files

from ..utils import calc_dive_phase, dive_phase_to_number


def load_mission_nc(filename):
    """
    Loads an EGO formatted glider mission file.

    Parameters
    ----------
    filename : str
        path and filename of the EGO netCDF file.

    Returns
    -------
        an xarray.Dataset object with all netCDF info attached
    ego_data : xr.Dataset
    """

    import xarray as xr

    xds = xr.open_dataset(filename)

    if "PHASE" in xds:
        phase = xds.PHASE.load()
        null_frac = phase.isnull().sum() / phase.size

    if (null_frac > 0.2) | ("PHASE" not in xds):
        time = xds.TIME.load()
        press = xds.PRES.load()
        phase = calc_dive_phase(time, press)

    xds["DIVES"] = dive_phase_to_number(phase)

    return xds
