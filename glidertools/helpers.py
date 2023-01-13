import inspect

from pkg_resources import DistributionNotFound, get_distribution


try:
    version = get_distribution("glidertools").version
except DistributionNotFound:
    version = "version_undefined"


class GliderToolsWarning(UserWarning):
    pass


class GliderToolsError(UserWarning):
    pass


def time_now():
    from pandas import Timestamp

    return str(Timestamp("today"))[:19]


def rebuild_func_call(frame):

    arginf = inspect.getargvalues(frame)
    name = frame.f_code.co_name
    args = arginf.args
    locl = arginf.locals

    module = inspect.getmodule(frame).__name__
    func = "{}.{}(".format(module, name)
    n_args = len(args)
    for c, arg_name in enumerate(args):
        arg_valu = str(locl[arg_name])
        if len(arg_valu) < 25:
            try:
                float(arg_valu)
            except ValueError:
                if (arg_valu == "True") | (arg_valu == "False"):
                    pass
                else:
                    arg_valu = "'{}'".format(arg_valu)
        else:
            arg_valu = "<{}>".format(arg_name)
        func += "{}={}".format(arg_name, arg_valu)

        if c < (n_args - 1):
            func += ", "
        else:
            func += ")"

    return func


def transfer_nc_attrs(frame, input_xds, output_arr, output_name, **attrs):
    import warnings

    import xarray as xr

    not_dataarray = not isinstance(input_xds, xr.DataArray)
    no_parent_frame = inspect.getmodule(frame.f_back) is None
    if not_dataarray:
        if no_parent_frame:
            msg = (
                "Primary input variable is not xr.DataArray data type - "
                "no metadata to pass on."
            )
            warnings.warn(msg, category=GliderToolsWarning)
        return output_arr
    else:
        if output_name is None:
            output_name = input_xds.name
        elif output_name.startswith("_"):
            output_name = input_xds.name + output_name

        attributes = input_xds.attrs.copy()
        history = "" if "history" not in attributes else attributes["history"]
        history += "[{}] (v{}) {};\n".format(
            time_now(), version, rebuild_func_call(frame)
        )
        attributes.update({"history": history})
        attributes.update(attrs)

        keys = list(attributes.keys())
        for key in keys:
            if attributes[key] == "":
                attributes.pop(key)

        xds = xr.DataArray(
            data=output_arr,
            coords=input_xds.coords,
            dims=input_xds.dims,
            name=output_name,
            attrs=attributes,
        )

        return xds


def printv(verbose, message):
    """
    A helper function that prints message if verbose=True (for cleaner code)

    Parameters
    ----------
    verbose : bool
    message : str
    """

    if verbose:
        print(message)
    else:
        pass
