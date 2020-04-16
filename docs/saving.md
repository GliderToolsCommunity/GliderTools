Saving data
===========

We have not created an explicit way to save data in GliderTools. This is primarily due to the fact that the package is built on top of two packages that already do this very well: [*pandas*](https://pandas.pydata.org/pandas-docs/stable/getting_started/overview.html) and [*xarray*](http://xarray.pydata.org/en/stable/).
*pandas* is widely used and deals with tabular formatted data (2D). *xarray* widely used in earth sciences as it supports multi-dimensional indexing (3D+). We highly recommend that you read through the documentation for these packages as they are incredibly powerful and you would benefit from knowing these tools regardless of using GliderTools or not!

We have written GliderTools primarily with *xarray* as the backend, due to the ability to store attributes (or metadata) alongside the data - something that *pandas* does not yet do. Moreover, we have also created the tool so that metadata is passed to the output of each function, while appending the function call to the *history* attribute. This ensures that the user of the data knows when and what functions (and arguements) were called and for which version of GliderTools this was done.

Examples
--------

First we give an example of how to save and read files to netCDF (which we recommend).

```python
import xarray as xr

# xds is an xarray.DataFrame with record of dimensions, coordinates and variables
xds.to_netcdf('data_with_meta.nc')

# this file can simply be loaded in the same way, without using GliderTools
# all the information that was attached to the data is still in the netCDF
xds = xr.open_dataset('data_with_meta.nc')
```

In this second example we show how to save the data to a CSV. While this is a common and widely used format, we do not recommend this as the go to format, as all metadata is lost when the file is saved.
```python
import pandas as pd

# If you prefer to save your data as a text file, you can easily do this with Pandas
# note that converting the file to a dataframe discards all the metadata
df = xds.to_dataframe()
df.to_csv('data_without_meta.csv')

# this file can simply be loaded in the same way, without using GliderTools
# there will be no more metadata attached to each variable
df = pd.read_csv('data_without_meta.csv')

# finally, you can also convert the file back to an xarray.Dataset
# however, the data will still be lost
xds = df.to_xarray()
```
