Loading data
============

To start using Glider Tools you first need to import the package to the
interactive workspace.


## Import `GliderTools`


```python
# pylab for more MATLAB like environment and inline displays plots below cells
%pylab inline

# if gsw Warning shows, manually install gsw if possible - will still work without
import glidertools as gt
from cmocean import cm as cmo  # we use this for colormaps
```

    Populating the interactive namespace from numpy and matplotlib


## Working with Seaglider base station files

GliderTools supports loading Seaglider files, including `scicon` data (different sampling frequencies).
There is a function that makes it easier to find variable names that you'd like to load: `gt.load.seaglider_show_variables`

This function is demonstrated in the cell below.
The function accepts a **list of file names** and can also receive a string with a wildcard placeholder (`*`) and basic regular expressions are also supported. In the example below we use a simple asterisk placeholder for all the files.

Note that the function chooses only one file from the passed list or glob string - this file name will be shown. The returned table shows the variable name, dimensions, units and brief comment if it is available.


```python
filenames = '/Users/luke/Work/Data/sg542/p5420*.nc'

gt.load.seaglider_show_variables(filenames)
```

    information is based on file: /Users/luke/Work/Data/sg542/p5420177.nc

    <table will be displayed here>



## Load variables

From the variable listing, one can choose multiple variables to load. Note that one only needs the variable name to load the data. Below, we've created a list of variables that we'll be using for this demo.

The `gt.load.seaglider_basestation_netCDFs` function is used to load a list of variables. It requires the filename string or list (as described above) and keys. It may be that these variables are not sampled at the same frequency. In this case, the loading function will load the sampling frequency dimensions separately. The function will try to find a time variable for each sampling frequency/dimension.

### Coordinates and automatic *time* fetching
All associated coordinate variables will also be loaded with the data if coordinates are documented. These may included *latitude, longitude, depth* and *time* (naming may vary). If time cannot be found for a dimension, a *time* variable from a different dimension with the same number of observations is used instead. This insures that data can be merged based on the time of sampling.

### Merging data based on time
If the `return_merged` is set to *True*, the function will merge the dimensions if the dimension has an associated *time* variable.

The function returns a dictionary of `xarray.Datasets` - a Python package that deals with coordinate indexed multi-dimensional arrays. We recommend that you read the documentation (http://xarray.pydata.org/en/stable/) as this package is used throughout *GliderTools*. This allows the original metadata to be copied with the data. The dictionary keys are the names of the dimensions. If `return_merged` is set to *True* an additional entry under the key `merged` will be included.

The structure of a dimension output is shown below. Note that the merged data will use the largest dimension as the primary dataset and the other data will be merged onto that time index. Data is linearly interpolated to the nearest time measurement of the primary index, but only by one measurement to ensure transparancy.


```python
names = [
    'ctd_depth',
    'ctd_time',
    'ctd_pressure',
    'salinity',
    'temperature',
    'eng_wlbb2flvmt_Chlsig',
    'eng_wlbb2flvmt_wl470sig',
    'eng_wlbb2flvmt_wl700sig',
    'aanderaa4330_dissolved_oxygen',
    'eng_qsp_PARuV',
]

ds_dict = gt.load.seaglider_basestation_netCDFs(
    filenames, names,
    return_merged=True,
    keep_global_attrs=False
)
```

    DIMENSION: sg_data_point
    {
        ctd_pressure, eng_wlbb2flvmt_wl470sig, eng_wlbb2flvmt_wl700sig, temperature,
        ctd_time, ctd_depth, latitude, aanderaa4330_dissolved_oxygen, salinity,
        eng_wlbb2flvmt_Chlsig, longitude
    }


    100%|██████████| 336/336 [00:04<00:00, 73.66it/s]



    DIMENSION: qsp2150_data_point
    {eng_qsp_PARuV, time}


    100%|██████████| 336/336 [00:01<00:00, 181.67it/s]



    Merging dimensions on time indicies: sg_data_point, qsp2150_data_point,


The returned data contains the dimensions of the requested variables a `merged` object is also returned if return_merged=True
```python

print(ds_dict.keys())
```

    dict_keys(['sg_data_point', 'qsp2150_data_point', 'merged'])


### Metadata handling
If the keyword arguement `keep_global_attrs=True`, the attributes from the original files (for all that are the same) are passed on to the output *Datasets* from the original netCDF attributes. The variable attributes (units, comments, axis...) are passed on by default, but can also be set to False if not wanted. GliderTools functions will automatically pass on these attributes to function outputs if a `xarray.DataArray` with attributes is given.
All functions applied to data will also be recorded under the variable attribute `processing`.


The merged dataset contains all the data interpolated to the nearest observation of the longest dimension the metadata is also shown for the example below
```python
ds_dict['merged']
```




    xarray.Dataset>
    Dimensions:                        (merged: 382151)
    Coordinates:
        ctd_depth                      (merged) float64 -0.08821 0.018 ... -0.1422
        latitude                       (merged) float64 -42.7 -42.7 ... -43.0 -43.0
        longitude                      (merged) float64 8.744 8.744 ... 8.5 8.5
        ctd_time_dt64                  (merged) datetime64[ns] 2015-12-08T07:36:16 ...

    Dimensions without coordinates: merged
    Data variables:
        ctd_pressure                   (merged) float64 -0.08815 0.01889 ... -0.1432
        eng_wlbb2flvmt_wl470sig        (merged) float64 375.0 367.0 ... 98.0 91.0
        eng_wlbb2flvmt_wl700sig        (merged) float64 2.647e+03 ... 137.0
        temperature                    (merged) float64 11.55 11.54 ... 11.06 10.97
        ctd_time                       (merged) float64 1.45e+09 ... 1.455e+09
        aanderaa4330_dissolved_oxygen  (merged) float64 nan nan nan ... 269.1 269.1
        salinity                       (merged) float64 nan nan nan ... 34.11 34.11
        eng_wlbb2flvmt_Chlsig          (merged) float64 145.0 126.0 ... 215.0 215.0
        dives                          (merged) float64 1.0 1.0 1.0 ... 344.5 344.5
        eng_qsp_PARuV                  (merged) float64 0.551 0.203 ... 0.021 0.023
        time                           (merged) float64 1.45e+09 ... 1.455e+09
        time_dt64                      (merged) datetime64[ns] 2015-12-08T07:36:16 ...

    Attributes:
        date_created:             2019-07-11 14:08:40
        number_of_dives:          344.0
        files:                    ['p5420001.nc', 'p5420002.nc', 'p5420004.nc', '...
        time_coverage_start:      2015-12-08 07:36:16
        time_coverage_end:        2016-02-08 04:39:04
        geospatial_vertical_min:  -0.6323553853732649
        geospatial_vertical_max:  1011.1000623417478
        geospatial_lat_min:       -43.085757609206
        geospatial_lat_max:       -42.70088638031523
        geospatial_lon_min:       8.29983279020758
        geospatial_lon_max:       8.7753734452125
        processing:               [2019-07-11 14:08:40] imported data with Glider...



### Renaming for ease of access
When renaming, just be sure that there are no variables with names that you are trying to replace. In the example below we remove `time` in case it exists in the files.
```python
# Here we drop the time variables imported for the PAR variable
# we don't need these anymore. You might have to change this
# depening on the dataset
merged = ds_dict['merged']
if 'time' in merged:
    merged = merged.drop(["time", "time_dt64"])


# To make it easier and clearer to work with, we rename the
# original variables to something that makes more sense. This
# is done with the xarray.Dataset.rename({}) function.
# We only use the merged dataset as this contains all the
# imported dimensions.
# NOTE: The renaming has to be specific to the dataset otherwise an error will occur
dat = merged.rename({
    'salinity': 'salt_raw',
    'temperature': 'temp_raw',
    'ctd_pressure': 'pressure',
    'ctd_depth': 'depth',
    'ctd_time_dt64': 'time',
    'ctd_time': 'time_raw',
    'eng_wlbb2flvmt_wl700sig': 'bb700_raw',
    'eng_wlbb2flvmt_wl470sig': 'bb470_raw',
    'eng_wlbb2flvmt_Chlsig': 'flr_raw',
    'eng_qsp_PARuV': 'par_raw',
    'aanderaa4330_dissolved_oxygen': 'oxy_raw',
})

print(dat)

# variable assignment for conveniant access
depth = dat.depth
dives = dat.dives
lats = dat.latitude
lons = dat.longitude
time = dat.time
pres = dat.pressure
temp = dat.temp_raw
salt = dat.salt_raw
par = dat.par_raw
bb700 = dat.bb700_raw
bb470 = dat.bb470_raw
fluor = dat.flr_raw

# name coordinates for quicker plotting
x = dat.dives
y = dat.depth
```
