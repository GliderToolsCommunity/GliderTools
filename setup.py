from os.path import exists

from setuptools import find_packages, setup

if exists("README.rst"):
    with open("README.rst") as f:
        long_description = f.read()
else:
    long_description = ""

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.7",
]

setup(
    name="glidertools",
    version="v2020.5",
    author="Luke Gregor",
    author_email="lukegre@gmail.com",
    description=(
        "A toolkit for processing Seaglider base station NetCDF files: "
        "despiking, smoothing, outlier detection, backscatter, fluorescence "
        "quenching, calibration, gridding, interpolation. Documentation "
        "at https://glidertools.readthedocs.io"
    ),
    keywords="GliderTools",
    license="GNUv3",
    classifiers=CLASSIFIERS,
    url="https://github.com/GliderToolsCommunity/GliderTools",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[
        "astral>=2.2",
        "matplotlib>=3",
        "numexpr",
        "netcdf4==1.5.4",
        "scikit-learn>=0.22",
        "seawater>=3.3",
        "tqdm>=4",
        "xarray>=0.16.0",
    ],
    test_suite="glidertools/tests",
    tests_require=["pytest-cov"],
    setup_requires=[
        "setuptools_scm",
        "setuptools>=30.3.0",
        "setuptools_scm_git_archive",
    ],
)
