from os.path import exists

from setuptools import find_packages, setup

if exists('README.rst'):
    with open('README.rst') as f:
        long_description = f.read()
else:
    long_description = ''

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
]

setup(
    name='glidertools',
    author='Luke Gregor',
    author_email='lukegre@gmail.com',
    description=(
        'A toolkit for processing Seaglider base station NetCDF files: '
        'despiking, smoothing, outlier detection, backscatter, fluorescence '
        'quenching, calibration, gridding, interpolation. Documentation '
        'at https://glidertools.readthedocs.io'
    ),
    keywords='GliderTools',
    license='GNUv3',
    classifiers=CLASSIFIERS,
    url='https://github.com/luke-gregor/GliderTools',
    use_scm_version={
        'version_scheme': 'post-release',
        'local_scheme': 'dirty-tag',
    },
    long_description=long_description,
    packages=find_packages(),
    install_requires=install_requires,
    test_suite='glidertools/tests',
    tests_require=['pytest-cov'],
    setup_requires=[
        'setuptools_scm',
        'setuptools>=30.3.0',
        'setuptools_scm_git_archive',
    ],
)
