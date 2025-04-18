#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="cmip6_tpx",
    version="0.1.0",
    description="Analysis and visualization toolkit for CMIP6 multi-scale temperature and precipitation extremes",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Roberto Suarez",
    author_email="roberto.suarez.science@gmail.com",
    url="https://github.com/rob-ds/cmip6_tpx",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "xarray",
        "dask",
        "netcdf4",
        "cdsapi",
        "matplotlib",
        "cartopy",
        "seaborn",
        "pymannkendall",
        "statsmodels",
    ],
    entry_points={
        "console_scripts": [
            "cmip6_download=scripts.download_data:main",
            "cmip6_anomalies=scripts.compute_anomalies:main",
            "cmip6_extremes=scripts.compute_extremes:main",
            "cmip6_plots=scripts.generate_plots:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="climate CMIP6 extremes temperature precipitation analysis visualization",
)
