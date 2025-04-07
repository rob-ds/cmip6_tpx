# CMIP6 Temperature & Precipitation Extremes Analysis (cmip6_tpx)

A Python package for analyzing EC-Earth3-CC climate model data from CMIP6, focusing on near-surface air temperature and precipitation variables.

## Overview

This package provides tools to:
- Download CMIP6 climate model data from CDS (Climate Data Store)
- Calculate anomalies relative to historical baseline (1995-2014)
- Apply Butterworth filters to separate interannual and decadal variability
- Analyze variability changes using sliding window approaches
- Calculate climate extremes (R95p) from daily data
- Generate visualizations of climate projections

## Installation

```bash
# Clone repository
git clone https://github.com/rob-ds/cmip6_tpx.git
cd cmip6_tpx

# Create conda environment
conda env create -f cmip6_tpx.yml
conda activate cmip6_tpx
```

## Usage

```bash
# Download EC-Earth3-CC data
python scripts/download_data.py

# Calculate anomalies
python scripts/compute_anomalies.py

# Additional analysis scripts
python scripts/compute_variability.py
python scripts/compute_extremes.py
python scripts/generate_plots.py
```

## Project Structure

- `data/`: Storage for raw and processed climate data
- `src/`: Core package functionality
  - `data/`: Data retrieval and processing
  - `analysis/`: Statistical analysis methods
  - `visualization/`: Plotting and visualization tools
- `scripts/`: Execution scripts for analysis workflow
- `tests/`: Unit and integration tests

## Dependencies

Core dependencies include:
- xarray, netCDF4 (data handling)
- numpy, scipy (numerical operations)
- cdsapi (data retrieval)
- matplotlib (visualization)

See `cmip6_tpx.yml` for complete environment specification.