# CMIP6 Temperature & Precipitation Extremes Analysis (CMIP6_TPX)

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python toolkit for analyzing multi-scale climate variability and extremes from CMIP6 model data, with a focus on near-surface air temperature and precipitation.

## Overview

CMIP6_TPX provides an end-to-end workflow for climate scientists to investigate how temperature and precipitation patterns may change in future climate scenarios. The package enables users to:

- Download CMIP6 climate data from the Climate Data Store (CDS)
- Process multiple models, variables, and scenarios (SSP2-4.5, SSP5-8.5)
- Calculate and decompose climate anomalies across multiple time scales
- Quantify changes in climate extremes using standardized indices
- Generate publication-quality visualizations for analysis results

This toolkit implements adapted methodologies from the Expert Team on Climate Change Detection and Indices (ETCCDI) for consistent analysis of climate extremes.

## Key Features

- **Data Acquisition**: Streamlined interface to the CDS API for retrieving CMIP6 model data
- **Multi-Model Support**: Analysis across 25+ CMIP6 models with consistent processing
- **Multi-Scale Decomposition**: Butterworth filtering to separate interannual and decadal climate variability
- **Climate Extremes Analysis**:
  - Temperature indices: warm/cold spells, threshold exceedance frequencies
  - Precipitation indices: intensity, duration, extremes (R95p, Rx1day, etc.)
  - Compound extreme events: hot-dry, hot-wet, and cold-wet conditions
- **Robust Visualization**:
  - Time series analysis with trend statistics
  - Monthly heatmaps with significance testing
  - Multi-scale anomaly decomposition plots
  - Geospatial reference mapping

## Installation

### Using Conda (recommended)

```bash
# Clone repository
git clone https://github.com/example/cmip6_tpx.git
cd cmip6_tpx

# Create conda environment
conda env create -f cmip6_tpx.yml
conda activate cmip6_tpx

# Install package in development mode
pip install -e .
```

### Using pip

```bash
pip install cmip6_tpx
```

### CDS API Setup

This package requires access to the Climate Data Store API. Configure your credentials:

1. Register at https://cds.climate.copernicus.eu/
2. Generate an API key from your user profile
3. Create a `.cdsapirc` file in your home directory with:

```
url: https://cds.climate.copernicus.eu/api/v2
key: <your-api-key>
```

## Case Studies

The repository includes data and analysis results for three regional case studies, demonstrating the package's capabilities across different climate zones:

1. **Alicante, Spain** (Mediterranean climate): `38.27°N, 0.71°W`
2. **Tambacounda, Senegal** (Tropical savanna climate): `13.77°N, 13.67°W`
3. **Bergen, Norway** (Oceanic climate): `60.39°N, 5.32°E`

Each case study includes:
- Raw NetCDF data in `data/raw/historical` and `data/raw/projections`
- Processed anomaly data in `data/processed/anomalies`
- Computed extreme indices in `data/processed/extremes`
- Visualization outputs in `figures/`

### Example Workflows for Case Studies

#### 1. Data Download

Download climate data for each study location:

```bash
# Download data for Alicante, Spain
python scripts/download_data.py --lat 38.27 --lon -0.71 --model ec_earth3_cc --variables temperature precipitation

# Download data for Tambacounda, Senegal
python scripts/download_data.py --lat 13.77 --lon -13.67 --model hadgem3_gc31_ll --variables temperature precipitation

# Download data for Bergen, Norway
python scripts/download_data.py --lat 60.39 --lon 5.32 --model noresm2_mm --variables temperature precipitation
```

#### 2. Computing Anomalies

For each case study location, multi-scale climate anomalies have been computed for all 12 months of the year, using both SSP2-4.5 and SSP5-8.5 scenarios. This comprehensive approach allows for complete seasonal analysis of climate variability.

The general command format used is:

```bash
python scripts/compute_anomalies.py --variable [temperature|precipitation] --experiment [ssp245|ssp585] --month [1-12] --model [model_name]
```

For example:
```bash
# Example for Alicante (EC-Earth3-CC model)
python scripts/compute_anomalies.py --variable temperature --experiment ssp245 --month 10 --model ec_earth3_cc

# Example for Tambacounda (HadGEM3-GC31-LL model)
python scripts/compute_anomalies.py --variable precipitation --experiment ssp585 --month 8 --model hadgem3_gc31_ll
```

This process was repeated for all combinations of variables, experiments, and months for each location.

#### 3. Computing Extreme Indices

Similarly, standardized climate extreme indices have been calculated for all 12 months for each location. This allows for comprehensive analysis of extreme events across seasons.

The general command format used is:

```bash
python scripts/compute_extremes.py --experiment [ssp245|ssp585] --month [1-12] --model [model_name] --temperature-threshold 90 --precipitation-threshold 95
```

For example:
```bash
# Example for Alicante with default parameters
python scripts/compute_extremes.py --experiment ssp245 --month 10 --model ec_earth3_cc --temperature-threshold 90 --precipitation-threshold 95

# Example for Tambacounda with modified heat wave duration parameter
python scripts/compute_extremes.py --experiment ssp585 --month 8 --model hadgem3_gc31_ll --temperature-threshold 90 --precipitation-threshold 95 --heat-wave-min-duration 3
```

This process was executed for all months and both climate scenarios for each case study location.

#### 4. Visualization

The repository includes a comprehensive set of visualization examples in the `visualization_commands.md` file. Examples include:

```bash
# Alicante - Maximum 1-day precipitation (flash floods)
python scripts/generate_plots.py --viz-type timeseries --variable precipitation --month 10 --metric rx1day --consistent-range --model ec_earth3_cc

# Tambacounda - Hot-Dry Frequency (drought patterns)
python scripts/generate_plots.py --viz-type heatmap --variable temperature --experiment ssp585 --metric hot_dry_frequency --months all --model hadgem3_gc31_ll

# Bergen - Cold days percentage (winter extremes)
python scripts/generate_plots.py --viz-type timeseries --variable temperature --month 1 --metric tm10p --consistent-range --model noresm2_mm
```

For a complete set of visualization commands, refer to the `visualization_commands.md` file included in the repository.

## Usage

CMIP6_TPX provides both a command-line interface and Python API for analysis workflows.

### Command-Line Interface

```bash
# Download data for EC-Earth3-CC model
cmip6_download --model ec_earth3_cc --lat 43.5 --lon 10.2 --variables temperature precipitation

# Calculate temperature anomalies for July using SSP2-4.5 scenario
cmip6_anomalies --variable temperature --experiment ssp245 --month 7 --model ec_earth3_cc

# Compute climate extremes indices for January
cmip6_extremes --experiment ssp585 --month 1 --model hadgem3_gc31_ll --temperature-threshold 90 --precipitation-threshold 95

# Generate visualization of warm spell days trend
cmip6_plots --viz-type timeseries --variable temperature --experiment ssp585 --month 7 --metric warm_spell_days --model noresm2_mm
```

### Python API

```python
from cmip6_tpx.analysis.variability import VariabilityAnalyzer
from cmip6_tpx.visualization.timeseries import TimeSeriesPlotter

# Calculate multi-scale anomalies
analyzer = VariabilityAnalyzer(
    variable="temperature",
    experiment="ssp245",
    month=7,
    model="ec_earth3_cc"
)
results = analyzer.compute()

# Create visualization
plotter = TimeSeriesPlotter(
    variable="temperature",
    experiment="ssp245",
    month=7,
    model="ec_earth3_cc"
)
fig = plotter.plot_anomaly_decomposition()
fig.savefig("temperature_anomaly_decomposition.png")
```

## Project Structure

```
cmip6_tpx/
│
├── .gitignore                  # Ignore data files, temp files, etc.
├── README.md                   # Project documentation
├── cmip6_tpx.yml               # Conda environment specifications
├── LICENSE.txt                 # License
├── setup.py                    # Package installation config
├── visualization_commands.md   # Example visualization commands
│
├── data/                       # Data storage
│   ├── raw/                    # Downloaded CMIP6 data
│   │   ├── historical/         # 1995-2014 baseline data
│   │   └── projections/        # Future 2015-2100 climate projections
│   └── processed/              # Processed datasets
│       ├── anomalies/          # Multi-scale anomaly time series
│       └── extremes/           # ETCCDI-derived indices
│
├── figures/                    # Figure storage
│
├── scripts/                    # Execution scripts
│   ├── download_data.py        # Script for CDS data retrieval
│   ├── compute_anomalies.py    # Calculate multi-scale anomalies from raw data
│   ├── compute_extremes.py     # Calculate metrics of extreme indices
│   └── generate_plots.py       # Create visualizations
│
└── src/                        # Core package
    ├── data/                   # Data handling
    │   └── retrieval.py        # CDS API interface
    │
    ├── analysis/               # Data analysis
    │   ├── base_analyzer.py    # Abstract base class
    │   ├── anomalies.py        # Anomaly calculation
    │   ├── filters.py          # Butterworth filters
    │   ├── variability.py      # Multi-scale variability
    │   └── extremes.py         # ETCCDI metrics
    │
    ├── utils/                  # Utility functions
    │   ├── geo_utils.py        # Geographic utility functions
    │   └── netcdf_utils.py     # Netcdf utility functions
    │
    └── visualization/          # Visualization tools
        ├── plotter.py          # Base plotting class
        ├── timeseries.py       # Time series plots
        ├── extreme_statistics.py # Extreme indices plots
        ├── location_map.py     # Location mapping
        └── export.py           # Figure export utilities
```

## Available CMIP6 Models

CMIP6_TPX supports a wide range of CMIP6 models that provide both historical and future climate projections:

| Model Key | Full Name |
|-----------|-----------|
| access_cm2 | ACCESS-CM2 (Australia) |
| awi_cm_1_1_mr | AWI-CM-1-1-MR (Germany) |
| cesm2 | CESM2 (USA) |
| ec_earth3_cc | EC-Earth3-CC (Europe) |
| hadgem3_gc31_ll | HadGEM3-GC31-LL (UK) |
| ipsl_cm6a_lr | IPSL-CM6A-LR (France) |
| miroc6 | MIROC6 (Japan) |
| mri_esm2_0 | MRI-ESM2-0 (Japan) |
| noresm2_mm | NorESM2-MM (Norway) |

*See the full list in `src/data/retrieval.py`.*

## Climate Indices

The package calculates a comprehensive set of climate indices:

### Temperature Indices
- TM_max/min: Maximum/minimum daily mean temperature
- TM90p/TM10p: Percentage of days with temperature above 90th/below 10th percentile
- Warm/Cold spell days: Count of days in warm/cold spells

### Precipitation Indices
- R95p: Precipitation on very wet days (>95th percentile)
- RX1day/RX5day: Maximum 1-day/5-day precipitation
- SDII: Simple daily intensity index
- CDD/CWD: Consecutive dry/wet days

### Compound Indices
- Hot-dry/hot-wet/cold-wet frequency: Count of days meeting compound criteria
- CWHD: Consecutive wet-hot days
- WSPI: Warm-period precipitation intensity

## Citation

If you use CMIP6_TPX in your research, please cite:

```
@software{cmip6_tpx,
  author = {Roberto Suarez},
  title = {CMIP6_TPX: A toolkit for multi-scale analysis of climate extremes},
  year = {2025},
  url = {https://github.com/rob-ds/cmip6_tpx}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- This package utilizes data from the Copernicus Climate Data Store
- Climate indices methodology adapted from ETCCDI recommendations
- Thanks to the CMIP6 modeling community for their contributions