"""
Base Analyzer Module

This module defines the abstract base class for all analyzers in the CMIP6_TPX project.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Union, Any

import xarray as xr

# Configure logger
logger = logging.getLogger(__name__)


class BaseAnalyzer(ABC):
    """
    Abstract base class for climate data analysis in CMIP6_TPX project.

    This class defines the common interface for all analyzers, including
    methods for loading data, processing it, and saving results.
    """

    def __init__(
            self,
            variable: str,
            experiment: str,
            month: int,
            input_dir: Union[str, Path],
            output_dir: Union[str, Path],
    ):
        """
        Initialize the analyzer.

        Args:
            variable: Climate variable to analyze ('temperature' or 'precipitation')
            experiment: Experiment to analyze ('ssp245' or 'ssp585')
            month: Month to analyze (1-12)
            input_dir: Directory containing input data
            output_dir: Directory to store output data
        """
        self.variable = variable
        self.experiment = experiment
        self.month = month

        # Convert path strings to Path objects
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Variable name mapping
        self.var_name_mapping = {
            'temperature': 'tas',
            'precipitation': 'pr'
        }

        # Check if variable is valid
        if self.variable not in self.var_name_mapping:
            raise ValueError(f"Invalid variable: {variable}. Must be one of {list(self.var_name_mapping.keys())}")

        # Set netCDF variable name
        self.nc_var_name = self.var_name_mapping[self.variable]

        # Initialize data containers
        self.historical_data = None
        self.projection_data = None
        self.lat = None
        self.lon = None

        # Track processing state
        self.is_data_loaded = False

        logger.info(f"Initialized {self.__class__.__name__} for {variable}, {experiment}, month {month}")

    def load_data(self) -> None:
        """
        Load historical and projection data for the specified variable and experiment.

        This method loads the data from the input directory, extracts the center point
        from the 3x3 grid, and stores the results in the instance variables.
        """
        # Construct file paths
        historical_file = self._find_file(self.input_dir / "raw" / "historical", f"historical_{self.variable}")
        projection_file = self._find_file(self.input_dir / "raw" / "projections", f"{self.experiment}_{self.variable}")

        logger.info(f"Loading historical data from {historical_file}")
        logger.info(f"Loading projection data from {projection_file}")

        # Load historical data
        historical_ds = xr.open_dataset(historical_file)

        # Load projection data
        projection_ds = xr.open_dataset(projection_file)

        # Extract center point coordinates
        self.lat = float(historical_ds.lat[1].values)
        self.lon = float(historical_ds.lon[1].values)

        logger.info(f"Center point coordinates: lat={self.lat}, lon={self.lon}")

        # Extract center point data for the variable
        historical_var = historical_ds[self.nc_var_name][:, 1, 1].values
        projection_var = projection_ds[self.nc_var_name][:, 1, 1].values

        # Extract time variables
        historical_time = historical_ds.time.values
        projection_time = projection_ds.time.values

        # Create xarray DataArrays with time as dimension
        self.historical_data = xr.DataArray(
            data=historical_var,
            coords={'time': historical_time},
            dims=['time']
        )

        self.projection_data = xr.DataArray(
            data=projection_var,
            coords={'time': projection_time},
            dims=['time']
        )

        # Filter for the specified month
        self._filter_month()

        # Convert units if needed
        self._convert_units()

        # Mark as loaded
        self.is_data_loaded = True

        logger.info("Data loading complete")

    @staticmethod
    def _find_file(directory: Path, name_pattern: str) -> Path:
        """
        Find a file in the specified directory that matches the name pattern.

        Args:
            directory: Directory to search in
            name_pattern: Pattern to match in the file name

        Returns:
            Path to the found file

        Raises:
            FileNotFoundError: If no matching file is found
        """
        for file_path in directory.glob("*.nc"):
            if name_pattern in file_path.name:
                return file_path

        raise FileNotFoundError(f"No file matching '{name_pattern}' found in {directory}")

    def _filter_month(self) -> None:
        """
        Filter data for the specified month.

        This method converts the daily data to monthly averages for the specified month.
        """
        logger.info(f"Filtering data for month {self.month}")

        # Extract month from datetime
        historical_months = xr.DataArray(
            self.historical_data.time.dt.month,
            coords={'time': self.historical_data.time},
            dims=['time']
        )

        projection_months = xr.DataArray(
            self.projection_data.time.dt.month,
            coords={'time': self.projection_data.time},
            dims=['time']
        )

        # Filter for the specified month
        self.historical_data = self.historical_data.where(historical_months == self.month, drop=True)
        self.projection_data = self.projection_data.where(projection_months == self.month, drop=True)

        # Group by year and calculate monthly averages
        self.historical_data = self.historical_data.groupby('time.year').mean('time')
        self.projection_data = self.projection_data.groupby('time.year').mean('time')

        logger.info(f"Historical data shape after filtering: {self.historical_data.shape}")
        logger.info(f"Projection data shape after filtering: {self.projection_data.shape}")

    def _convert_units(self) -> None:
        """
        Convert units if needed.

        For precipitation, convert from kg.m-2.s-1 to mm.day-1.
        """
        if self.variable == 'precipitation':
            logger.info("Converting precipitation units from kg.m-2.s-1 to mm.day-1")

            # Multiply by seconds per day (86400)
            self.historical_data = self.historical_data * 86400
            self.projection_data = self.projection_data * 86400

    @abstractmethod
    def compute(self) -> Dict[str, Any]:
        """
        Compute analysis results.

        This method must be implemented by all derived classes.

        Returns:
            Dictionary containing analysis results
        """
        pass

    def save_results(self, results: Dict[str, Any], filename: str = None) -> Path:
        """
        Save analysis results to a NetCDF file.

        Args:
            results: Dictionary containing analysis results
            filename: Optional filename override

        Returns:
            Path to the saved file
        """
        if not filename:
            # Create default filename
            spatial_info = f"lat{self.lat:.2f}_lon{self.lon:.2f}".replace('.', 'p').replace('-', 'n')
            filename = f"{self.variable}_{self.experiment}_month{self.month:02d}_{spatial_info}.nc"

        output_path = self.output_dir / filename

        # Convert results to a xarray Dataset
        ds = xr.Dataset.from_dict(results)

        # Add metadata
        ds.attrs['variable'] = self.variable
        ds.attrs['experiment'] = self.experiment
        ds.attrs['month'] = self.month
        ds.attrs['latitude'] = self.lat
        ds.attrs['longitude'] = self.lon

        # Save to NetCDF
        ds.to_netcdf(output_path)

        logger.info(f"Results saved to {output_path}")

        return output_path
