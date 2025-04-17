"""
Base Analyzer Module

This module defines the abstract base class for all analyzers in the CMIP6_TPX project.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from typing import Dict, Union, Any, Tuple

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
            model: str = "ec_earth3_cc",  # Default model for backward compatibility
    ):
        """
        Initialize the analyzer.

        Args:
            variable: Climate variable to analyze ('temperature' or 'precipitation')
            experiment: Experiment to analyze ('ssp245' or 'ssp585')
            month: Month to analyze (1-12)
            input_dir: Directory containing input data
            output_dir: Directory to store output data
            model: CMIP6 model to use (default: ec_earth3_cc for backward compatibility)
        """
        self.variable = variable
        self.experiment = experiment
        self.month = month
        self.model = model  # Store model name

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

        logger.info(f"Initialized {self.__class__.__name__} for {variable}, {experiment}, month {month}, model {model}")

    @staticmethod
    def _format_coordinate(value: float, prefix: str = "") -> str:
        """
        Format a coordinate value for use in filenames.

        Args:
            value: Coordinate value (latitude or longitude)
            prefix: Optional prefix to add (e.g., 'lat' or 'lon')

        Returns:
            Formatted coordinate string (e.g., 'lat38p25' for 38.25)
        """
        formatted = f"{value:.2f}".replace('.', 'p').replace('-', 'n')
        if prefix:
            return f"{prefix}{formatted}"
        return formatted

    @staticmethod
    def _find_center_indices(latitude_array, longitude_array) -> Tuple[int, int]:
        """
        Find the center indices for latitude and longitude dimensions.

        This method handles arrays of any length:
        - For length 1, returns 0
        - For odd lengths, returns the true middle
        - For even lengths, returns the index just before the middle

        Args:
            latitude_array: Array of latitude values
            longitude_array: Array of longitude values

        Returns:
            Tuple of (lat_center_idx, lon_center_idx)
        """
        lat_center_idx = (len(latitude_array) - 1) // 2
        lon_center_idx = (len(longitude_array) - 1) // 2

        return lat_center_idx, lon_center_idx

    @staticmethod
    def _extract_variable_at_center(dataset, variable_name, lat_idx, lon_idx) -> np.ndarray:
        """
        Extract variable values at the center point from a dataset.

        This method handles different dataset structures:
        - 3D arrays with (time, lat, lon) dimensions
        - 1D arrays with only time dimension

        Args:
            dataset: xarray Dataset containing the variable
            variable_name: Name of the variable to extract
            lat_idx: Center latitude index
            lon_idx: Center longitude index

        Returns:
            NumPy array of variable values at the center point
        """
        var_data = dataset[variable_name]

        # Check if the variable has spatial dimensions
        if len(var_data.shape) > 1:  # Multi-dimensional grid
            return var_data[:, lat_idx, lon_idx].values
        else:  # Single point data (only time dimension)
            return var_data[:].values

    def load_data(self) -> None:
        """
        Load historical and projection data for the specified variable and experiment.

        This method loads the data from the input directory, extracts the center point
        from the 3x3 grid, and stores the results in the instance variables.
        """
        # Construct file patterns - handle both new and legacy file formats
        # New format with model and specific coordinates
        historical_pattern = f"cmip6_{self.model}_historical_{self.variable}"
        projection_pattern = f"cmip6_{self.model}_{self.experiment}_{self.variable}"

        # Legacy format (fallback)
        legacy_historical_pattern = f"historical_{self.variable}"
        legacy_projection_pattern = f"{self.experiment}_{self.variable}"

        # Try to find files with new format first, then fall back to legacy format
        try:
            historical_file = self._find_file(self.input_dir / "raw" / "historical", historical_pattern)
            logger.info(f"Found historical data using new filename pattern: {historical_file}")
        except FileNotFoundError:
            # Try legacy format as fallback
            historical_file = self._find_file(self.input_dir / "raw" / "historical", legacy_historical_pattern)
            logger.info(f"Found historical data using legacy filename pattern: {historical_file}")

        try:
            projection_file = self._find_file(self.input_dir / "raw" / "projections", projection_pattern)
            logger.info(f"Found projection data using new filename pattern: {projection_file}")
        except FileNotFoundError:
            # Try legacy format as fallback
            projection_file = self._find_file(self.input_dir / "raw" / "projections", legacy_projection_pattern)
            logger.info(f"Found projection data using legacy filename pattern: {projection_file}")

        logger.info(f"Loading historical data from {historical_file}")
        logger.info(f"Loading projection data from {projection_file}")

        # Load historical data
        historical_ds = xr.open_dataset(historical_file)

        # Load projection data
        projection_ds = xr.open_dataset(projection_file)

        # Find center indices for lat and lon dimensions
        lat_center_idx, lon_center_idx = self._find_center_indices(historical_ds.lat, historical_ds.lon)

        # Extract center point coordinates
        self.lat = float(historical_ds.lat[lat_center_idx].values)
        self.lon = float(historical_ds.lon[lon_center_idx].values)

        logger.info(
            f"Center point coordinates: lat={self.lat}, lon={self.lon} "
            f"(from {len(historical_ds.lat)}×{len(historical_ds.lon)} grid)")

        # Extract center point data for the variable
        historical_var = self._extract_variable_at_center(historical_ds, self.nc_var_name, lat_center_idx,
                                                          lon_center_idx)
        projection_var = self._extract_variable_at_center(projection_ds, self.nc_var_name, lat_center_idx,
                                                          lon_center_idx)

        # Create xarray DataArrays with time as dimension
        self.historical_data = xr.DataArray(
            data=historical_var,
            coords={'time': historical_ds.time.values},
            dims=['time']
        )

        self.projection_data = xr.DataArray(
            data=projection_var,
            coords={'time': projection_ds.time.values},
            dims=['time']
        )

        # Filter for the specified month
        self._filter_month()

        # Convert units if needed
        self._convert_units()

        # Mark as loaded
        self.is_data_loaded = True

        logger.info("Data loading complete")

    def _find_file(self, directory: Path, name_pattern: str) -> Path:
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
        matching_files = []

        for file_path in directory.glob("*.nc"):
            if name_pattern in file_path.name:
                # If we have latitude and longitude, try to match more specifically
                if self.lat is not None and self.lon is not None:
                    lat_str = self._format_coordinate(self.lat, "lat")
                    lon_str = self._format_coordinate(self.lon, "lon")

                    # If file contains the specific coordinate, prioritize it
                    if lat_str in file_path.name and lon_str in file_path.name:
                        logger.debug(f"Found exact coordinate match: {file_path}")
                        return file_path

                matching_files.append(file_path)

        if not matching_files:
            raise FileNotFoundError(f"No file matching '{name_pattern}' found in {directory}")

        # If multiple files match but none with exact coordinates, select the most recent one
        newest_file = sorted(matching_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        logger.debug(f"Selected newest matching file: {newest_file}")
        return newest_file

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
            # Format latitude and longitude for filename
            lat_str = self._format_coordinate(self.lat, "lat")
            lon_str = self._format_coordinate(self.lon, "lon")

            # Create default filename including model
            filename = f"{self.variable}_{self.model}_{self.experiment}_month{self.month:02d}_{lat_str}_{lon_str}.nc"

        output_path = self.output_dir / filename

        # Convert results to a xarray Dataset
        ds = xr.Dataset.from_dict(results)

        # Add metadata
        ds.attrs['variable'] = self.variable
        ds.attrs['experiment'] = self.experiment
        ds.attrs['month'] = self.month
        ds.attrs['latitude'] = self.lat
        ds.attrs['longitude'] = self.lon
        ds.attrs['model'] = self.model  # Add model to metadata

        # Save to NetCDF
        ds.to_netcdf(output_path)

        logger.info(f"Results saved to {output_path}")

        return output_path
