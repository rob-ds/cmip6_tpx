"""
Extremes Module

This module provides functions to analyze climate extreme events from historical and
projection data, with support for both single-variable and compound extremes.
"""

import logging
from pathlib import Path
from typing import Dict, Union, Any, Optional, Tuple

import numpy as np
import xarray as xr

from .base_analyzer import BaseAnalyzer

# Configure logger
logger = logging.getLogger(__name__)


class ExtremesAnalyzer(BaseAnalyzer):
    """
    Analyzer for climate extreme events.

    This class computes extreme event metrics, including frequency, persistence,
    and intensity of values exceeding specified thresholds. It supports both
    single-variable extremes and compound extremes (with a secondary variable).
    """

    def __init__(
            self,
            variable: str,
            experiment: str,
            month: int,
            input_dir: Union[str, Path],
            output_dir: Union[str, Path],
            threshold_percentile: float = 95.0,
            secondary_variable: Optional[str] = None,
            secondary_threshold_percentile: float = 95.0,
            wet_day_threshold: float = 1.0,
            dry_day_threshold: float = 1.0,
            heat_wave_min_duration: int = 3
    ):
        """
        Initialize the analyzer.

        Args:
            variable: Primary climate variable to analyze ('temperature' or 'precipitation')
            experiment: Experiment to analyze ('ssp245' or 'ssp585')
            month: Month to analyze (1-12)
            input_dir: Directory containing input data
            output_dir: Directory to store output data
            threshold_percentile: Percentile threshold for extreme events detection
            secondary_variable: Optional secondary variable for compound extremes
            secondary_threshold_percentile: Percentile threshold for secondary variable
            wet_day_threshold: Threshold for wet day identification (mm/day)
            dry_day_threshold: Threshold for dry day identification (mm/day)
        """
        super().__init__(variable, experiment, month, input_dir, output_dir)

        self.threshold_percentile = threshold_percentile
        self.secondary_variable = secondary_variable
        self.secondary_threshold_percentile = secondary_threshold_percentile
        self.wet_day_threshold = wet_day_threshold
        self.dry_day_threshold = dry_day_threshold
        self.heat_wave_min_duration = heat_wave_min_duration

        # Primary thresholds
        self.threshold_value = None

        # Secondary variable data and thresholds
        self.secondary_historical_data = None
        self.secondary_projection_data = None
        self.secondary_threshold_value = None
        self.secondary_nc_var_name = None

        # Initialize containers for results
        self.frequency = None
        self.persistence = None
        self.intensity = None
        self.compound_hot_dry = None
        self.compound_hot_wet = None

        logger.info(f"Initialized {self.__class__.__name__} with {variable}, "
                    f"threshold_percentile={threshold_percentile}")

        if secondary_variable:
            logger.info(f"Secondary variable: {secondary_variable}, "
                        f"threshold_percentile={secondary_threshold_percentile}")

    def load_data(self) -> None:
        """
        Load historical and projection data for the specified variable(s).

        For compound extremes, also loads the secondary variable data.
        """
        # Load primary variable data using parent method
        super().load_data()

        # If secondary variable is specified, load it as well
        if self.secondary_variable:
            logger.info(f"Loading secondary variable: {self.secondary_variable}")

            # Backup primary variable data
            primary_historical = self.historical_data
            primary_projection = self.projection_data
            primary_nc_var_name = self.nc_var_name
            primary_variable = self.variable

            # Set secondary variable name
            self.secondary_nc_var_name = self.var_name_mapping.get(self.secondary_variable)

            if not self.secondary_nc_var_name:
                raise ValueError(f"Invalid secondary variable: {self.secondary_variable}. "
                                 f"Must be one of {list(self.var_name_mapping.keys())}")

            # Temporarily swap variable names
            self.variable = self.secondary_variable
            self.nc_var_name = self.secondary_nc_var_name

            # Load secondary variable
            super().load_data()

            # Store secondary data
            self.secondary_historical_data = self.historical_data
            self.secondary_projection_data = self.projection_data

            # Restore primary variable data
            self.variable = primary_variable
            self.nc_var_name = primary_nc_var_name
            self.historical_data = primary_historical
            self.projection_data = primary_projection

            logger.info(f"Secondary variable loaded: {self.secondary_variable}")

    def _filter_data_by_month(self, data: xr.DataArray) -> xr.DataArray:
        """
        Filter a data array to keep only values from the specified month and add year coordinate.

        Args:
            data: Data array to filter

        Returns:
            Filtered data array with year as a coordinate
        """
        # Extract month from datetime
        months = xr.DataArray(
            data.time.dt.month,
            coords={'time': data.time},
            dims=['time']
        )

        # Filter for the specified month
        filtered_data = data.where(months == self.month, drop=True)

        # Add year as a coordinate for easier grouping
        filtered_data = filtered_data.assign_coords(
            year=("time", filtered_data.time.dt.year.values)
        )

        return filtered_data

    def _filter_month(self) -> None:
        """
        Filter data for the specified month.

        Override to keep daily data (rather than monthly averages).
        """
        logger.info(f"Filtering data for month {self.month}")

        # Filter primary data
        self.historical_data = self._filter_data_by_month(self.historical_data)
        self.projection_data = self._filter_data_by_month(self.projection_data)

        # If secondary variable exists, filter it as well
        if hasattr(self, 'secondary_historical_data') and self.secondary_historical_data is not None:
            self.secondary_historical_data = self._filter_data_by_month(self.secondary_historical_data)
            self.secondary_projection_data = self._filter_data_by_month(self.secondary_projection_data)

        logger.info(f"Historical data shape after filtering: {self.historical_data.shape}")
        logger.info(f"Projection data shape after filtering: {self.projection_data.shape}")

    def _preprocess_precipitation(self, data: xr.DataArray) -> xr.DataArray:
        """
        Preprocess precipitation data by applying wet day threshold.

        Args:
            data: Precipitation data to preprocess

        Returns:
            Preprocessed data with values below wet_day_threshold set to zero
        """
        if self.variable == 'precipitation' or (
                self.secondary_variable == 'precipitation' and
                (data.equals(self.secondary_historical_data) or
                 data.equals(self.secondary_projection_data))):
            logger.info(f"Applying wet day threshold of {self.wet_day_threshold} mm/day")
            return xr.where(data >= self.wet_day_threshold, data, 0)
        return data

    def _calculate_threshold(self) -> None:
        """
        Calculate extreme event thresholds based on historical data.

        For primary and (if present) secondary variables.
        """
        logger.info(f"Calculating thresholds from historical data")

        # Preprocess data if it's precipitation
        historical_data = self._preprocess_precipitation(self.historical_data)

        # Calculate historical statistics
        if self.variable == 'precipitation':
            total_days = len(historical_data.values)
            wet_days = np.sum(historical_data.values > 0)  # Count non-zero days
            wet_days_threshold = np.sum(historical_data.values >= self.wet_day_threshold)

            logger.info(
                f"Precipitation days: {wet_days}/{total_days} days ({wet_days / total_days * 100:.1f}%) > 0mm/day")
            logger.info(
                f"Wet days (≥{self.wet_day_threshold}mm/day): "
                f"{wet_days_threshold}/{total_days} days "
                f"({wet_days_threshold / total_days * 100:.1f}%)")

        # Calculate primary threshold
        self.threshold_value = float(
            np.percentile(historical_data.values, self.threshold_percentile)
        )

        logger.info(f"Primary threshold ({self.threshold_percentile}th percentile): {self.threshold_value}")

        # Calculate secondary threshold if needed
        if self.secondary_variable:
            secondary_historical_data = self._preprocess_precipitation(self.secondary_historical_data)

            self.secondary_threshold_value = float(
                np.percentile(secondary_historical_data.values, self.secondary_threshold_percentile)
            )

            logger.info(f"Secondary threshold ({self.secondary_threshold_percentile}th percentile): "
                        f"{self.secondary_threshold_value}")

    @staticmethod
    def _find_consecutive_runs(binary_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find runs of consecutive True values in a binary array.

        Args:
            binary_array: Boolean array to analyze

        Returns:
            Tuple containing (start_indices, run_lengths)
        """
        # Find runs of consecutive True values
        runs = np.diff(np.concatenate(([0], binary_array, [0])))
        starts = np.where(runs == 1)[0]
        ends = np.where(runs == -1)[0]
        durations = ends - starts

        return starts, durations

    def _calculate_consecutive_days(self, binary_array: np.ndarray) -> int:
        """
        Calculate maximum number of consecutive True values in a binary array.
        """
        # Handle empty array or all False
        if len(binary_array) == 0 or not np.any(binary_array):
            return 0

        # Find runs of consecutive True values
        _, durations = self._find_consecutive_runs(binary_array)

        return np.max(durations) if len(durations) > 0 else 0

    def _calculate_single_variable_extremes(self) -> Dict[str, np.ndarray]:
        """
        Calculate single-variable extreme metrics.

        Returns:
            Dictionary with extreme metrics arrays for each year
        """
        logger.info("Calculating single-variable extreme metrics")

        # Preprocess data if needed
        projection_data = self._preprocess_precipitation(self.projection_data)

        # Get unique years
        years = np.unique(projection_data.year.values)

        # Initialize result arrays for all metrics
        frequency_array = np.zeros(len(years))
        persistence_array = np.zeros(len(years))
        intensity_array = np.zeros(len(years))

        # Initialize heat wave metrics arrays (will only be populated for temperature)
        hw_count_array = np.zeros(len(years))
        hw_days_array = np.zeros(len(years))
        hw_max_duration_array = np.zeros(len(years))
        hw_mean_duration_array = np.zeros(len(years))

        # Calculate metrics for each year
        for i, year in enumerate(years):
            # Get data for this year
            year_mask = projection_data.year == year
            year_data = projection_data.isel(time=year_mask)

            # Create binary array where True means exceeding threshold
            exceeds_threshold = year_data.values > self.threshold_value

            # Frequency: count days exceeding threshold
            frequency_array[i] = np.sum(exceeds_threshold)

            # Persistence: max consecutive days exceeding threshold
            persistence_array[i] = self._calculate_consecutive_days(exceeds_threshold)

            # Intensity: mean value of days exceeding threshold
            if np.any(exceeds_threshold):
                intensity_array[i] = np.mean(year_data.values[exceeds_threshold])
            else:
                intensity_array[i] = np.nan

            # For temperature, calculate heat wave metrics
            if self.variable == 'temperature':
                hw_metrics = self._calculate_heat_waves(exceeds_threshold)
                hw_count_array[i] = hw_metrics["count"]
                hw_days_array[i] = hw_metrics["total_days"]
                hw_max_duration_array[i] = hw_metrics["max_duration"]
                hw_mean_duration_array[i] = hw_metrics["mean_duration"]

        # Build result dictionary (always include basic metrics)
        result = {
            'frequency': frequency_array,
            'persistence': persistence_array,
            'intensity': intensity_array,
            'years': years
        }

        # Add heat wave metrics only for temperature
        if self.variable == 'temperature':
            result.update({
                'hw_count': hw_count_array,
                'hw_days': hw_days_array,
                'hw_max_duration': hw_max_duration_array,
                'hw_mean_duration': hw_mean_duration_array
            })

        return result

    def _calculate_heat_waves(self, binary_exceeded: np.ndarray) -> Dict[str, Any]:
        """
        Calculate heat wave metrics from binary exceedance array.
        """
        if len(binary_exceeded) < self.heat_wave_min_duration:
            return {"count": 0, "total_days": 0, "max_duration": 0, "mean_duration": 0.0}

        # Find runs of consecutive days
        starts, durations = self._find_consecutive_runs(binary_exceeded)

        # Filter for heat waves (min duration criteria)
        heat_wave_indices = durations >= self.heat_wave_min_duration
        heat_wave_durations = durations[heat_wave_indices]

        # Calculate metrics
        hw_count = len(heat_wave_durations)
        hw_total_days = np.sum(heat_wave_durations) if hw_count > 0 else 0
        hw_max_duration = np.max(heat_wave_durations) if hw_count > 0 else 0
        hw_mean_duration = np.mean(heat_wave_durations) if hw_count > 0 else 0

        return {
            "count": hw_count,
            "total_days": hw_total_days,
            "max_duration": hw_max_duration,
            "mean_duration": hw_mean_duration
        }

    def _calculate_compound_extremes(self) -> Dict[str, np.ndarray]:
        """
        Calculate compound extreme metrics (hot-dry and hot-wet days).

        Returns:
            Dictionary with compound extreme metrics arrays for each year
        """
        if not self.secondary_variable:
            logger.warning("No secondary variable specified, skipping compound extremes")
            return {}

        logger.info("Calculating compound extreme metrics")

        # Determine which variable is temperature and which is precipitation
        is_temp_primary = self.variable == 'temperature'

        # Get temperature and precipitation data
        temp_data = self.projection_data if is_temp_primary else self.secondary_projection_data
        precip_data = self.secondary_projection_data if is_temp_primary else self.projection_data

        # Get thresholds
        temp_threshold = self.threshold_value if is_temp_primary else self.secondary_threshold_value
        precip_threshold = self.secondary_threshold_value if is_temp_primary else self.threshold_value

        # Preprocess precipitation data
        precip_data = self._preprocess_precipitation(precip_data)

        # Get unique years
        years = np.unique(temp_data.year.values)

        # Initialize result arrays
        hot_dry_array = np.zeros(len(years))
        hot_wet_array = np.zeros(len(years))

        # Calculate metrics for each year
        for i, year in enumerate(years):
            # Get data for this year
            temp_year_mask = temp_data.year == year
            year_temp = temp_data.isel(time=temp_year_mask)

            precip_year_mask = precip_data.year == year
            year_precip = precip_data.isel(time=precip_year_mask)

            # Ensure matching dimensions
            if len(year_temp) != len(year_precip):
                logger.warning(f"Mismatched dimensions for year {year}, skipping")
                continue

            # Create binary arrays for conditions
            hot_condition = year_temp.values > temp_threshold
            dry_condition = year_precip.values < self.dry_day_threshold
            wet_condition = year_precip.values > precip_threshold

            # Count hot-dry days
            hot_dry_days = np.logical_and(hot_condition, dry_condition)
            hot_dry_array[i] = np.sum(hot_dry_days)

            # Count hot-wet days
            hot_wet_days = np.logical_and(hot_condition, wet_condition)
            hot_wet_array[i] = np.sum(hot_wet_days)

        return {
            'hot_dry_frequency': hot_dry_array,
            'hot_wet_frequency': hot_wet_array,
            'years': years
        }

    def compute(self) -> Dict[str, Any]:
        """
        Compute climate extreme metrics.

        Returns:
            Dictionary containing analysis results, including:
                - frequency: Number of days per month exceeding threshold
                - persistence: Maximum consecutive days above threshold
                - intensity: Mean value of days exceeding threshold
                - heat wave metrics (if temperature variable)
                - compound extremes (if secondary variable provided)
        """
        if not self.is_data_loaded:
            logger.info("Data not loaded. Loading data...")
            self.load_data()

        # Calculate thresholds from historical data
        self._calculate_threshold()

        # Calculate single-variable extremes
        single_var_results = self._calculate_single_variable_extremes()

        # Calculate compound extremes if applicable
        compound_results = self._calculate_compound_extremes() if self.secondary_variable else {}

        # Store results in instance variables
        self.frequency = single_var_results['frequency']
        self.persistence = single_var_results['persistence']
        self.intensity = single_var_results['intensity']

        if self.secondary_variable:
            self.compound_hot_dry = compound_results.get('hot_dry_frequency')
            self.compound_hot_wet = compound_results.get('hot_wet_frequency')

        # Prepare results dictionary for saving
        years = single_var_results['years']

        results = {
            'data_vars': {
                'frequency': {
                    'dims': ['year'],
                    'data': self.frequency,
                    'attrs': {
                        'long_name': f'Frequency of {self.variable} extremes',
                        'units': 'days per month',
                        'description': (f'Number of days per month exceeding the '
                                        f'{self.threshold_percentile}th percentile')
                    }
                },
                'persistence': {
                    'dims': ['year'],
                    'data': self.persistence,
                    'attrs': {
                        'long_name': f'Persistence of {self.variable} extremes',
                        'units': 'days',
                        'description': (f'Maximum consecutive days per month exceeding the '
                                        f'{self.threshold_percentile}th percentile')
                    }
                },
                'intensity': {
                    'dims': ['year'],
                    'data': self.intensity,
                    'attrs': {
                        'long_name': f'Intensity of {self.variable} extremes',
                        'units': 'K' if self.variable == 'temperature' else 'mm/day',
                        'description': f'Mean value of days exceeding the {self.threshold_percentile}th percentile'
                    }
                }
            },
            'coords': {
                'year': {
                    'dims': ['year'],
                    'data': years,
                    'attrs': {
                        'long_name': 'Year',
                        'units': 'year'
                    }
                }
            },
            'attrs': {
                'description': f'Climate extremes for {self.variable}, {self.experiment}, month {self.month}',
                'historical_period': '1995-2014',
                'projection_period': '2015-2100',
                'threshold_percentile': self.threshold_percentile,
                'threshold_value': self.threshold_value
            }
        }

        # Add heat wave metrics for temperature variables
        if self.variable == 'temperature':
            results['data_vars']['hw_count'] = {
                'dims': ['year'],
                'data': single_var_results['hw_count'],
                'attrs': {
                    'long_name': 'Heat wave count',
                    'units': 'events per month',
                    'description': (f'Number of heat wave events per month '
                                    f'(≥{self.heat_wave_min_duration} consecutive days above threshold)')
                }
            }

            results['data_vars']['hw_days'] = {
                'dims': ['year'],
                'data': single_var_results['hw_days'],
                'attrs': {
                    'long_name': 'Heat wave days',
                    'units': 'days per month',
                    'description': f'Total number of days in heat wave conditions per month'
                }
            }

            results['data_vars']['hw_max_duration'] = {
                'dims': ['year'],
                'data': single_var_results['hw_max_duration'],
                'attrs': {
                    'long_name': 'Maximum heat wave duration',
                    'units': 'days',
                    'description': f'Duration of longest heat wave event in the month'
                }
            }

            results['data_vars']['hw_mean_duration'] = {
                'dims': ['year'],
                'data': single_var_results['hw_mean_duration'],
                'attrs': {
                    'long_name': 'Mean heat wave duration',
                    'units': 'days',
                    'description': f'Average duration of heat wave events in the month'
                }
            }

            # Add heat wave definition to attributes
            results['attrs']['heat_wave_min_duration'] = self.heat_wave_min_duration

        # Add compound extremes if applicable
        if self.secondary_variable:
            results['data_vars']['hot_dry_frequency'] = {
                'dims': ['year'],
                'data': self.compound_hot_dry,
                'attrs': {
                    'long_name': 'Frequency of hot-dry days',
                    'units': 'days per month',
                    'description': f'Number of days per month with temperature above the {self.threshold_percentile}th '
                                   f'percentile and precipitation below {self.dry_day_threshold} mm/day'
                }
            }

            results['data_vars']['hot_wet_frequency'] = {
                'dims': ['year'],
                'data': self.compound_hot_wet,
                'attrs': {
                    'long_name': 'Frequency of hot-wet days',
                    'units': 'days per month',
                    'description': (f'Number of days per month with temperature above the '
                                    f'{self.threshold_percentile}th percentile and precipitation above the '
                                    f'{self.secondary_threshold_percentile}th percentile')
                }
            }

            results['attrs']['secondary_variable'] = self.secondary_variable
            results['attrs']['secondary_threshold_percentile'] = self.secondary_threshold_percentile
            results['attrs']['secondary_threshold_value'] = self.secondary_threshold_value

        logger.info("Extremes computation complete")

        return results
