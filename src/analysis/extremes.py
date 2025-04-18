"""
Extremes Module

This module provides functions to analyze climate extreme events from historical and
projection data, following the ETCCDI (Expert Team on Climate Change Detection and Indices)
methodology for standardized climate indices, adapted for daily mean temperature data.

The module includes both standard precipitation indices and modified temperature indices
suitable for daily mean data, as well as compound extreme indices that capture the
relationship between temperature and precipitation extremes.
"""

import logging
from pathlib import Path
from typing import Dict, Union, Any, Tuple

import numpy as np
import xarray as xr
import pandas as pd

from .base_analyzer import BaseAnalyzer

# Configure logger
logger = logging.getLogger(__name__)


class ExtremesAnalyzer(BaseAnalyzer):
    """
    Analyzer for climate extreme events following adapted ETCCDI methodology.

    This class computes standardized extreme event metrics for both temperature and
    precipitation variables, including frequency, intensity, and duration indices.
    It adapts the methods defined by the Expert Team on Climate Change Detection
    and Indices (ETCCDI) for use with daily mean temperature data, and adds
    compound extreme indices that combine temperature and precipitation data.
    """

    def __init__(
            self,
            experiment: str,
            month: int,
            input_dir: Union[str, Path],
            output_dir: Union[str, Path],
            model: str = "ec_earth3_cc",  # Add model parameter with default
            temperature_threshold_percentile: float = 90.0,
            precipitation_threshold_percentile: float = 95.0,
            wet_day_threshold: float = 1.0,
            dry_day_threshold: float = 1.0,
            heat_wave_min_duration: int = 6  # ETCCDI standard is 6 days
    ):
        """
        Initialize the analyzer for both temperature and precipitation variables.

        Args:
            experiment: Experiment to analyze ('ssp245' or 'ssp585')
            month: Month to analyze (1-12)
            input_dir: Directory containing input data
            output_dir: Directory to store output data
            model: CMIP6 model to use (default: ec_earth3_cc for backward compatibility)
            temperature_threshold_percentile: Percentile threshold for temperature extremes (90 for standard ETCCDI)
            precipitation_threshold_percentile: Percentile threshold for precipitation extremes (95 for standard ETCCDI)
            wet_day_threshold: Threshold for wet day identification (1.0 mm/day by ETCCDI standard)
            dry_day_threshold: Threshold for dry day identification in hot-dry analysis (1.0 mm/day by ETCCDI standard)
            heat_wave_min_duration: Minimum consecutive days for heat wave/warm spell (6 days by ETCCDI standard)
        """
        # Call the parent constructor with the default variable to initialize base attributes
        # We'll load both variables later
        super().__init__('temperature', experiment, month, input_dir, output_dir, model)

        # Store thresholds
        self.temperature_threshold_percentile = temperature_threshold_percentile
        self.precipitation_threshold_percentile = precipitation_threshold_percentile
        self.wet_day_threshold = wet_day_threshold
        self.dry_day_threshold = dry_day_threshold
        self.heat_wave_min_duration = heat_wave_min_duration

        # Initialize threshold values (will be calculated from historical data)
        self.temperature_threshold_value = None
        self.temperature_threshold_value_low = None  # For lower percentiles (10th)
        self.temperature_median_value = None  # For median temperature
        self.precipitation_threshold_value = None

        # Initialize data containers for temperature
        self.temperature_historical_data = None
        self.temperature_projection_data = None

        # Initialize data containers for precipitation
        self.precipitation_historical_data = None
        self.precipitation_projection_data = None

        # Initialize result containers - temperature metrics appropriate for daily mean data
        self.tm_max = None  # Maximum daily mean temperature
        self.tm_min = None  # Minimum daily mean temperature
        self.tm90p = None  # Percentage of days when daily mean temperature > 90th percentile
        self.tm10p = None  # Percentage of days when daily mean temperature < 10th percentile
        self.warm_spell_days = None  # Count of days in warm spells (≥ consecutive days > 90th percentile)
        self.cold_spell_days = None  # Count of days in cold spells (≥ consecutive days < 10th percentile)

        # Initialize result containers - precipitation metrics
        self.r95p = None  # Precipitation amount on very wet days
        self.prcptot = None  # Total precipitation
        self.rx1day = None  # Max 1-day precipitation
        self.rx5day = None  # Max 5-day precipitation
        self.r10mm = None  # Number of days with precip >= 10mm
        self.r20mm = None  # Number of days with precip >= 20mm
        self.sdii = None  # Simple daily intensity index
        self.cdd = None  # Consecutive dry days
        self.cwd = None  # Consecutive wet days

        # Initialize result containers - basic compound extreme metrics
        self.hot_dry_frequency = None  # Count of hot-dry days
        self.hot_wet_frequency = None  # Count of hot-wet days
        self.cold_wet_frequency = None  # Count of cold-wet days

        # Initialize result containers - advanced compound metrics
        self.cwhd = None  # Consecutive wet-hot days
        self.wspi = None  # Warm-period precipitation intensity
        self.wpd = None  # Warm precipitation days

        # Track processing state
        self.is_data_loaded = False

        logger.info(f"Initialized {self.__class__.__name__} for both temperature and precipitation, "
                    f"{experiment}, month {month}, model {model}")

    def load_data(self) -> None:
        """
        Load historical and projection data for both temperature and precipitation.
        """
        from src.utils.netcdf_utils import find_and_load_cmip6_data

        logger.info("Loading temperature and precipitation data")

        # Load temperature data
        (temp_historical_ds, temp_projection_ds, self.lat, self.lon,
         lat_center_idx, lon_center_idx) = find_and_load_cmip6_data(
            self.input_dir,
            'temperature',
            self.experiment,
            self.model,
            logger,
            coords=(self.lat, self.lon) if hasattr(self, 'lat') and self.lat is not None else None
        )

        # Extract temperature data
        temp_historical_var = self._extract_variable_at_center(temp_historical_ds, 'tas', lat_center_idx,
                                                               lon_center_idx)
        temp_projection_var = self._extract_variable_at_center(temp_projection_ds, 'tas', lat_center_idx,
                                                               lon_center_idx)

        # Create temperature DataArrays
        self.temperature_historical_data = xr.DataArray(
            data=temp_historical_var,
            coords={'time': temp_historical_ds.time.values},
            dims=['time']
        )
        self.temperature_projection_data = xr.DataArray(
            data=temp_projection_var,
            coords={'time': temp_projection_ds.time.values},
            dims=['time']
        )

        # Load precipitation data (pass coordinates to avoid recalculation)
        precip_historical_ds, precip_projection_ds, _, _, lat_center_idx, lon_center_idx = find_and_load_cmip6_data(
            self.input_dir,
            'precipitation',
            self.experiment,
            self.model,
            logger,
            coords=(self.lat, self.lon)
        )

        # Extract precipitation data
        precip_historical_var = self._extract_variable_at_center(precip_historical_ds, 'pr', lat_center_idx,
                                                                 lon_center_idx)
        precip_projection_var = self._extract_variable_at_center(precip_projection_ds, 'pr', lat_center_idx,
                                                                 lon_center_idx)

        # Create precipitation DataArrays
        self.precipitation_historical_data = xr.DataArray(
            data=precip_historical_var,
            coords={'time': precip_historical_ds.time.values},
            dims=['time']
        )
        self.precipitation_projection_data = xr.DataArray(
            data=precip_projection_var,
            coords={'time': precip_projection_ds.time.values},
            dims=['time']
        )

        # Convert precipitation units
        logger.info("Converting precipitation units from kg.m-2.s-1 to mm.day-1")
        self.precipitation_historical_data = self.precipitation_historical_data * 86400
        self.precipitation_projection_data = self.precipitation_projection_data * 86400

        # Filter data for the specified month
        self._filter_data_for_month()

        # Mark as loaded
        self.is_data_loaded = True

        logger.info("Data loading complete for both temperature and precipitation")

    def _filter_data_for_month(self):
        """
        Filter both temperature and precipitation data for the specified month.
        """
        logger.info(f"Filtering data for month {self.month}")

        # Filter temperature data
        self.temperature_historical_data = self._filter_data_by_month(self.temperature_historical_data)
        self.temperature_projection_data = self._filter_data_by_month(self.temperature_projection_data)

        # Filter precipitation data
        self.precipitation_historical_data = self._filter_data_by_month(self.precipitation_historical_data)
        self.precipitation_projection_data = self._filter_data_by_month(self.precipitation_projection_data)

        logger.info(f"Temperature historical data shape after filtering: {self.temperature_historical_data.shape}")
        logger.info(f"Temperature projection data shape after filtering: {self.temperature_projection_data.shape}")
        logger.info(f"Precipitation historical data shape after filtering: {self.precipitation_historical_data.shape}")
        logger.info(f"Precipitation projection data shape after filtering: {self.precipitation_projection_data.shape}")

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

    def _identify_wet_days(self, data: xr.DataArray) -> np.ndarray:
        """
        Identify wet days in precipitation data using ETCCDI standard (>= 1mm).

        Args:
            data: Precipitation data array

        Returns:
            Boolean mask where True indicates wet days
        """
        return data.values >= self.wet_day_threshold

    def _calculate_threshold(self) -> None:
        """
        Calculate extreme event thresholds based on historical data for both variables.

        Follows ETCCDI methodology:
        - For precipitation, percentiles are calculated from wet days only
        - For temperature, both high (90th), median (50th) and low (10th) percentiles are calculated
        """
        logger.info("Calculating thresholds from historical data for both variables")

        # Calculate temperature thresholds
        self.temperature_threshold_value = float(
            np.percentile(self.temperature_historical_data.values, self.temperature_threshold_percentile)
        )
        self.temperature_threshold_value_low = float(
            np.percentile(self.temperature_historical_data.values, 10.0)  # 10th percentile for cold extremes
        )
        self.temperature_median_value = float(
            np.percentile(self.temperature_historical_data.values, 50.0)  # Median for warm period metrics
        )

        logger.info(f"Temperature upper threshold ({self.temperature_threshold_percentile}th percentile): "
                    f"{self.temperature_threshold_value}")
        logger.info(f"Temperature lower threshold (10th percentile): {self.temperature_threshold_value_low}")
        logger.info(f"Temperature median (50th percentile): {self.temperature_median_value}")

        # For precipitation: calculate percentiles from wet days only (>= 1mm)
        wet_day_mask = self._identify_wet_days(self.precipitation_historical_data)
        wet_day_values = self.precipitation_historical_data.values[wet_day_mask]

        if len(wet_day_values) == 0:
            logger.warning(f"No wet days found in historical data for month {self.month}")
            self.precipitation_threshold_value = self.wet_day_threshold
        else:
            self.precipitation_threshold_value = float(
                np.percentile(wet_day_values, self.precipitation_threshold_percentile)
            )

        # Calculate precipitation statistics
        total_days = len(self.precipitation_historical_data.values)
        wet_days = np.sum(wet_day_mask)

        logger.info(
            f"Wet days (≥{self.wet_day_threshold}mm/day): "
            f"{wet_days}/{total_days} days "
            f"({wet_days / total_days * 100:.1f}%)")
        logger.info(f"Precipitation threshold ({self.precipitation_threshold_percentile}th percentile): "
                    f"{self.precipitation_threshold_value}")

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

    def _count_consecutive_days(self, binary_array: np.ndarray, min_duration: int = 1) -> int:
        """
        Count total number of days in runs of consecutive True values that meet minimum duration.

        Args:
            binary_array: Boolean array indicating condition is met
            min_duration: Minimum number of consecutive days to count

        Returns:
            Total count of days in qualifying runs
        """
        # Handle empty array or all False
        if len(binary_array) == 0 or not np.any(binary_array):
            return 0

        # Find runs of consecutive True values
        _, durations = self._find_consecutive_runs(binary_array)

        # Sum days in runs that meet minimum duration
        qualifying_runs = durations[durations >= min_duration]

        return np.sum(qualifying_runs) if len(qualifying_runs) > 0 else 0

    @staticmethod
    def _calculate_running_sum(data: np.ndarray, window: int) -> np.ndarray:
        """
        Calculate running sum of values over specified window.

        Args:
            data: Input data array
            window: Window size for running sum

        Returns:
            Array of running sums
        """
        # Use pandas rolling window for simplicity
        return pd.Series(data).rolling(window=window).sum().values

    def _calculate_temperature_indices(self, years: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate temperature indices adapted for daily mean temperature data.

        This method computes indices that are appropriate for daily mean temperature
        rather than the standard ETCCDI indices designed for daily maximum and minimum.

        Args:
            years: Array of unique years

        Returns:
            Dictionary with temperature indices arrays for each year
        """
        logger.info("Calculating temperature indices for daily mean temperature")

        # Initialize result arrays
        tm_max_array = np.zeros(len(years))  # Maximum daily mean temperature
        tm_min_array = np.zeros(len(years))  # Minimum daily mean temperature
        tm90p_array = np.zeros(len(years))  # Percentage of days with daily mean temperature > 90th percentile
        tm10p_array = np.zeros(len(years))  # Percentage of days with daily mean temperature < 10th percentile
        warm_spell_days_array = np.zeros(len(years))  # Count of days in warm spells
        cold_spell_days_array = np.zeros(len(years))  # Count of days in cold spells

        # Calculate indices for each year
        for i, year in enumerate(years):
            # Get data for this year
            year_mask = self.temperature_projection_data.year == year
            year_data = self.temperature_projection_data.isel(time=year_mask)

            # Skip if no data for this year
            if len(year_data) == 0:
                continue

            # TM_max: Maximum daily mean temperature
            tm_max_array[i] = np.max(year_data.values)

            # TM_min: Minimum daily mean temperature
            tm_min_array[i] = np.min(year_data.values)

            # TM90p: Percentage of days when daily mean temperature > 90th percentile
            tm90p_array[i] = 100.0 * np.mean(year_data.values > self.temperature_threshold_value)

            # TM10p: Percentage of days when daily mean temperature < 10th percentile
            tm10p_array[i] = 100.0 * np.mean(year_data.values < self.temperature_threshold_value_low)

            # Warm spell days: Number of days in warm spells (≥ consecutive days > 90th percentile)
            warm_days = np.array(year_data.values > self.temperature_threshold_value)
            warm_spell_days_array[i] = self._count_consecutive_days(warm_days, self.heat_wave_min_duration)

            # Cold spell days: Number of days in cold spells (≥ consecutive days < 10th percentile)
            cold_days = np.array(year_data.values < self.temperature_threshold_value_low)
            cold_spell_days_array[i] = self._count_consecutive_days(cold_days, self.heat_wave_min_duration)

        return {
            'tm_max': tm_max_array,
            'tm_min': tm_min_array,
            'tm90p': tm90p_array,
            'tm10p': tm10p_array,
            'warm_spell_days': warm_spell_days_array,
            'cold_spell_days': cold_spell_days_array
        }

    def _calculate_precipitation_indices(self, years: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate ETCCDI precipitation indices for monthly analysis.

        This method computes standard precipitation indices on a monthly basis
        rather than the annual scale typically used by ETCCDI.

        Args:
            years: Array of unique years

        Returns:
            Dictionary with precipitation indices arrays for each year
        """
        logger.info("Calculating precipitation indices")

        # Initialize result arrays
        r95p_array = np.zeros(len(years))
        prcptot_array = np.zeros(len(years))
        rx1day_array = np.zeros(len(years))
        rx5day_array = np.zeros(len(years))
        r10mm_array = np.zeros(len(years))
        r20mm_array = np.zeros(len(years))
        sdii_array = np.zeros(len(years))
        cdd_array = np.zeros(len(years))
        cwd_array = np.zeros(len(years))

        # Calculate indices for each year
        for i, year in enumerate(years):
            # Get data for this year
            year_mask = self.precipitation_projection_data.year == year
            year_data = self.precipitation_projection_data.isel(time=year_mask)

            # Skip if no data for this year
            if len(year_data) == 0:
                continue

            # Create wet day mask (≥ 1mm)
            wet_day_mask = self._identify_wet_days(year_data)
            wet_day_values = year_data.values[wet_day_mask]

            # R95p: Monthly total precipitation when daily precipitation > 95th percentile
            very_wet_days = year_data.values > self.precipitation_threshold_value
            r95p_array[i] = np.sum(year_data.values[very_wet_days]) if np.any(very_wet_days) else 0

            # PRCPTOT: Monthly total precipitation on wet days
            prcptot_array[i] = np.sum(wet_day_values)

            # RX1day: Maximum 1-day precipitation
            rx1day_array[i] = np.max(year_data.values) if len(year_data) > 0 else 0

            # RX5day: Maximum 5-day precipitation (within month)
            if len(year_data) >= 5:
                rolling_sums = self._calculate_running_sum(year_data.values, 5)
                rx5day_array[i] = np.nanmax(rolling_sums) if np.any(~np.isnan(rolling_sums)) else 0
            else:
                rx5day_array[i] = np.sum(year_data.values)  # Sum all days if less than 5

            # R10mm: Monthly count of days when precipitation ≥ 10mm
            r10mm_array[i] = np.sum(year_data.values >= 10.0)

            # R20mm: Monthly count of days when precipitation ≥ 20mm
            r20mm_array[i] = np.sum(year_data.values >= 20.0)

            # SDII: Simple daily intensity index (mean precipitation on wet days)
            sdii_array[i] = np.mean(wet_day_values) if len(wet_day_values) > 0 else 0

            # CDD: Maximum consecutive dry days within month
            dry_days = year_data.values < self.wet_day_threshold
            cdd_array[i] = self._calculate_consecutive_days(dry_days)

            # CWD: Maximum consecutive wet days within month
            cwd_array[i] = self._calculate_consecutive_days(wet_day_mask)

        return {
            'r95p': r95p_array,
            'prcptot': prcptot_array,
            'rx1day': rx1day_array,
            'rx5day': rx5day_array,
            'r10mm': r10mm_array,
            'r20mm': r20mm_array,
            'sdii': sdii_array,
            'cdd': cdd_array,
            'cwd': cwd_array
        }

    def _calculate_basic_compound_extremes(self, years: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate basic compound extreme metrics (hot-dry, hot-wet, cold-wet days).

        Args:
            years: Array of unique years

        Returns:
            Dictionary with basic compound extreme metrics arrays for each year
        """
        logger.info("Calculating basic compound extreme metrics")

        # Initialize result arrays
        hot_dry_array = np.zeros(len(years))
        hot_wet_array = np.zeros(len(years))
        cold_wet_array = np.zeros(len(years))

        # Calculate metrics for each year
        for i, year in enumerate(years):
            # Get temperature and precipitation data for this year
            temp_year_mask = self.temperature_projection_data.year == year
            year_temp = self.temperature_projection_data.isel(time=temp_year_mask)

            precip_year_mask = self.precipitation_projection_data.year == year
            year_precip = self.precipitation_projection_data.isel(time=precip_year_mask)

            # Ensure matching dimensions
            if len(year_temp) != len(year_precip):
                logger.warning(f"Mismatched dimensions for year {year}, skipping")
                continue

            # Create binary arrays for conditions
            hot_condition = year_temp.values > self.temperature_threshold_value  # Warm days
            cold_condition = year_temp.values < self.temperature_threshold_value_low  # Cold days
            dry_condition = year_precip.values < self.wet_day_threshold  # Dry days
            wet_condition = year_precip.values > self.precipitation_threshold_value  # Very wet days

            # Count compound extreme days
            hot_dry_days = np.logical_and(hot_condition, dry_condition)
            hot_dry_array[i] = np.sum(hot_dry_days)

            hot_wet_days = np.logical_and(hot_condition, wet_condition)
            hot_wet_array[i] = np.sum(hot_wet_days)

            cold_wet_days = np.logical_and(cold_condition, wet_condition)
            cold_wet_array[i] = np.sum(cold_wet_days)

        return {
            'hot_dry_frequency': hot_dry_array,
            'hot_wet_frequency': hot_wet_array,
            'cold_wet_frequency': cold_wet_array
        }

    def _calculate_advanced_compound_metrics(self, years: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate advanced compound extreme metrics.

        This method computes three compound extreme indices:
        1. CWHD: Consecutive Wet-Hot Days - maximum number of consecutive days
                 when both precipitation exceeds 95th percentile and
                 temperature exceeds 90th percentile
        2. WSPI: Warm-Period Precipitation Intensity - maximum 5-day precipitation
                 during periods when temperature exceeds median
        3. WPD: Warm Precipitation Days - count of days with precipitation >1mm
                when temperature >0°C

        Args:
            years: Array of unique years

        Returns:
            Dictionary with advanced compound extreme metrics arrays for each year
        """
        logger.info("Calculating advanced compound extreme metrics")

        # Initialize result arrays
        cwhd_array = np.zeros(len(years))  # Consecutive Wet-Hot Days
        wspi_array = np.zeros(len(years))  # Warm-Period Precipitation Intensity
        wpd_array = np.zeros(len(years))  # Warm Precipitation Days

        # Calculate metrics for each year
        for i, year in enumerate(years):
            # Get temperature and precipitation data for this year
            temp_year_mask = self.temperature_projection_data.year == year
            year_temp = self.temperature_projection_data.isel(time=temp_year_mask)

            precip_year_mask = self.precipitation_projection_data.year == year
            year_precip = self.precipitation_projection_data.isel(time=precip_year_mask)

            # Ensure matching dimensions
            if len(year_temp) != len(year_precip):
                logger.warning(f"Mismatched dimensions for year {year}, skipping")
                continue

            # 1. Calculate CWHD: Consecutive Wet-Hot Days
            # Define days that are both very wet and hot
            wet_hot_days = np.logical_and(
                year_precip.values > self.precipitation_threshold_value,  # Very wet days
                year_temp.values > self.temperature_threshold_value  # Hot days
            )

            # Find maximum consecutive wet-hot days
            cwhd_array[i] = self._calculate_consecutive_days(wet_hot_days)

            # 2. Calculate WSPI: Warm-Period Precipitation Intensity
            # Identify warm days (above median temperature)
            warm_days = year_temp.values > self.temperature_median_value

            # Find runs of consecutive warm days
            warm_starts, warm_durations = self._find_consecutive_runs(np.array(warm_days))
            wspi_value = 0.0

            # For each warm period of at least 5 days
            for start, duration in zip(warm_starts, warm_durations):
                if duration >= 5:
                    # Calculate maximum 5-day precipitation in this warm period
                    for j in range(0, duration - 4):
                        # Calculate 5-day sum starting at each position in warm period
                        precip_window = year_precip.values[start + j:start + j + 5]
                        window_sum = np.sum(precip_window)
                        wspi_value = max(wspi_value, window_sum)

            wspi_array[i] = wspi_value

            # 3. Calculate WPD: Warm Precipitation Days
            # Count days with precipitation >1mm when temperature >0°C
            freezing_temp_kelvin = 273.15  # 0°C in Kelvin
            above_freezing = year_temp.values > freezing_temp_kelvin
            above_wetday = year_precip.values >= self.wet_day_threshold

            # Count days meeting both conditions
            wpd_array[i] = np.sum(np.logical_and(above_freezing, above_wetday))

        return {
            'cwhd': cwhd_array,
            'wspi': wspi_array,
            'wpd': wpd_array
        }

    def compute(self) -> Dict[str, Any]:
        """
        Compute climate extreme indices following adapted ETCCDI methodology for both
        temperature and precipitation, including compound extreme indices.

        Returns:
            Dictionary containing analysis results with all calculated indices.
        """
        if not self.is_data_loaded:
            logger.info("Data not loaded. Loading data...")
            self.load_data()

        # Calculate thresholds from historical data
        self._calculate_threshold()

        # Get unique years from projection data
        years = np.unique(self.temperature_projection_data.year.values)

        # Calculate temperature indices
        temp_indices = self._calculate_temperature_indices(years)

        # Store temperature results in instance variables
        self.tm_max = temp_indices['tm_max']
        self.tm_min = temp_indices['tm_min']
        self.tm90p = temp_indices['tm90p']
        self.tm10p = temp_indices['tm10p']
        self.warm_spell_days = temp_indices['warm_spell_days']
        self.cold_spell_days = temp_indices['cold_spell_days']

        # Calculate precipitation indices
        precip_indices = self._calculate_precipitation_indices(years)

        # Store precipitation results in instance variables
        self.r95p = precip_indices['r95p']
        self.prcptot = precip_indices['prcptot']
        self.rx1day = precip_indices['rx1day']
        self.rx5day = precip_indices['rx5day']
        self.r10mm = precip_indices['r10mm']
        self.r20mm = precip_indices['r20mm']
        self.sdii = precip_indices['sdii']
        self.cdd = precip_indices['cdd']
        self.cwd = precip_indices['cwd']

        # Calculate basic compound extremes
        basic_compound_results = self._calculate_basic_compound_extremes(years)

        # Store basic compound results
        self.hot_dry_frequency = basic_compound_results['hot_dry_frequency']
        self.hot_wet_frequency = basic_compound_results['hot_wet_frequency']
        self.cold_wet_frequency = basic_compound_results['cold_wet_frequency']

        # Calculate advanced compound metrics
        advanced_compound_results = self._calculate_advanced_compound_metrics(years)

        # Store advanced compound results
        self.cwhd = advanced_compound_results['cwhd']
        self.wspi = advanced_compound_results['wspi']
        self.wpd = advanced_compound_results['wpd']

        # Prepare results dictionary
        results = self._prepare_results_dict(years)

        logger.info("Extremes computation complete")

        return results

    def _prepare_results_dict(self, years: np.ndarray) -> Dict[str, Any]:
        """
        Prepare comprehensive results dictionary for saving to netCDF.

        Args:
            years: Array of years

        Returns:
            Dictionary formatted for xarray dataset creation
        """
        # Create base results dictionary with coordinates
        results = {
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
                'description': f'Climate extremes for {self.experiment}, month {self.month}, model {self.model}',
                'historical_period': '1995-2014',
                'projection_period': '2015-2100',
                'methodology': 'Adapted ETCCDI for daily mean temperature data',
                'temperature_threshold_percentile': self.temperature_threshold_percentile,
                'temperature_threshold_value': self.temperature_threshold_value,
                'temperature_threshold_value_low': self.temperature_threshold_value_low,
                'temperature_median_value': self.temperature_median_value,
                'precipitation_threshold_percentile': self.precipitation_threshold_percentile,
                'precipitation_threshold_value': self.precipitation_threshold_value,
                'wet_day_threshold': self.wet_day_threshold,
                'dry_day_threshold': self.dry_day_threshold,
                'heat_wave_min_duration': self.heat_wave_min_duration,
                'model': self.model  # Add model to attributes
            },
            'data_vars': {}
        }

        # Add temperature indices appropriate for daily mean data
        results['data_vars']['tm_max'] = {
            'dims': ['year'],
            'data': self.tm_max,
            'attrs': {
                'long_name': 'Maximum daily mean temperature',
                'units': 'K',
                'description': 'Maximum value of daily mean temperature in the month'
            }
        }

        results['data_vars']['tm_min'] = {
            'dims': ['year'],
            'data': self.tm_min,
            'attrs': {
                'long_name': 'Minimum daily mean temperature',
                'units': 'K',
                'description': 'Minimum value of daily mean temperature in the month'
            }
        }

        results['data_vars']['tm90p'] = {
            'dims': ['year'],
            'data': self.tm90p,
            'attrs': {
                'long_name': 'Warm days (daily mean)',
                'units': '%',
                'description': 'Percentage of days when daily mean temperature > 90th percentile'
            }
        }

        results['data_vars']['tm10p'] = {
            'dims': ['year'],
            'data': self.tm10p,
            'attrs': {
                'long_name': 'Cold days (daily mean)',
                'units': '%',
                'description': 'Percentage of days when daily mean temperature < 10th percentile'
            }
        }

        results['data_vars']['warm_spell_days'] = {
            'dims': ['year'],
            'data': self.warm_spell_days,
            'attrs': {
                'long_name': 'Warm spell days',
                'units': 'days',
                'description': (f'Monthly count of days in spells with at least {self.heat_wave_min_duration} '
                                'consecutive days when daily mean temperature > 90th percentile')
            }
        }

        results['data_vars']['cold_spell_days'] = {
            'dims': ['year'],
            'data': self.cold_spell_days,
            'attrs': {
                'long_name': 'Cold spell days',
                'units': 'days',
                'description': (f'Monthly count of days in spells with at least {self.heat_wave_min_duration} '
                                'consecutive days when daily mean temperature < 10th percentile')
            }
        }

        # Add precipitation indices
        results['data_vars']['r95p'] = {
            'dims': ['year'],
            'data': self.r95p,
            'attrs': {
                'long_name': 'Very wet days',
                'units': 'mm',
                'description': (f'Monthly total precipitation when daily precipitation > '
                                f'{self.precipitation_threshold_percentile}th percentile of wet days')
            }
        }

        results['data_vars']['prcptot'] = {
            'dims': ['year'],
            'data': self.prcptot,
            'attrs': {
                'long_name': 'Total precipitation',
                'units': 'mm',
                'description': 'Monthly total precipitation on wet days (≥ 1mm)'
            }
        }

        results['data_vars']['rx1day'] = {
            'dims': ['year'],
            'data': self.rx1day,
            'attrs': {
                'long_name': 'Maximum 1-day precipitation',
                'units': 'mm',
                'description': 'Monthly maximum 1-day precipitation'
            }
        }

        results['data_vars']['rx5day'] = {
            'dims': ['year'],
            'data': self.rx5day,
            'attrs': {
                'long_name': 'Maximum 5-day precipitation',
                'units': 'mm',
                'description': 'Monthly maximum consecutive 5-day precipitation'
            }
        }

        results['data_vars']['r10mm'] = {
            'dims': ['year'],
            'data': self.r10mm,
            'attrs': {
                'long_name': 'Heavy precipitation days',
                'units': 'days',
                'description': 'Monthly count of days when precipitation ≥ 10mm'
            }
        }

        results['data_vars']['r20mm'] = {
            'dims': ['year'],
            'data': self.r20mm,
            'attrs': {
                'long_name': 'Very heavy precipitation days',
                'units': 'days',
                'description': 'Monthly count of days when precipitation ≥ 20mm'
            }
        }

        results['data_vars']['sdii'] = {
            'dims': ['year'],
            'data': self.sdii,
            'attrs': {
                'long_name': 'Simple daily intensity index',
                'units': 'mm/day',
                'description': 'Monthly mean precipitation on wet days'
            }
        }

        results['data_vars']['cdd'] = {
            'dims': ['year'],
            'data': self.cdd,
            'attrs': {
                'long_name': 'Consecutive dry days',
                'units': 'days',
                'description': 'Maximum number of consecutive days with precipitation < 1mm within the month'
            }
        }

        results['data_vars']['cwd'] = {
            'dims': ['year'],
            'data': self.cwd,
            'attrs': {
                'long_name': 'Consecutive wet days',
                'units': 'days',
                'description': 'Maximum number of consecutive days with precipitation ≥ 1mm within the month'
            }
        }

        # Add basic compound extremes
        results['data_vars']['hot_dry_frequency'] = {
            'dims': ['year'],
            'data': self.hot_dry_frequency,
            'attrs': {
                'long_name': 'Frequency of hot-dry days',
                'units': 'days',
                'description': f'Number of days with daily mean temperature above the '
                               f'{self.temperature_threshold_percentile}th '
                               f'percentile and precipitation below {self.wet_day_threshold} mm/day'
            }
        }

        results['data_vars']['hot_wet_frequency'] = {
            'dims': ['year'],
            'data': self.hot_wet_frequency,
            'attrs': {
                'long_name': 'Frequency of hot-wet days',
                'units': 'days',
                'description': f'Number of days with daily mean temperature above the '
                               f'{self.temperature_threshold_percentile}th '
                               f'percentile and precipitation above the '
                               f'{self.precipitation_threshold_percentile}th percentile'
            }
        }

        results['data_vars']['cold_wet_frequency'] = {
            'dims': ['year'],
            'data': self.cold_wet_frequency,
            'attrs': {
                'long_name': 'Frequency of cold-wet days',
                'units': 'days',
                'description': f'Number of days with daily mean temperature below the 10th percentile '
                               f'and precipitation above the {self.precipitation_threshold_percentile}th percentile'
            }
        }

        # Add advanced compound metrics
        results['data_vars']['cwhd'] = {
            'dims': ['year'],
            'data': self.cwhd,
            'attrs': {
                'long_name': 'Consecutive wet-hot days',
                'units': 'days',
                'description': f'Maximum number of consecutive days with both precipitation above the '
                               f'{self.precipitation_threshold_percentile}th percentile and daily mean temperature '
                               f'above the {self.temperature_threshold_percentile}th percentile'
            }
        }

        results['data_vars']['wspi'] = {
            'dims': ['year'],
            'data': self.wspi,
            'attrs': {
                'long_name': 'Warm-period precipitation intensity',
                'units': 'mm',
                'description': 'Maximum 5-day precipitation total that occurs during periods when daily mean '
                               'temperature exceeds the monthly median temperature'
            }
        }

        results['data_vars']['wpd'] = {
            'dims': ['year'],
            'data': self.wpd,
            'attrs': {
                'long_name': 'Warm precipitation days',
                'units': 'days',
                'description': 'Number of days with precipitation ≥1mm occurring when daily mean temperature '
                               'is above freezing (0°C)'
            }
        }

        return results

    def save_results(self, results: Dict[str, Any], filename: str = None) -> Path:
        """
        Save results to a single NetCDF file.

        Args:
            results: Dictionary with results data
            filename: Optional filename override

        Returns:
            Path to output file
        """
        if filename is None:
            # Format latitude and longitude for filename
            lat_str = self._format_coordinate(self.lat, "lat")
            lon_str = self._format_coordinate(self.lon, "lon")

            # Create default filename including model and coordinates
            filename = f"extremes_{self.model}_{self.experiment}_month{self.month:02d}_{lat_str}_{lon_str}.nc"

        # Use parent class method to save the file
        output_file = super().save_results(results, filename=filename)

        return output_file
