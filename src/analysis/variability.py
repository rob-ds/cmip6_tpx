"""
Variability Module

This module provides functions to analyze climate variability at different time scales.
"""

import logging
from pathlib import Path
from typing import Dict, Union, Any

import numpy as np
import xarray as xr
from scipy import stats

from .base_analyzer import BaseAnalyzer
from .anomalies import AnomalyAnalyzer
from .filters import highpass_filter, lowpass_filter

# Configure logger
logger = logging.getLogger(__name__)


class VariabilityAnalyzer(BaseAnalyzer):
    """
    Analyzer for multiscale climate variability.

    This class decomposes climate anomalies into different time scales:
    - Interannual variability (high-pass filtered)
    - Decadal variability (low-pass filtered)
    - Long-term trend
    - Trend of interannual variability (sliding window standard deviation)
    """

    def __init__(
            self,
            variable: str,
            experiment: str,
            month: int,
            input_dir: Union[str, Path],
            output_dir: Union[str, Path],
            highpass_cutoff: float = 10.0,
            lowpass_cutoff: float = 20.0,
            window_size: int = 11
    ):
        """
        Initialize the analyzer.

        Args:
            variable: Climate variable to analyze ('temperature' or 'precipitation')
            experiment: Experiment to analyze ('ssp245' or 'ssp585')
            month: Month to analyze (1-12)
            input_dir: Directory containing input data
            output_dir: Directory to store output data
            highpass_cutoff: Cutoff period for high-pass filter (years)
            lowpass_cutoff: Cutoff period for low-pass filter (years)
            window_size: Size of sliding window for standard deviation (years)
        """
        super().__init__(variable, experiment, month, input_dir, output_dir)

        self.highpass_cutoff = highpass_cutoff
        self.lowpass_cutoff = lowpass_cutoff
        self.window_size = window_size

        # Initialize containers for derived data
        self.raw_anomalies = None
        self.interannual_variability = None
        self.decadal_variability = None
        self.long_term_trend = None
        self.variability_trend = None

        logger.info(f"Initialized {self.__class__.__name__} with highpass_cutoff={highpass_cutoff}, "
                    f"lowpass_cutoff={lowpass_cutoff}, window_size={window_size}")

    def _standardize_timeseries(self, data: np.ndarray) -> np.ndarray:
        """
        Standardize a time series by dividing by its own standard deviation.
        Only applies to precipitation variables.

        Args:
            data: Time series data to standardize

        Returns:
            Standardized time series if precipitation, original data if temperature
        """
        if self.variable == 'precipitation':
            std_dev = np.std(data)
            if std_dev > 0:
                return data / std_dev
            else:
                logger.warning("Standard deviation is zero, returning original data")
                return data

        # For temperature, return original data
        return data

    def compute(self) -> Dict[str, Any]:
        """
        Compute multiscale climate variability.

        Returns:
            Dictionary containing analysis results, including:
                - raw_anomalies: Raw anomalies from historical climatology
                - interannual_variability: High-pass filtered anomalies
                - decadal_variability: Low-pass filtered anomalies
                - long_term_trend: Long-term trend of anomalies
                - variability_trend: Trend of interannual variability
        """
        if not self.is_data_loaded:
            logger.info("Data not loaded. Loading data...")
            self.load_data()

        logger.info("Computing multi-scale climate variability")

        # First, calculate raw anomalies
        anomaly_analyzer = AnomalyAnalyzer(
            variable=self.variable,
            experiment=self.experiment,
            month=self.month,
            input_dir=self.input_dir,
            output_dir=self.output_dir
        )

        # Use the loaded data instead of loading again
        anomaly_analyzer.historical_data = self.historical_data
        anomaly_analyzer.projection_data = self.projection_data
        anomaly_analyzer.is_data_loaded = True
        anomaly_analyzer.lat = self.lat
        anomaly_analyzer.lon = self.lon

        # Compute anomalies
        anomaly_results = anomaly_analyzer.compute()

        # Extract raw anomalies
        raw_anomalies = xr.DataArray(
            data=anomaly_results['data_vars']['raw_anomalies']['data'],
            coords={'year': anomaly_results['coords']['year']['data']},
            dims=['year']
        )

        # Compute interannual variability (high-pass filter)
        interannual_variability = highpass_filter(
            data=raw_anomalies,
            cutoff_period=self.highpass_cutoff
        )

        # Compute decadal variability (low-pass filter)
        decadal_variability = lowpass_filter(
            data=raw_anomalies,
            cutoff_period=self.lowpass_cutoff
        )

        # Extract years for later use
        years = raw_anomalies.year.values

        # Extract historical interannual values for reference
        historical_interannual = None
        historical_years = None

        if hasattr(self, 'historical_data') and self.historical_data is not None:
            # Calculate historical anomalies
            historical_climatology = float(self.historical_data.mean().values)
            historical_anomalies = self.historical_data.values - historical_climatology

            # Calculate historical interannual variability
            historical_interannual = highpass_filter(
                data=historical_anomalies,
                cutoff_period=self.highpass_cutoff
            )
            historical_years = self.historical_data.year.values

        # Standardize raw anomalies, interannual and decadal variability (for precipitation only)
        standardized_raw_anomalies = self._standardize_timeseries(raw_anomalies.values)
        standardized_interannual = self._standardize_timeseries(interannual_variability)
        standardized_decadal = self._standardize_timeseries(decadal_variability)

        # Compute trend from standardized raw anomalies for precipitation, original for temperature
        trend_data = standardized_raw_anomalies if self.variable == 'precipitation' else raw_anomalies.values
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, trend_data)
        long_term_trend = intercept + slope * years

        # Compute standardized historical interannual variability (for precipitation only)
        standardized_historical_interannual = None
        if historical_interannual is not None and self.variable == 'precipitation':
            standardized_historical_interannual = self._standardize_timeseries(historical_interannual)

        # Compute trend of interannual variability (sliding window standard deviation)
        # For precipitation: apply to standardized interannual variability
        # For temperature: apply to original interannual variability
        years_for_std = []
        std_values = []
        half_window = self.window_size // 2

        # Data for sliding window analysis
        window_data = standardized_interannual if self.variable == 'precipitation' else interannual_variability

        for i in range(half_window, len(years) - half_window):
            window_start = i - half_window
            window_end = i + half_window + 1
            window_values = window_data[window_start:window_end]

            years_for_std.append(years[i])
            std_values.append(np.std(window_values))

        # Calculate historical reference value for variability trend
        historical_std_mean = None

        if historical_interannual is not None and len(historical_years) > self.window_size:
            historical_std_means = []
            for i in range(half_window, len(historical_years) - half_window):
                hist_window_start = i - half_window
                hist_window_end = i + half_window + 1

                # For precipitation: use standardized historical interannual
                # For temperature: use original historical interannual
                if self.variable == 'precipitation' and standardized_historical_interannual is not None:
                    hist_window_values = standardized_historical_interannual[hist_window_start:hist_window_end]
                else:
                    hist_window_values = historical_interannual[hist_window_start:hist_window_end]

                historical_std_means.append(np.std(hist_window_values))

            historical_std_mean = np.mean(historical_std_means)
            logger.info(f"Historical reference value for {self.variable}: {historical_std_mean}")

            # Subtract historical reference
            std_values = np.array(std_values) - historical_std_mean
        else:
            # If historical period is too short or not available
            logger.warning("Historical period too short or not available for proper reference")
            # For precipitation, use 1.0 as reference; for temperature, keep original values
            if self.variable == 'precipitation':
                std_values = np.array(std_values) - 1.0

        # Store results in instance variables
        self.raw_anomalies = raw_anomalies.values
        self.interannual_variability = interannual_variability
        self.decadal_variability = decadal_variability
        self.long_term_trend = long_term_trend
        self.variability_trend = (np.array(years_for_std), np.array(std_values))

        # Units description
        raw_units = 'K' if self.variable == 'temperature' else 'mm/day'
        standardized_units = 'K' if self.variable == 'temperature' else 'standardized units'
        variability_trend_units = 'K' if self.variable == 'temperature' else 'standardized units'
        variability_trend_notes = (
            'anomalies relative to historical variability' if historical_std_mean is not None
            else 'raw values' if self.variable == 'temperature'
            else 'anomalies relative to 1.0'
        )

        # Prepare results for saving
        results = {
            'data_vars': {
                'raw_anomalies': {
                    'dims': ['year'],
                    'data': raw_anomalies.values,
                    'attrs': {
                        'long_name': f'{self.variable} anomalies (raw)',
                        'units': raw_units
                    }
                },
                'standardized_raw_anomalies': {
                    'dims': ['year'],
                    'data': standardized_raw_anomalies,
                    'attrs': {
                        'long_name': f'{self.variable} anomalies (standardized)',
                        'units': standardized_units
                    }
                },
                'interannual_variability': {
                    'dims': ['year'],
                    'data': interannual_variability,
                    'attrs': {
                        'long_name': f'Interannual variability of {self.variable} (raw)',
                        'units': raw_units,
                        'highpass_cutoff': self.highpass_cutoff
                    }
                },
                'standardized_interannual_variability': {
                    'dims': ['year'],
                    'data': standardized_interannual,
                    'attrs': {
                        'long_name': f'Interannual variability of {self.variable} (standardized)',
                        'units': standardized_units,
                        'highpass_cutoff': self.highpass_cutoff
                    }
                },
                'decadal_variability': {
                    'dims': ['year'],
                    'data': decadal_variability,
                    'attrs': {
                        'long_name': f'Decadal variability of {self.variable} (raw)',
                        'units': raw_units,
                        'lowpass_cutoff': self.lowpass_cutoff
                    }
                },
                'standardized_decadal_variability': {
                    'dims': ['year'],
                    'data': standardized_decadal,
                    'attrs': {
                        'long_name': f'Decadal variability of {self.variable} (standardized)',
                        'units': standardized_units,
                        'lowpass_cutoff': self.lowpass_cutoff
                    }
                },
                'long_term_trend': {
                    'dims': ['year'],
                    'data': long_term_trend,
                    'attrs': {
                        'long_name': f'Long-term trend of {self.variable}',
                        'units': standardized_units if self.variable == 'precipitation' else raw_units,
                        'slope': slope,
                        'intercept': intercept,
                        'r_value': r_value,
                        'p_value': p_value,
                        'std_err': std_err
                    }
                },
                'variability_trend': {
                    'dims': ['year_std'],
                    'data': self.variability_trend[1],
                    'attrs': {
                        'long_name': f'Trend of interannual variability of {self.variable}',
                        'units': variability_trend_units,
                        'window_size': self.window_size,
                        'notes': variability_trend_notes,
                        'historical_reference': historical_std_mean
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
                },
                'year_std': {
                    'dims': ['year_std'],
                    'data': self.variability_trend[0],
                    'attrs': {
                        'long_name': 'Year (for std)',
                        'units': 'year'
                    }
                }
            },
            'attrs': {
                'description': (
                    f'Multi-scale climate variability for {self.variable}, '
                    f'{self.experiment}, month {self.month}'
                ),
                'historical_period': '1995-2014',
                'projection_period': '2015-2100',
                'highpass_cutoff': self.highpass_cutoff,
                'lowpass_cutoff': self.lowpass_cutoff,
                'window_size': self.window_size
            }
        }

        logger.info("Variability computation complete")

        return results
