"""
Anomalies Module

This module provides functions to calculate climate anomalies from historical and
projection data.
"""

import logging
from typing import Dict, Any, Union
from pathlib import Path

from .base_analyzer import BaseAnalyzer

# Configure logger
logger = logging.getLogger(__name__)


class AnomalyAnalyzer(BaseAnalyzer):
    """
    Analyzer for calculating climate anomalies.

    This class computes raw anomalies from historical climatology for
    the specified variable, experiment, and month.
    """

    def __init__(
            self,
            variable: str,
            experiment: str,
            month: int,
            input_dir: Union[str, Path],
            output_dir: Union[str, Path],
            model: str = "ec_earth3_cc"  # Add model parameter with default
    ):
        """
        Initialize the analyzer.

        Args:
            variable: Climate variable to analyze ('temperature' or 'precipitation')
            experiment: Experiment to analyze ('ssp245' or 'ssp585')
            month: Month to analyze (1-12)
            input_dir: Directory containing input data
            output_dir: Directory to store output data
            model: CMIP6 model to use
        """
        super().__init__(variable, experiment, month, input_dir, output_dir, model)

    def compute(self) -> Dict[str, Any]:
        """
        Compute climate anomalies.

        Returns:
            Dictionary containing analysis results, including:
                - historical_data: Historical time series
                - historical_climatology: Historical climatological mean
                - projection_data: Projection time series
                - raw_anomalies: Raw anomalies from historical climatology
        """
        if not self.is_data_loaded:
            logger.info("Data not loaded. Loading data...")
            self.load_data()

        logger.info("Computing climate anomalies")

        # Calculate historical climatology (mean over all years)
        historical_climatology = float(self.historical_data.mean().values)

        logger.info(f"Historical climatology for {self.variable}, month {self.month}: {historical_climatology}")

        # Calculate raw anomalies (departure from historical climatology)
        raw_anomalies = self.projection_data - historical_climatology

        # Prepare results
        results = {
            'data_vars': {
                'historical_data': {
                    'dims': ['year'],
                    'data': self.historical_data.values,
                    'attrs': {
                        'long_name': f'Historical {self.variable} data',
                        'units': 'K' if self.variable == 'temperature' else 'mm/day'
                    }
                },
                'historical_climatology': {
                    'dims': [],
                    'data': historical_climatology,
                    'attrs': {
                        'long_name': f'Historical {self.variable} climatology',
                        'units': 'K' if self.variable == 'temperature' else 'mm/day'
                    }
                },
                'projection_data': {
                    'dims': ['year'],
                    'data': self.projection_data.values,
                    'attrs': {
                        'long_name': f'Projection {self.variable} data',
                        'units': 'K' if self.variable == 'temperature' else 'mm/day'
                    }
                },
                'raw_anomalies': {
                    'dims': ['year'],
                    'data': raw_anomalies.values,
                    'attrs': {
                        'long_name': f'{self.variable} anomalies',
                        'units': 'K' if self.variable == 'temperature' else 'mm/day'
                    }
                }
            },
            'coords': {
                'year': {
                    'dims': ['year'],
                    'data': self.projection_data.year.values,
                    'attrs': {
                        'long_name': 'Year',
                        'units': 'year'
                    }
                }
            },
            'attrs': {
                'description': (
                    f'Climate anomalies for {self.variable}, {self.experiment}, '
                    f'month {self.month}, model {self.model}'
                ),
                'historical_period': '1995-2014',
                'projection_period': '2015-2100',
                'model': self.model  # Add model to attributes
            }
        }

        logger.info("Anomaly computation complete")

        return results
