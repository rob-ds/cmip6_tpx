"""
Base Plotter Module

This module provides a base class for creating visualizations from CMIP6 climate data.
"""

import logging
from pathlib import Path
from typing import Optional, Union, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Configure logger
logger = logging.getLogger(__name__)


class BasePlotter:
    """
    Base class for creating plots from CMIP6 climate data.

    This class provides common functionality for loading data and setting up
    figures with consistent styling.
    """

    # Default color schemes for different metric types
    DEFAULT_COLOR_SCHEMES = {
        'temperature': {
            'single': {
                'ssp245': '#F4A582',  # Light red
                'ssp585': '#B2182B'  # Dark red
            },
            'colormap': 'YlOrRd'
        },
        'precipitation': {
            'single': {
                'ssp245': '#92C5DE',  # Light blue
                'ssp585': '#2166AC'  # Dark blue
            },
            'colormap': 'Blues'
        },
        'compound': {
            'single': {
                'ssp245': '#D1B3DF',  # Light purple
                'ssp585': '#762A83'  # Dark purple
            },
            'colormap': 'Purples'
        }
    }

    # Metric categorization
    METRIC_CATEGORIES = {
        # Temperature metrics
        'tm_max': 'temperature',
        'tm_min': 'temperature',
        'tm90p': 'temperature',
        'tm10p': 'temperature',
        'warm_spell_days': 'temperature',
        'cold_spell_days': 'temperature',

        # Precipitation metrics
        'r95p': 'precipitation',
        'prcptot': 'precipitation',
        'rx1day': 'precipitation',
        'rx5day': 'precipitation',
        'r10mm': 'precipitation',
        'r20mm': 'precipitation',
        'sdii': 'precipitation',
        'cdd': 'precipitation',
        'cwd': 'precipitation',

        # Compound metrics
        'hot_dry_frequency': 'compound',
        'hot_wet_frequency': 'compound',
        'cold_wet_frequency': 'compound',
        'cwhd': 'compound',
        'wspi': 'compound',
        'wpd': 'compound',

        # Legacy metrics (kept for backward compatibility)
        'frequency': 'temperature',
        'persistence': 'temperature',
        'intensity': 'temperature',
        'hw_count': 'temperature',
        'hw_days': 'temperature',
        'hw_max_duration': 'temperature',
        'hw_mean_duration': 'temperature',
    }

    def __init__(
            self,
            variable: str,
            experiment: str,
            month: int,
            input_dir: Union[str, Path],
            output_dir: Union[str, Path],
            metric: Optional[str] = None,
            color_scheme: Optional[str] = None,
            dpi: int = 300,
            figsize: Tuple[float, float] = (10, 8)
    ):
        """
        Initialize the plotter.

        Args:
            variable: Climate variable to analyze ('temperature' or 'precipitation')
            experiment: Experiment to analyze ('ssp245' or 'ssp585')
            month: Month to analyze (1-12)
            input_dir: Directory containing input data
            output_dir: Directory to store output figures
            metric: Specific metric to analyze (if None, uses a default based on variable)
            color_scheme: Color scheme to use (if None, uses default based on metric)
            dpi: Resolution for saved figures
            figsize: Default figure size (width, height) in inches
        """
        self.variable = variable
        self.experiment = experiment
        self.month = month
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.metric = metric
        self.color_scheme = color_scheme
        self.dpi = dpi
        self.figsize = figsize

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data container
        self.data = None
        self.is_data_loaded = False
        self.latitude = None
        self.longitude = None
        self.metric_category = None
        self.metric_long_name = None
        self.metric_units = None

        # Figure and axes containers
        self.fig = None
        self.axes = None

        # Set up color mappings
        self.color_mappings = {
            'temperature': {
                'raw_anomalies': 'red',
                'interannual_variability': 'blue',
                'decadal_variability': 'green',
                'long_term_trend': 'purple',
                'variability_trend': 'black',
                'ssp245': '#F4A582',  # Light red
                'ssp585': '#B2182B'  # Dark red
            },
            'precipitation': {
                'raw_anomalies': 'red',
                'interannual_variability': 'blue',
                'decadal_variability': 'green',
                'long_term_trend': 'purple',
                'variability_trend': 'black',
                'ssp245': '#92C5DE',  # Light blue
                'ssp585': '#2166AC'  # Dark blue
            },
            'compound': {
                'raw_anomalies': 'red',
                'interannual_variability': 'blue',
                'decadal_variability': 'green',
                'long_term_trend': 'purple',
                'variability_trend': 'black',
                'ssp245': '#D1B3DF',  # Light purple
                'ssp585': '#762A83'  # Dark purple
            }
        }

        # If color scheme is provided, override the defaults
        if color_scheme:
            if color_scheme.lower() == 'reds':
                self.color_mappings = {k: self.color_mappings['temperature'] for k in self.color_mappings}
            elif color_scheme.lower() == 'blues':
                self.color_mappings = {k: self.color_mappings['precipitation'] for k in self.color_mappings}
            elif color_scheme.lower() == 'purples':
                self.color_mappings = {k: self.color_mappings['compound'] for k in self.color_mappings}

        logger.info(
            f"Initialized {self.__class__.__name__} with {variable}, {experiment}, month {month}, metric {metric}")

    def get_metric_category(self, metric: str = None) -> str:
        """
        Determine the category of a metric.

        Args:
            metric: The metric to categorize (if None, uses self.metric)

        Returns:
            Category of the metric ('temperature', 'precipitation', or 'compound')
        """
        metric = metric or self.metric
        return self.METRIC_CATEGORIES.get(metric, 'unknown')

    def get_colormap_for_heatmap(self) -> str:
        """
        Get the colormap to use for heatmap visualization based on metric category.

        Returns:
            Name of matplotlib colormap
        """
        if not self.metric:
            # Default to temperature colormap if no metric specified
            return self.DEFAULT_COLOR_SCHEMES['temperature']['colormap']

        category = self.get_metric_category()

        # If custom color scheme is specified
        if self.color_scheme:
            if self.color_scheme.lower() == 'reds':
                return 'YlOrRd'
            elif self.color_scheme.lower() == 'blues':
                return 'Blues'
            elif self.color_scheme.lower() == 'purples':
                return 'Purples'

        # Otherwise use default for the category
        return self.DEFAULT_COLOR_SCHEMES.get(category, {}).get('colormap', 'viridis')

    def load_data(self, data_type: str = 'anomalies', months: List[int] = None) -> None:
        """
        Load data from NetCDF file.

        Args:
            data_type: Type of data to load ('anomalies' or 'extremes')
            months: List of months to load (for multi-month analysis)
        """
        # Default to current month if months not specified
        months = months or [self.month]

        # For anomalies, keep the original loading logic
        if data_type == 'anomalies':
            data_dir = self.input_dir / 'processed' / data_type
            file_name = f"{self.variable}_{self.experiment}_month{self.month:02d}_lat38p25_lon0p00.nc"
            file_path = data_dir / file_name

            logger.info(f"Loading anomalies data from {file_path}")

            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")

            self.data = xr.open_dataset(file_path, decode_timedelta=True)
            self.is_data_loaded = True

        # For extremes, use the new combined file format
        elif data_type == 'extremes':
            # Load data for all requested months
            monthly_datasets = {}

            for month in months:
                data_dir = self.input_dir / 'processed' / data_type
                file_name = f"extremes_{self.experiment}_month{month:02d}.nc"
                file_path = data_dir / file_name

                logger.info(f"Loading extremes data for month {month} from {file_path}")

                if not file_path.exists():
                    raise FileNotFoundError(f"Data file not found: {file_path}")

                ds = xr.open_dataset(file_path, decode_timedelta=True)
                monthly_datasets[month] = ds

            # If single month, store directly
            if len(months) == 1:
                self.data = monthly_datasets[months[0]]
                self.is_data_loaded = True
            # If multiple months, store as dictionary
            else:
                self.data = monthly_datasets
                self.is_data_loaded = True

        # Extract metadata if data is loaded and is a single dataset
        if self.is_data_loaded and not isinstance(self.data, dict):
            # Extract latitude and longitude from global attributes
            if hasattr(self.data, 'latitude') and hasattr(self.data, 'longitude'):
                self.latitude = float(self.data.latitude)
                self.longitude = float(self.data.longitude)
                logger.info(f"Location: {self.latitude:.2f}°, {self.longitude:.2f}°")

            # Extract metric information if a metric is specified
            if self.metric and self.metric in self.data:
                self.metric_category = self.get_metric_category()
                self.metric_long_name = self.data[self.metric].attrs.get('long_name', self.metric)
                self.metric_units = self.data[self.metric].attrs.get('units', '')
                logger.info(f"Metric: {self.metric} ({self.metric_long_name}), Units: {self.metric_units}")

    def setup_figure(self, nrows: int = 1, ncols: int = 1, figsize: Optional[Tuple[float, float]] = None) -> None:
        """
        Set up figure and axes with consistent styling.

        Args:
            nrows: Number of rows in the figure
            ncols: Number of columns in the figure
            figsize: Figure size (width, height) in inches
        """
        if figsize is None:
            figsize = self.figsize

        logger.info(f"Setting up figure with {nrows} rows and {ncols} columns")

        # Create figure and axes
        self.fig, self.axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, constrained_layout=True)

        # Set global figure style
        plt.style.use('seaborn-v0_8-whitegrid')

        # Apply consistent styling
        self.fig.patch.set_facecolor('white')

        # Convert self.axes to array for consistent indexing
        if nrows * ncols == 1:
            self.axes = np.array([self.axes])

        # Apply consistent styling to axes
        for ax in self.axes.flatten():
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    def add_title(self, title: str, subtitle: Optional[str] = None) -> None:
        """
        Add title and optional subtitle to the figure.

        Args:
            title: Main title
            subtitle: Optional subtitle
        """
        if subtitle:
            self.fig.suptitle(title, fontsize=16, weight='bold')
            self.fig.text(0.5, 0.91, subtitle, ha='center', fontsize=12, style='italic')
        else:
            self.fig.suptitle(title, fontsize=16, weight='bold')

    def add_legend(self, ax: Optional[plt.Axes] = None, **kwargs) -> None:
        """
        Add legend to the axis with consistent styling.

        Args:
            ax: Axis to add legend to (if None, uses the first axis)
            **kwargs: Additional arguments for legend
        """
        if ax is None:
            ax = self.axes.flatten()[0]

        # Default legend settings
        legend_settings = {
            'loc': 'best',
            'frameon': True,
            'facecolor': 'white',
            'edgecolor': 'gray',
            'framealpha': 0.9
        }

        # Update with user settings
        legend_settings.update(kwargs)

        # Add legend
        ax.legend(**legend_settings)

    def save_figure(self, filename: Optional[str] = None, formats: list = None) -> list:
        """
        Save figure to disk in specified formats.

        Args:
            filename: Base filename (without extension)
            formats: List of formats to save in

        Returns:
            List of saved file paths
        """
        if filename is None:
            if self.metric:
                filename = f"{self.variable}_{self.experiment}_{self.metric}_month{self.month:02d}"
            else:
                filename = f"{self.variable}_{self.experiment}_month{self.month:02d}"

        if formats is None:
            formats = ['png', 'pdf']

        saved_files = []

        for fmt in formats:
            output_path = self.output_dir / f"{filename}.{fmt}"
            logger.info(f"Saving figure to {output_path}")

            self.fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            saved_files.append(output_path)

        return saved_files

    def close(self) -> None:
        """Close figure and free memory."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = None

    @staticmethod
    def _get_month_name(month_number: int) -> str:
        """Convert month number to name."""
        return {
            1: 'January', 2: 'February', 3: 'March', 4: 'April',
            5: 'May', 6: 'June', 7: 'July', 8: 'August',
            9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }.get(month_number, f'Month {month_number}')
