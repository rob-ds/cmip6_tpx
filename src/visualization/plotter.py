"""
Base Plotter Module

This module provides a base class for creating visualizations from CMIP6 climate data.
"""

import logging
from pathlib import Path
from typing import Optional, Union, Tuple

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

    def __init__(
            self,
            variable: str,
            experiment: str,
            month: int,
            input_dir: Union[str, Path],
            output_dir: Union[str, Path],
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
            dpi: Resolution for saved figures
            figsize: Default figure size (width, height) in inches
        """
        self.variable = variable
        self.experiment = experiment
        self.month = month
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.figsize = figsize

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data container
        self.data = None
        self.is_data_loaded = False
        self.latitude = None
        self.longitude = None

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
                'ssp245': 'tab:orange',
                'ssp585': 'tab:red'
            },
            'precipitation': {
                'raw_anomalies': 'red',
                'interannual_variability': 'blue',
                'decadal_variability': 'green',
                'long_term_trend': 'purple',
                'variability_trend': 'black',
                'ssp245': 'tab:blue',
                'ssp585': 'tab:purple'
            }
        }

        logger.info(f"Initialized {self.__class__.__name__} with {variable}, {experiment}, month {month}")

    def load_data(self, data_type: str = 'anomalies') -> None:
        """
        Load data from NetCDF file.

        Args:
            data_type: Type of data to load ('anomalies' or 'extremes')
        """
        # Construct file path
        data_dir = self.input_dir / 'processed' / data_type
        file_name = f"{self.variable}_{self.experiment}_month{self.month:02d}_lat15p09_lon16p88.nc"
        file_path = data_dir / file_name

        logger.info(f"Loading data from {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # Load data
        self.data = xr.open_dataset(file_path)
        self.is_data_loaded = True

        logger.info(f"Data loaded: {list(self.data.data_vars)}")

        # Extract latitude and longitude from global attributes
        if hasattr(self.data, 'latitude') and hasattr(self.data, 'longitude'):
            self.latitude = float(self.data.latitude)
            self.longitude = float(self.data.longitude)
            logger.info(f"Location: {self.latitude:.2f}°, {self.longitude:.2f}°")

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
