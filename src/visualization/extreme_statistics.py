"""
Extreme Statistics Module

This module provides functions to analyze climate extreme events and create visualizations.
"""

import logging
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pymannkendall as mk

from .plotter import BasePlotter

# Configure logger
logger = logging.getLogger(__name__)


class ExtremeStatisticsPlotter(BasePlotter):
    """
    Class for analyzing and visualizing climate extreme events.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with parent class constructor."""
        super().__init__(*args, **kwargs)

        # Additional attributes for heat wave analysis
        self.mk_results = {}
        self.trend_statistics = {}

    def load_data(self, data_type: str = 'extremes') -> None:
        """
        Load extreme events data from NetCDF file.

        Args:
            data_type: Type of data to load (defaults to 'extremes')
        """
        super().load_data(data_type=data_type)

    def calculate_statistics(self) -> Dict[str, Any]:
        """
        Calculate trend statistics for extreme event metrics.

        Returns:
            Dictionary with Mann-Kendall test results
        """
        if not self.is_data_loaded:
            logger.info("Data not loaded. Loading extremes data...")
            self.load_data(data_type='extremes')

        logger.info("Calculating trend statistics for extreme event metrics")

        metrics = []
        # Check which metrics are available in the dataset
        if 'frequency' in self.data:
            metrics.append('frequency')
        if 'persistence' in self.data:
            metrics.append('persistence')
        if 'intensity' in self.data:
            metrics.append('intensity')

        # Heat wave specific metrics
        if 'hw_count' in self.data:
            metrics.append('hw_count')
        if 'hw_days' in self.data:
            metrics.append('hw_days')
        if 'hw_max_duration' in self.data:
            metrics.append('hw_max_duration')
        if 'hw_mean_duration' in self.data:
            metrics.append('hw_mean_duration')

        # Calculate Mann-Kendall and Sen's Slope for each metric
        for metric in metrics:
            data_values = self.data[metric].values

            # Skip if all NaN
            if np.all(np.isnan(data_values)):
                logger.warning(f"All NaN values for {metric}, skipping")
                continue

            # Remove NaN values for Mann-Kendall
            valid_idx = ~np.isnan(data_values)
            valid_data = data_values[valid_idx]

            if len(valid_data) < 10:
                logger.warning(f"Not enough valid data points for {metric}, skipping")
                continue

            # Run Mann-Kendall test
            try:
                mk_result = mk.original_test(valid_data)

                # Store results
                self.mk_results[metric] = {
                    'trend': mk_result.trend,
                    'h': mk_result.h,  # True if trend is significant
                    'p': mk_result.p,  # p-value
                    'z': mk_result.z,  # test statistic
                    'tau': mk_result.Tau,  # Kendall's Tau
                    's': mk_result.s,  # Mann-Kendall's S statistic
                    'var_s': mk_result.var_s,  # Variance of S
                    'slope': mk_result.slope,  # Sen's slope
                    'intercept': mk_result.intercept  # Sen's intercept
                }

                # Format for display
                self.trend_statistics[metric] = {
                    'tau': f"{mk_result.Tau:.3f}",
                    'p_value': (f"{mk_result.p:.3f}" if mk_result.p >= 0.001
                                else f"{mk_result.p:.2e}"),
                    'slope': f"{mk_result.slope:.5f}",
                    'significant': mk_result.h,
                    'trend_text': mk_result.trend
                }

                logger.info(f"Mann-Kendall for {metric}: {mk_result.trend}, "
                            f"p={mk_result.p:.4f}, slope={mk_result.slope:.5f}")

            except Exception as e:
                logger.error(f"Error calculating Mann-Kendall for {metric}: {e}")

        return self.trend_statistics

    def plot_heat_wave_metrics(self) -> plt.Figure:
        """
        Create a plot showing heat wave frequency and duration metrics.

        Returns:
            Matplotlib figure object
        """
        if not self.is_data_loaded:
            logger.info("Data not loaded. Loading extremes data...")
            self.load_data(data_type='extremes')

        # Calculate statistics if not already done
        if not self.trend_statistics:
            self.calculate_statistics()

        logger.info("Creating heat wave metrics plot")

        # Setup 2-panel figure
        self.setup_figure(nrows=2, ncols=1, figsize=(12, 10))

        # Extract data
        years = self.data.year.values

        # Plot heat wave count in top panel
        ax_top = self.axes[0]
        hw_count = self.data.hw_count.values

        # Get colors for this variable
        colors = self.color_mappings[self.variable]

        # Plot heat wave count
        ax_top.plot(years, hw_count, marker='o', markersize=4,
                    color=colors[self.experiment], linewidth=1.5)

        # Add trend line if available
        if 'hw_count' in self.mk_results:
            mk_res = self.mk_results['hw_count']
            x = np.array(years)
            y = mk_res['slope'] * x + mk_res['intercept']
            ax_top.plot(x, y, color='black', linestyle='--',
                        linewidth=2, label='Trend')

        # Add statistics text box if available
        if 'hw_count' in self.trend_statistics:
            stats = self.trend_statistics['hw_count']
            stats_text = (f"Mann-Kendall Statistics:\n"
                          f"Kendall's τ: {stats['tau']}\n"
                          f"Slope: {stats['slope']} events/year\n"
                          f"p-value: {stats['p_value']}\n"
                          f"Trend: {stats['trend_text']}")

            # Place text box in top left with background
            ax_top.text(0.03, 0.97, stats_text, transform=ax_top.transAxes,
                        fontsize=10, verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Add labels and title
        ax_top.set_ylabel('Heat Wave Events (per month)', fontsize=12)
        ax_top.set_title(f'Heat Wave Frequency', fontsize=14)

        # Plot maximum heat wave duration in bottom panel
        ax_bottom = self.axes[1]
        hw_max_duration = self.data.hw_max_duration.values

        # Plot heat wave maximum duration
        ax_bottom.plot(years, hw_max_duration, marker='o', markersize=4,
                       color=colors[self.experiment], linewidth=1.5)

        # Add trend line if available
        if 'hw_max_duration' in self.mk_results:
            mk_res = self.mk_results['hw_max_duration']
            x = np.array(years)
            y = mk_res['slope'] * x + mk_res['intercept']
            ax_bottom.plot(x, y, color='black', linestyle='--',
                           linewidth=2, label='Trend')

        # Add statistics text box if available
        if 'hw_max_duration' in self.trend_statistics:
            stats = self.trend_statistics['hw_max_duration']
            stats_text = (f"Mann-Kendall Statistics:\n"
                          f"Kendall's τ: {stats['tau']}\n"
                          f"Slope: {stats['slope']} days/year\n"
                          f"p-value: {stats['p_value']}\n"
                          f"Trend: {stats['trend_text']}")

            # Place text box in top left with background
            ax_bottom.text(0.03, 0.97, stats_text, transform=ax_bottom.transAxes,
                           fontsize=10, verticalalignment='top', horizontalalignment='left',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Add labels
        ax_bottom.set_xlabel('Year', fontsize=12)
        ax_bottom.set_ylabel('Maximum Duration (days)', fontsize=12)
        ax_bottom.set_title(f'Heat Wave Maximum Duration', fontsize=14)

        # Add threshold information from global attributes
        threshold_value = self.data.attrs.get('threshold_value', 'N/A')
        threshold_percentile = self.data.attrs.get('threshold_percentile', 'N/A')
        min_duration = self.data.attrs.get('heat_wave_min_duration', 'N/A')

        # Add threshold info text at the bottom
        threshold_text = (f"Heat wave definition: ≥{min_duration} consecutive days exceeding "
                          f"the {threshold_percentile}th percentile ({threshold_value:.2f} K)")

        self.fig.text(0.5, 0.01, threshold_text, ha='center', fontsize=10)

        # Add overall title
        title = f"Heat Wave Metrics - {self.experiment.upper()}"
        subtitle = f"Month: {self._get_month_name(self.month)}, Location: {self.latitude:.2f}°, {self.longitude:.2f}°"
        self.add_title(title, subtitle)

        # Adjust layout
        self.fig.tight_layout(rect=[0, 0.02, 1, 0.95])

        return self.fig

    def create_monthly_heatmap(self, all_months_data: Dict[int, xr.Dataset] = None) -> plt.Figure:
        """
        Create a heatmap of heat wave days by month and year.

        Args:
            all_months_data: Dictionary of datasets for all months (if provided)
                             Keys are month numbers, values are xarray Datasets

        Returns:
            Matplotlib figure object
        """
        logger.info("Creating monthly heat wave heatmap")

        # Load all months data if not provided
        if all_months_data is None:
            logger.info("Loading data for all months")
            all_months_data = {}

            # Backup current month
            original_month = self.month

            for month in range(1, 13):
                try:
                    # Temporarily set month
                    self.month = month

                    # Load data for this month
                    self.load_data(data_type='extremes')

                    # Store data
                    all_months_data[month] = self.data

                    # Reset data loaded flag for next iteration
                    self.is_data_loaded = False

                    logger.info(f"Loaded data for month {month}")
                except FileNotFoundError:
                    logger.warning(f"No data found for month {month}")
                finally:
                    # Restore original month
                    self.month = original_month

        # Check if we have data
        if not all_months_data:
            raise ValueError("No data available for heatmap")

        # Create a heatmap data array
        # Get the range of years from the first available dataset
        first_month = min(all_months_data.keys())
        years = all_months_data[first_month].year.values
        months = sorted(all_months_data.keys())

        # Create empty data array for heatmap
        heatmap_data = np.full((len(months), len(years)), np.nan)

        # Fill with heat wave days
        for i, month in enumerate(months):
            if month in all_months_data:
                ds = all_months_data[month]
                # Use hw_days for heat wave days if available, otherwise use frequency
                if 'hw_days' in ds:
                    heatmap_data[i, :] = ds.hw_days.values
                elif 'frequency' in ds:
                    heatmap_data[i, :] = ds.frequency.values

        # Create figure
        self.setup_figure(figsize=(14, 8))
        ax = self.axes[0]

        # Create heatmap
        cmap = 'YlOrRd'  # Yellow-Orange-Red colormap, good for heat waves
        im = ax.imshow(heatmap_data, aspect='auto', cmap=cmap,
                       interpolation='nearest', origin='lower')

        # Define x-axis (years) and y-axis (months) ticks
        # Show every 5 years on x-axis
        x_tick_indices = np.arange(0, len(years), 5)
        ax.set_xticks(x_tick_indices)
        ax.set_xticklabels([str(years[i]) for i in x_tick_indices], rotation=45)

        # Month names on y-axis
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # Use only available months for y labels
        y_tick_indices = np.arange(len(months))
        ax.set_yticks(y_tick_indices)
        ax.set_yticklabels([month_names[m - 1] for m in months])

        # Add colorbar
        cbar = self.fig.colorbar(im, ax=ax)
        cbar.set_label('Heat Wave Days per Month')

        # Add annotations for significant trends
        for i, month in enumerate(months):
            if month in all_months_data:
                ds = all_months_data[month]

                # Temporarily set month for Mann-Kendall calculation
                original_month = self.month
                self.month = month
                self.data = ds
                self.calculate_statistics()

                # Check if hw_days has significant trend
                metric = 'hw_days' if 'hw_days' in self.trend_statistics else 'frequency'
                if metric in self.trend_statistics and self.trend_statistics[metric]['significant']:
                    # Add a marker at the bottom of the heatmap cell
                    x_center = len(years) - 1  # Last year
                    y_center = i
                    ax.text(x_center, y_center, '*', color='black',
                            fontsize=14, ha='center', va='center')

                # Reset month
                self.month = original_month

        # Get threshold information from any of the datasets
        sample_ds = next(iter(all_months_data.values()))
        threshold_value = sample_ds.attrs.get('threshold_value', 'N/A')
        threshold_percentile = sample_ds.attrs.get('threshold_percentile', 'N/A')
        min_duration = sample_ds.attrs.get('heat_wave_min_duration', 'N/A')

        # Add title and threshold information
        title = f"Heat Wave Days by Month - {self.experiment.upper()}"
        subtitle = f"Location: {self.latitude:.2f}°, {self.longitude:.2f}°"
        self.add_title(title, subtitle)

        threshold_text = (f"Heat wave definition: ≥{min_duration} consecutive days exceeding "
                          f"the {threshold_percentile}th percentile ({threshold_value:.2f} K)")
        self.fig.text(0.5, 0.01, threshold_text, ha='center', fontsize=10)

        # Add asterisk explanation
        asterisk_text = "* Significant trend (p<0.05, Mann-Kendall test)"
        self.fig.text(0.98, 0.02, asterisk_text, ha='right', fontsize=9,
                      style='italic')

        # Adjust layout - fix for colorbar compatibility
        # self.fig.tight_layout()
        # Adjust the padding manually
        plt.subplots_adjust(top=0.90, bottom=0.05)

        return self.fig
