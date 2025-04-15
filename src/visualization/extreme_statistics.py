"""
Extreme Statistics Module

This module provides functions to analyze climate extreme events and create visualizations.
"""

import logging
from typing import Dict, Any, List
import calendar

import matplotlib.pyplot as plt
import numpy as np
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

        # Additional attributes for trend analysis
        self.mk_results = {}
        self.trend_statistics = {}

    def load_data(self, data_type: str = 'extremes', months: List[int] = None) -> None:
        """
        Load extreme events data from NetCDF file.

        Args:
            data_type: Type of data to load (defaults to 'extremes')
            months: List of months to load (for multi-month analysis)
        """
        super().load_data(data_type=data_type, months=months)

    def calculate_statistics(self, metric: str = None) -> Dict[str, Any]:
        """
        Calculate trend statistics for specified metric.

        Args:
            metric: Specific metric to analyze (if None, uses self.metric)

        Returns:
            Dictionary with Mann-Kendall test results
        """
        # Use provided metric or default to instance metric
        metric = metric or self.metric

        if not metric:
            raise ValueError("No metric specified for trend calculation")

        if not self.is_data_loaded:
            logger.info("Data not loaded. Loading extremes data...")
            self.load_data(data_type='extremes')

        logger.info(f"Calculating trend statistics for {metric}")

        # Get years data
        years = self.data.year.values

        # Extract data values for metric
        if metric in self.data:
            data_values = self.data[metric].values
        else:
            raise ValueError(f"Metric '{metric}' not found in loaded data")

        # Handle timedelta data type if present
        try:
            # If it's a timedelta, convert to float (days)
            if hasattr(data_values, 'dtype') and np.issubdtype(data_values.dtype, np.timedelta64):
                data_values = data_values.astype('timedelta64[D]').astype(float)
        except (TypeError, ValueError):
            # If conversion fails, just proceed with original values
            pass

        # Skip if all NaN
        if np.all(np.isnan(data_values)):
            logger.warning(f"All NaN values for {metric}, skipping")
            return {}

        # Remove NaN values for Mann-Kendall
        valid_idx = ~np.isnan(data_values)
        valid_data = data_values[valid_idx]
        valid_years = years[valid_idx]

        # Ensure numeric data
        try:
            valid_data = valid_data.astype(float)
        except (ValueError, TypeError):
            logger.warning(f"Cannot convert {metric} data to numeric, skipping")
            return {}

        if len(valid_data) < 10:
            logger.warning(f"Not enough valid data points for {metric}, skipping")
            return {}

        # Run Mann-Kendall test
        try:
            # Use original values for Mann-Kendall test
            mk_result = mk.original_test(valid_data)

            # Determine if this is a duration metric
            is_duration_metric = 'duration' in metric or metric in ['persistence', 'cdd', 'cwd']

            if is_duration_metric:
                # For duration metrics, calculate slope manually
                num_valid = len(valid_data)
                slopes = []
                for i in range(num_valid):
                    for j in range(i + 1, num_valid):
                        # Calculate slope as change in days per year
                        time_diff = valid_years[j] - valid_years[i]
                        if time_diff != 0:  # Avoid division by zero
                            slope = (valid_data[j] - valid_data[i]) / time_diff
                            slopes.append(slope)

                # Use median of slopes (Sen's slope estimator)
                corrected_slope = np.median(slopes) if slopes else 0.0

                # Store results
                self.mk_results[metric] = {
                    'trend': mk_result.trend,
                    'h': mk_result.h,
                    'p': mk_result.p,
                    'z': mk_result.z,
                    'tau': mk_result.Tau,
                    's': mk_result.s,
                    'var_s': mk_result.var_s,
                    'slope': corrected_slope,
                    'intercept': 0  # Not used for plotting
                }

                # Format for display
                self.trend_statistics[metric] = {
                    'tau': f"{mk_result.Tau:.3f}",
                    'p_value': (f"{mk_result.p:.3f}" if mk_result.p >= 0.001
                                else f"{mk_result.p:.2e}"),
                    'slope': f"{corrected_slope:.5f}",
                    'significant': mk_result.h,
                    'trend_text': mk_result.trend
                }

                logger.info(f"Mann-Kendall for {metric}: {mk_result.trend}, p={mk_result.p:.4f}, "
                            f"corrected_slope={corrected_slope:.5f} days/year")
            else:
                # For non-duration metrics, use original calculation
                self.mk_results[metric] = {
                    'trend': mk_result.trend,
                    'h': mk_result.h,
                    'p': mk_result.p,
                    'z': mk_result.z,
                    'tau': mk_result.Tau,
                    's': mk_result.s,
                    'var_s': mk_result.var_s,
                    'slope': mk_result.slope,
                    'intercept': mk_result.intercept
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

    def plot_metric_time_series(self, metric: str = None, y_min: float = None, y_max: float = None) -> plt.Figure:
        """
        Create a time series plot for any climate metric.

        Args:
            metric: Specific metric to plot (if None, uses self.metric)
            y_min: Optional minimum value for y-axis
            y_max: Optional maximum value for y-axis

        Returns:
            Matplotlib figure object
        """
        # Use provided metric or default to instance metric
        metric = metric or self.metric

        if not metric:
            raise ValueError("No metric specified for plotting")

        if not self.is_data_loaded:
            logger.info("Data not loaded. Loading extremes data...")
            self.load_data(data_type='extremes')

        # Calculate statistics if not already done
        if not self.trend_statistics or metric not in self.trend_statistics:
            self.calculate_statistics(metric=metric)

        logger.info(f"Creating time series plot for {metric}")

        # Setup figure
        self.setup_figure(figsize=(12, 8))

        # Extract data
        years = self.data.year.values
        metric_data = self.data[metric].values

        # Get metric metadata
        metric_long_name = self.data[metric].attrs.get('long_name', metric)
        metric_units = self.data[metric].attrs.get('units', '')

        # Get color based on metric category
        category = self.get_metric_category(metric)
        color = self.color_mappings[category][self.experiment]

        # Handle timedelta data type if present
        try:
            # If it's a timedelta, convert to float (days)
            if hasattr(metric_data, 'dtype') and np.issubdtype(metric_data.dtype, np.timedelta64):
                metric_data = metric_data.astype('timedelta64[D]').astype(float)
        except (TypeError, ValueError):
            # If conversion fails, just proceed with original values
            pass

        # Plot the data points
        ax = self.axes[0]
        ax.plot(years, metric_data, marker='o', markersize=7,
                color=color, linewidth=1.5,
                markerfacecolor=color, markeredgecolor='white',
                markeredgewidth=0.5, zorder=5)

        # Add trend line using simple linear regression
        valid_idx = ~np.isnan(metric_data)
        if np.sum(valid_idx) > 2:  # Need at least 3 points for a trend
            valid_years = years[valid_idx].astype(float)  # Ensure years are floats
            valid_values = metric_data[valid_idx].astype(float)  # Ensure values are floats

            # Simple linear regression
            slope, intercept = np.polyfit(valid_years, valid_values, 1)

            # Plot trend line
            trend_years = np.array([years[0], years[-1]]).astype(float)
            trend_values = slope * trend_years + intercept
            ax.plot(trend_years, trend_values, color='black', linestyle='--',
                    linewidth=2, label='Trend', zorder=4)

            # Explicitly add legend
            ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=12,
                      bbox_to_anchor=(0.98, 0.98), borderpad=0.5)

        # Add statistics text box if available
        if metric in self.trend_statistics:
            stats = self.trend_statistics[metric]
            stats_text = (f"Mann-Kendall Statistics:\n"
                          f"Kendall's τ: {stats['tau']}\n"
                          f"Slope: {stats['slope']} {metric_units}/year\n"
                          f"p-value: {stats['p_value']}\n"
                          f"Trend: {stats['trend_text']}")

            # Place text box in top left with background
            ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
                    fontsize=12, verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=1.0, pad=0.5),
                    zorder=100)

        # Set y-axis limits if provided
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)

        # Add labels and title
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(f"{metric_long_name} ({metric_units})", fontsize=12)

        # Format y-axis based on units
        if 'days' in metric_units.lower() or metric_units.lower() == 'days':
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Add overall title
        title = f"{metric_long_name} - {self.experiment.upper()}"
        subtitle = f"Month: {self._get_month_name(self.month)}, Location: {self.latitude:.2f}°, {self.longitude:.2f}°"
        self.add_title(title, subtitle)

        # Add description at the bottom
        metadata_text = (f"Variable: {self.variable}, Experiment: {self.experiment}, "
                         f"Month: {self.month}, Metric: {metric}, Type: Time Series")
        self.fig.text(0.5, 0.04, metadata_text, ha='center', fontsize=10, style='italic')

        # Adjust layout to accommodate text at bottom
        self.fig.tight_layout(rect=[0, 0.05, 1, 0.95])  # Increase bottom margin

        return self.fig

    def create_metric_heatmap(self, metric: str = None, months: List[int] = None) -> plt.Figure:
        """
        Create a heatmap of a metric by month and year.

        Args:
            metric: Specific metric to visualize (if None, uses self.metric)
            months: List of months to include (if None, uses all months)

        Returns:
            Matplotlib figure object
        """
        # Use provided metric or default to instance metric
        metric = metric or self.metric

        if not metric:
            raise ValueError("No metric specified for heatmap")

        # If months is None, use all months
        if months is None:
            months = list(range(1, 13))

        logger.info(f"Creating monthly heatmap for {metric} across months {months}")

        # Load all months data if not provided
        all_months_data = {}

        # Backup current month
        original_month = self.month

        for month in months:
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

        # Fill with metric data
        for i, month in enumerate(months):
            if month in all_months_data:
                ds = all_months_data[month]
                if metric in ds:
                    # Extract metric data
                    metric_data = ds[metric].values

                    # Handle timedelta data type if present
                    try:
                        # If it's a timedelta, convert to float (days)
                        if hasattr(metric_data, 'dtype') and np.issubdtype(metric_data.dtype, np.timedelta64):
                            metric_data = metric_data.astype('timedelta64[D]').astype(float)
                    except (TypeError, ValueError):
                        # If conversion fails, just proceed with original values
                        pass

                    heatmap_data[i, :] = metric_data
                else:
                    logger.warning(f"Metric {metric} not found in data for month {month}")

        # Create figure
        self.setup_figure(figsize=(14, 8))
        ax = self.axes[0]

        # Get metric metadata from first available dataset
        sample_ds = all_months_data[first_month]
        if metric in sample_ds:
            metric_long_name = sample_ds[metric].attrs.get('long_name', metric)
            metric_units = sample_ds[metric].attrs.get('units', '')
        else:
            metric_long_name = metric
            metric_units = ''

        # Create heatmap
        cmap = self.get_colormap_for_heatmap()
        im = ax.imshow(heatmap_data, aspect='auto', cmap=cmap,
                       interpolation='nearest', origin='lower')

        # Define x-axis (years) and y-axis (months) ticks
        # Show every 5 years on x-axis
        x_tick_indices = np.arange(0, len(years), 5)
        ax.set_xticks(x_tick_indices)
        ax.set_xticklabels([str(years[i]) for i in x_tick_indices], rotation=45)

        # Month names on y-axis
        month_names = [calendar.month_abbr[m] for m in range(1, 13)]

        # Use only available months for y labels
        y_tick_indices = np.arange(len(months))
        ax.set_yticks(y_tick_indices)
        ax.set_yticklabels([month_names[m - 1] for m in months])

        # Add colorbar
        cbar = self.fig.colorbar(im, ax=ax)
        cbar.set_label(f'{metric_long_name} ({metric_units})')

        # Add annotations for significant trends
        for i, month in enumerate(months):
            if month in all_months_data:
                ds = all_months_data[month]

                if metric in ds:
                    # Temporarily set month and data for Mann-Kendall calculation
                    original_month = self.month
                    self.month = month
                    self.data = ds
                    stats = self.calculate_statistics(metric)

                    # Check if metric has significant trend
                    if metric in stats and stats[metric]['significant']:
                        # Add a marker at the bottom of the heatmap cell
                        x_center = len(years) - 1  # Last year
                        y_center = i
                        ax.text(x_center, y_center, '*', color='black',
                                fontsize=14, ha='center', va='center')

                    # Reset month
                    self.month = original_month

        # Add title
        title = f"{metric_long_name} by Month - {self.experiment.upper()}"
        subtitle = f"Location: {self.latitude:.2f}°, {self.longitude:.2f}°"
        self.add_title(title, subtitle)

        # Add asterisk explanation
        asterisk_text = "* Significant trend (p<0.05, Mann-Kendall test)"
        self.fig.text(0.98, 0.02, asterisk_text, ha='right', fontsize=9,
                      style='italic')

        # Adjust layout
        plt.subplots_adjust(top=0.90, bottom=0.1)

        return self.fig
