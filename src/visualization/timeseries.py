"""
Time Series Visualization Module

This module provides classes for visualizing time series data from CMIP6 climate data.
"""

import logging
import matplotlib.pyplot as plt

from .plotter import BasePlotter

# Configure logger
logger = logging.getLogger(__name__)


class TimeSeriesPlotter(BasePlotter):
    """
    Class for creating time series plots from CMIP6 climate data.
    """

    def plot_anomaly_decomposition(self) -> plt.Figure:
        """
        Create a multi-scale anomaly decomposition plot.

        Returns:
            Matplotlib figure object
        """
        if not self.is_data_loaded:
            logger.info("Data not loaded. Loading anomalies data...")
            self.load_data(data_type='anomalies')

        logger.info("Creating multi-scale anomaly decomposition plot")

        # Setup 2-panel figure
        self.setup_figure(nrows=2, ncols=1, figsize=(12, 10))

        # Extract data
        years = self.data.year.values
        raw_anomalies = self.data.standardized_raw_anomalies.values
        interannual = self.data.standardized_interannual_variability.values
        decadal = self.data.standardized_decadal_variability.values
        long_term = self.data.long_term_trend.values

        # Years for bottom panel
        years_std = self.data.year_std.values
        variability_trend = self.data.variability_trend.values

        # Get colors for this variable
        colors = self.color_mappings[self.variable]

        # Plot top panel: Multi-scale decomposition
        ax_top = self.axes[0]

        # Plot interannual variability as bars
        ax_top.bar(years, interannual, color=colors['interannual_variability'],
                   alpha=0.7, label='Interannual Variability')

        # Plot standardized anomalies as line
        ax_top.plot(years, raw_anomalies, color=colors['raw_anomalies'],
                    linewidth=1.5, label='Raw Anomalies')

        # Plot decadal variability as line
        ax_top.plot(years, decadal, color=colors['decadal_variability'],
                    linewidth=2.5, label='Decadal Variability')

        # Plot long-term trend as line
        ax_top.plot(years, long_term, color=colors['long_term_trend'],
                    linewidth=3, label='Long-term Trend')

        # Add horizontal line at y=0
        ax_top.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

        # Add labels and legend
        if self.variable == 'temperature':
            ax_top.set_ylabel('Temperature Anomaly (°C)', fontsize=12)
        else:
            ax_top.set_ylabel('Standardized Anomaly', fontsize=12)
        ax_top.set_title(f'Multi-scale Decomposition of {self.variable.capitalize()} Anomalies',
                         fontsize=14)

        # Extract statistics from attributes
        slope = self.data.long_term_trend.attrs.get('slope', 'N/A')
        r_value = self.data.long_term_trend.attrs.get('r_value', 'N/A')
        p_value = self.data.long_term_trend.attrs.get('p_value', 'N/A')

        # Format p-value for scientific notation if very small
        if isinstance(p_value, (int, float)) and p_value < 0.001:
            p_value_str = f"{p_value:.2e}"
        else:
            p_value_str = f"{p_value}"

        # Add statistics text box
        stats_text = (f"Trend Statistics:\n"
                      f"Slope: {slope:.5f} per year\n"
                      f"R²: {r_value ** 2:.3f}\n"
                      f"p-value: {p_value_str}")

        # Place text box in top right
        ax_top.text(0.97, 0.97, stats_text, transform=ax_top.transAxes,
                    fontsize=12, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        # Add legend
        self.add_legend(ax=ax_top, loc='upper left')

        # Plot bottom panel: Variability trend
        ax_bottom = self.axes[1]

        # Plot variability trend
        ax_bottom.plot(years_std, variability_trend, color=colors['variability_trend'],
                       linewidth=2.5)

        # Add horizontal line at y=0
        ax_bottom.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

        # Get historical reference value
        hist_ref = self.data.variability_trend.attrs.get('historical_reference', 'N/A')
        window_size = self.data.attrs.get('window_size', 'N/A')

        # Add box around both plots with solid black lines
        for ax in self.axes:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1.0)

            # Make sure grid is behind data
            ax.set_axisbelow(True)

        # Add labels and title
        ax_bottom.set_xlabel('Year', fontsize=12)
        if self.variable == 'temperature':
            ax_bottom.set_ylabel('Temperature Variability (°C)', fontsize=12)
        else:
            ax_bottom.set_ylabel('Standardized Variability', fontsize=12)
        ax_bottom.set_title(f'Trend of Interannual Variability (Window Size: {window_size} Years)',
                            fontsize=14)

        # Add reference value text
        ref_text = f"Historical Reference: {hist_ref:.5f}"
        ax_bottom.text(0.03, 0.97, ref_text, transform=ax_bottom.transAxes,
                       fontsize=12, verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Add overall title
        title = f"{self.variable.capitalize()} Variability Analysis - {self.experiment.upper()}"

        # Convert coordinates to cardinal format
        lat_dir = "N" if self.latitude >= 0 else "S"
        lat_val = abs(self.latitude)

        # Handle longitude conversion (values >180 should be converted to negative/western values)
        adj_lon = self.longitude if self.longitude <= 180 else self.longitude - 360
        lon_dir = "E" if adj_lon >= 0 else "W"
        lon_val = abs(adj_lon)

        subtitle = (f"Month: {self._get_month_name(self.month)}, "
                    f"Location: {lat_val:.2f}°{lat_dir}, {lon_val:.2f}°{lon_dir}")
        self.add_title(title, subtitle)

        # Adjust layout
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])

        return self.fig
