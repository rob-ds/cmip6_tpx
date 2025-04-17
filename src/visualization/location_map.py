"""
Location Map Module

This module provides functions to visualize the geographical location of study points.
"""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from .plotter import BasePlotter

# Configure logger
logger = logging.getLogger(__name__)


class LocationMapPlotter(BasePlotter):
    """
    Class for creating location maps of study points.
    """

    def plot_location_map(self,
                          latitude: Optional[float] = None,
                          longitude: Optional[float] = None,
                          buffer: float = 5.0,
                          show_inset: bool = True) -> plt.Figure:
        """
        Create a map showing the location of the study point.

        Args:
            latitude: Latitude of study point (if None, uses the value from loaded data)
            longitude: Longitude of study point (if None, uses the value from loaded data)
            buffer: Buffer around the point in degrees for map extent
            show_inset: Whether to show an inset world map

        Returns:
            Matplotlib figure object
        """
        # Use provided coordinates or load from data
        if latitude is None or longitude is None:
            if not self.is_data_loaded:
                # Try to load data for the current month
                try:
                    self.load_data(data_type='anomalies')
                except FileNotFoundError:
                    try:
                        self.load_data(data_type='extremes')
                    except FileNotFoundError:
                        raise ValueError("No data loaded and no coordinates provided")

            latitude = self.latitude
            longitude = self.longitude

        logger.info(f"Creating location map for coordinates: {latitude}, {longitude}")

        # Setup figure with cartopy projection and adjusted layout
        fig = plt.figure(figsize=(10, 8))
        plt.subplots_adjust(left=-0.15, right=0.9)  # Adjust horizontal positioning

        # Create main map with Plate Carree projection
        ax_main = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # Set map extent based on point location with buffer
        ax_main.set_extent([longitude - buffer, longitude + buffer,
                            latitude - buffer, latitude + buffer], crs=ccrs.PlateCarree())

        # Add country borders and coastlines
        ax_main.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='gray')
        ax_main.add_feature(cfeature.COASTLINE, linewidth=1.5)

        # Add grid lines
        gl = ax_main.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False

        # Plot the location point with a marker
        ax_main.plot(longitude, latitude, 'o', markersize=10, color='red',
                     transform=ccrs.PlateCarree())

        # Add a circle around the point
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_radius = 0.25  # degrees
        circle_x = longitude + circle_radius * np.cos(theta)
        circle_y = latitude + circle_radius * np.sin(theta)
        ax_main.plot(circle_x, circle_y, 'r-', linewidth=1.5, transform=ccrs.PlateCarree())

        # Add text with coordinates
        ax_main.text(longitude + 0.3, latitude + 0.3,
                     f"Lat: {latitude:.4f}°\nLon: {longitude:.4f}°",
                     horizontalalignment='left',
                     transform=ccrs.PlateCarree(),
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        # Add inset world map if requested
        if show_inset:
            # Create larger inset map
            ax_inset = plt.axes((0.5, 0.08, 0.3, 0.3), projection=ccrs.PlateCarree())

            # Show a much wider area in the inset
            wider_buffer = 25.0
            ax_inset.set_extent([longitude - wider_buffer, longitude + wider_buffer,
                                 latitude - wider_buffer, latitude + wider_buffer],
                                crs=ccrs.PlateCarree())

            # Add coastlines to inset
            ax_inset.add_feature(cfeature.COASTLINE, linewidth=0.5)
            # Solid light grey borders instead of dashed
            ax_inset.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5, edgecolor='gray')

            # Add gridlines without storing in unused variable
            ax_inset.gridlines(draw_labels=False, linewidth=0.3, color='gray', alpha=0.5, linestyle=':')

            # Plot point on inset map
            ax_inset.plot(longitude, latitude, 'o', markersize=5, color='red',
                          transform=ccrs.PlateCarree())

            # Add a box showing the main map extent
            extent = [longitude - buffer, longitude + buffer,
                      latitude - buffer, latitude + buffer]

            # Create box corners
            box_x = [extent[0], extent[1], extent[1], extent[0], extent[0]]
            box_y = [extent[2], extent[2], extent[3], extent[3], extent[2]]

            # Plot box
            ax_inset.plot(box_x, box_y, 'r-', linewidth=1, transform=ccrs.PlateCarree())

        # Add title
        title = f"Study Location"
        subtitle = f"Variable: {self.variable.capitalize()}, Scenario: {self.experiment.upper()}"

        plt.suptitle(title, fontsize=16, weight='bold', x=0.28, ha='left')
        plt.figtext(0.37, 0.92, subtitle, ha='center', fontsize=12, style='italic')

        # Store figure
        self.fig = fig

        return fig
