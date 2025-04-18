"""Geographic coordinates utility functions."""

from typing import Tuple


def format_coordinates_cardinal(latitude: float, longitude: float, precision: int = 2) -> Tuple[str, str, str]:
    """
    Format geographic coordinates in cardinal format (N/S, E/W).

    Args:
        latitude: Latitude value in decimal degrees
        longitude: Longitude value in decimal degrees
        precision: Number of decimal places to display

    Returns:
        Tuple containing (latitude string, longitude string, combined location string)
    """
    # Format latitude
    lat_dir = "N" if latitude >= 0 else "S"
    lat_val = abs(latitude)
    lat_str = f"{lat_val:.{precision}f}°{lat_dir}"

    # Format longitude (convert values >180 to negative/western values)
    adj_lon = longitude if longitude <= 180 else longitude - 360
    lon_dir = "E" if adj_lon >= 0 else "W"
    lon_val = abs(adj_lon)
    lon_str = f"{lon_val:.{precision}f}°{lon_dir}"

    # Full location string
    location_str = f"{lat_str}, {lon_str}"

    return lat_str, lon_str, location_str
