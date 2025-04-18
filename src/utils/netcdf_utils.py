"""Utilities for working with NetCDF climate data files."""

import logging
import xarray as xr
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def format_coordinate(value: float, prefix: str = "") -> str:
    """
        Format a coordinate value for use in filenames.

        Args:
            value: Coordinate value (latitude or longitude)
            prefix: Optional prefix to add (e.g., 'lat' or 'lon')

        Returns:
            Formatted coordinate string (e.g., 'lat38p25' for 38.25)
        """
    formatted = f"{value:.2f}".replace('.', 'p').replace('-', 'n')
    if prefix:
        return f"{prefix}{formatted}"
    return formatted


def find_netcdf_file(
    directory: Path,
    name_pattern: str,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None
) -> Path:
    """
    Find a NetCDF file in a directory that matches a specified name pattern.

    This function searches for NetCDF files (.nc extension) in the specified directory
    that contain the given name pattern. When latitude and longitude coordinates are
    provided, it attempts to find files that specifically match these coordinates
    in their filenames. If multiple files match the pattern, the function prioritizes:

    1. First, files that contain the specific coordinates in their names
    2. If no coordinate-specific matches are found, selects the most recently modified file

    This is useful for CMIP6 climate data files that include location information
    in their filenames (e.g., lat38p50_lon240p75.nc for 38.50°N, 240.75°E).

    Args:
        directory (Path): Directory path to search for NetCDF files
        name_pattern (str): String pattern to match in filenames
        latitude (Optional[float]): Latitude coordinate to match in filenames,
                                    formatted as "latXXpYY" where XX.YY is the value
        longitude (Optional[float]): Longitude coordinate to match in filenames,
                                     formatted as "lonXXpYY" where XX.YY is the value

    Returns:
        Path: Path object pointing to the matched NetCDF file

    Raises:
        FileNotFoundError: If no files matching the pattern are found in the directory

    Examples:
        >>> find_netcdf_file(Path("/data/cmip6"), "temperature_ssp245")
        PosixPath('/data/cmip6/cmip6_ec_earth3_cc_ssp245_temperature.nc')

        >>> find_netcdf_file(Path("/data/cmip6"), "precipitation",
        ...                  latitude=38.5, longitude=240.75)
        PosixPath('/data/cmip6/cmip6_ec_earth3_cc_ssp245_precipitation_lat38p50_lon240p75.nc')
    """
    matching_files = []

    for file_path in directory.glob("*.nc"):
        if name_pattern in file_path.name:
            # If coordinates provided, try to match specifically
            if latitude is not None and longitude is not None:
                lat_str = format_coordinate(latitude, "lat")
                lon_str = format_coordinate(longitude, "lon")

                # If file contains the specific coordinate, prioritize it
                if lat_str in file_path.name and lon_str in file_path.name:
                    logger.debug(f"Found exact coordinate match: {file_path}")
                    return file_path

            matching_files.append(file_path)

    if not matching_files:
        raise FileNotFoundError(f"No file matching '{name_pattern}' found in {directory}")

    # If multiple files match but none with exact coordinates, select the most recent one
    newest_file = sorted(matching_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    logger.debug(f"Selected newest matching file: {newest_file}")
    return newest_file


def find_and_load_cmip6_data(
        input_dir: Path,
        variable: str,
        experiment: str,
        model: str,
        log: logging.Logger,
        coords: Optional[Tuple[float, float]] = None
) -> Tuple[xr.Dataset, xr.Dataset, float, float, int, int]:
    """
    Find and load CMIP6 historical and projection NetCDF data.

    This function handles both new-format and legacy-format filenames with fallback logic.
    It loads both historical and projection datasets for the specified variable,
    and extracts the center point coordinates.

    Args:
        input_dir: Base input directory containing raw/historical and raw/projections
        variable: Climate variable name (e.g., 'temperature', 'precipitation')
        experiment: Experiment name (e.g., 'ssp245', 'ssp585')
        model: CMIP6 model name
        log: Logger instance for tracking progress
        coords: Optional tuple of (latitude, longitude) to avoid recomputing

    Returns:
        Tuple containing:
        - historical_ds: Historical dataset
        - projection_ds: Projection dataset
        - lat: Center latitude value
        - lon: Center longitude value
        - lat_idx: Center latitude index
        - lon_idx: Center longitude index
    """
    # New format with model pattern
    historical_pattern = f"cmip6_{model}_historical_{variable}"
    projection_pattern = f"cmip6_{model}_{experiment}_{variable}"

    # Legacy format (fallback)
    legacy_historical_pattern = f"historical_{variable}"
    legacy_projection_pattern = f"{experiment}_{variable}"

    # Try to find historical file with fallback
    try:
        historical_file = find_netcdf_file(
            input_dir / "raw" / "historical",
            historical_pattern,
            coords[0] if coords else None,
            coords[1] if coords else None
        )
        log.info(f"Found {variable} historical data using new filename pattern: {historical_file}")
    except FileNotFoundError:
        historical_file = find_netcdf_file(
            input_dir / "raw" / "historical",
            legacy_historical_pattern,
            coords[0] if coords else None,
            coords[1] if coords else None
        )
        log.info(f"Found {variable} historical data using legacy filename pattern: {historical_file}")

    # Try to find projection file with fallback
    try:
        projection_file = find_netcdf_file(
            input_dir / "raw" / "projections",
            projection_pattern,
            coords[0] if coords else None,
            coords[1] if coords else None
        )
        log.info(f"Found {variable} projection data using new filename pattern: {projection_file}")
    except FileNotFoundError:
        projection_file = find_netcdf_file(
            input_dir / "raw" / "projections",
            legacy_projection_pattern,
            coords[0] if coords else None,
            coords[1] if coords else None
        )
        log.info(f"Found {variable} projection data using legacy filename pattern: {projection_file}")

    log.info(f"Loading {variable} historical data from {historical_file}")
    log.info(f"Loading {variable} projection data from {projection_file}")

    # Load datasets
    historical_ds = xr.open_dataset(historical_file)
    projection_ds = xr.open_dataset(projection_file)

    # Find center indices for lat and lon dimensions
    lat_center_idx = (len(historical_ds.lat) - 1) // 2
    lon_center_idx = (len(historical_ds.lon) - 1) // 2

    # Extract center point coordinates
    lat = float(historical_ds.lat[lat_center_idx].values)
    lon = float(historical_ds.lon[lon_center_idx].values)

    log.info(f"Center point coordinates: lat={lat}, lon={lon} "
             f"(from {len(historical_ds.lat)}×{len(historical_ds.lon)} grid)")

    return historical_ds, projection_ds, lat, lon, lat_center_idx, lon_center_idx
