"""
CMIP6 Data Retrieval Module

This module provides functions to retrieve CMIP6 climate data from the Climate Data Store (CDS)
API for specific point locations. It supports downloading both historical and projection data
for temperature and precipitation variables.

Functions:
    create_cds_client(): Create and return a CDS API client
    build_request_params(): Builds CDS request parameters
    get_output_filename(): Generates appropriate output filenames
    extract_netcdf_from_zip(): Extracts NetCDF files from zip archives
    retrieve_cmip6_data(): Main function to retrieve CMIP6 data for specific coordinates
    retrieve_all_data_for_point(): Retrieves multiple datasets for a single coordinate point
"""

import logging
import cdsapi
import zipfile
import shutil
from pathlib import Path
from typing import Dict, List, Union, Optional

# Configure module logger
logger = logging.getLogger(__name__)

# Constants
DATASET = "projections-cmip6"
VARIABLES = {
    "temperature": "near_surface_air_temperature",
    "precipitation": "precipitation"
}
EXPERIMENTS = {
    "historical": "historical",
    "ssp245": "ssp2_4_5",
    "ssp585": "ssp5_8_5"
}
TEMPORAL_RESOLUTION = "daily"
ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]

# Year ranges
HISTORICAL_YEARS = [str(y) for y in range(1995, 2015)]
PROJECTION_YEARS = [str(y) for y in range(2015, 2101)]


def create_cds_client() -> cdsapi.Client:
    """
    Create and return a CDS API client.

    Returns:
        cdsapi.Client: Initialized CDS API client

    Raises:
        RuntimeError: If CDS API key is not properly configured
    """
    try:
        client = cdsapi.Client()
        return client
    except Exception as e:
        logger.error(f"Failed to initialize CDS API client: {str(e)}")
        raise RuntimeError(f"CDS API client initialization failed: {str(e)}") from e


def build_request_params(
        variable: str,
        experiment: str,
        latitude: float,
        longitude: float,
        years: Optional[List[str]] = None
) -> Dict:
    """
    Build request parameters for the CDS API.

    Args:
        variable: Climate variable ('temperature' or 'precipitation')
        experiment: Experiment type ('historical', 'ssp245', or 'ssp585')
        latitude: Latitude coordinate (float)
        longitude: Longitude coordinate (float)
        years: Optional list of years to retrieve (defaults to appropriate years for experiment)

    Returns:
        Dict: Dictionary of request parameters for CDS API

    Raises:
        ValueError: If invalid variable or experiment is provided
    """
    if variable not in VARIABLES:
        raise ValueError(f"Invalid variable: {variable}. Must be one of {list(VARIABLES.keys())}")

    if experiment not in EXPERIMENTS:
        raise ValueError(f"Invalid experiment: {experiment}. Must be one of {list(EXPERIMENTS.keys())}")

    # Set appropriate years based on experiment if not explicitly provided
    if years is None:
        if experiment == "historical":
            years = HISTORICAL_YEARS
        else:
            years = PROJECTION_YEARS

    # Round coordinates to integer values for consistent area definition
    lat_rounded = round(latitude)
    lon_rounded = round(longitude)

    # Create request parameters
    request_params = {
        "temporal_resolution": TEMPORAL_RESOLUTION,
        "experiment": EXPERIMENTS[experiment],
        "variable": VARIABLES[variable],
        "model": "ec_earth3_cc",  # Using EC-Earth3-CC model
        "year": years,
        "month": ALL_MONTHS,
        "day": ALL_DAYS,
        # Use a standard area size that's known to work
        # CDS requires [North, West, South, East] format
        "area": [lat_rounded + 1.0, lon_rounded - 1.0,
                 lat_rounded - 1.0, lon_rounded + 1.0]
    }

    return request_params


def get_output_filename(
        variable: str,
        experiment: str,
        latitude: float,
        longitude: float,
        year_range: str
) -> str:
    """
    Generate standardized output filename for downloaded data.

    Args:
        variable: Climate variable
        experiment: Experiment type
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        year_range: String representing year range (e.g., "1995-2014")

    Returns:
        str: Standardized filename
    """
    # Format coordinates for filename
    lat_str = f"lat{latitude:.2f}".replace('.', 'p').replace('-', 'n')
    lon_str = f"lon{longitude:.2f}".replace('.', 'p').replace('-', 'n')

    # Create filename
    filename = f"cmip6_{experiment}_{variable}_{lat_str}_{lon_str}_{year_range}.nc"

    return filename


def extract_netcdf_from_zip(zip_path: Path, extract_dir: Path) -> Path:
    """
    Extract NetCDF file from a zip archive.

    Args:
        zip_path: Path to the zip file
        extract_dir: Directory to extract to

    Returns:
        Path: Path to the extracted NetCDF file

    Raises:
        RuntimeError: If no NetCDF file is found in the archive
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List all files in the zip
            file_list = zip_ref.namelist()
            logger.info(f"Files in zip archive: {file_list}")

            # Look for NetCDF files (typically .nc extension)
            nc_files = [f for f in file_list if f.lower().endswith('.nc')]

            if not nc_files:
                # If no .nc files found, look for other potentially relevant files
                data_files = [f for f in file_list if not f.endswith('/') and not f.startswith('__MACOSX')]

                if not data_files:
                    raise RuntimeError(f"No data files found in zip archive: {file_list}")

                # Extract all data files
                for file_path in data_files:
                    zip_ref.extract(file_path, extract_dir)

                # Return the path to the first extracted file
                return extract_dir / data_files[0]

            # Extract only the NetCDF files
            for nc_file in nc_files:
                zip_ref.extract(nc_file, extract_dir)

            # Return the path to the first extracted NetCDF file
            return extract_dir / nc_files[0]
    except zipfile.BadZipFile:
        # File is not a zip file, might already be a NetCDF file
        logger.warning(f"File {zip_path} is not a valid zip file. Treating as direct NetCDF file.")
        target_path = extract_dir / zip_path.name
        # Copy the file rather than move, in case path validation is needed
        shutil.copy2(zip_path, target_path)
        return target_path


def retrieve_cmip6_data(
        variable: str,
        experiment: str,
        latitude: float,
        longitude: float,
        output_dir: Union[str, Path],
        years: Optional[List[str]] = None
) -> Path:
    """
    Retrieve CMIP6 data for a specific point location.

    Args:
        variable: Climate variable ('temperature' or 'precipitation')
        experiment: Experiment type ('historical', 'ssp245', or 'ssp585')
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        output_dir: Directory to store downloaded data
        years: Optional list of years to retrieve (defaults to appropriate years for experiment)

    Returns:
        Path: Path to the downloaded file

    Raises:
        ValueError: If invalid parameters are provided
        RuntimeError: If download fails
    """
    # Validate and prepare output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary directory for download and extraction
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Determine destination based on experiment
    if experiment == "historical":
        dest_dir = output_dir / "historical"
    else:
        dest_dir = output_dir / "projections"

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Determine years to use and create year range string for filename
    if years is None:
        if experiment == "historical":
            years = HISTORICAL_YEARS
            year_range = "1995-2014"
        else:
            years = PROJECTION_YEARS
            year_range = "2015-2100"
    else:
        year_range = f"{years[0]}-{years[-1]}"

    # Build request parameters
    request_params = build_request_params(
        variable=variable,
        experiment=experiment,
        latitude=latitude,
        longitude=longitude,
        years=years
    )

    # Generate output filename
    output_filename = get_output_filename(
        variable=variable,
        experiment=experiment,
        latitude=latitude,
        longitude=longitude,
        year_range=year_range
    )

    output_file = dest_dir / output_filename

    # Check if file already exists
    if output_file.exists():
        logger.info(f"File already exists: {output_file}")
        return output_file

    # Create client and download data
    logger.info(f"Downloading {variable} data for {experiment}")

    try:
        client = create_cds_client()

        # Download to a temporary zip file
        temp_zip = temp_dir / f"temp_{output_filename}.zip"

        # Remove any existing temp files
        if temp_zip.exists():
            temp_zip.unlink()

        # Download the file
        client.retrieve(DATASET, request_params, str(temp_zip))

        # Verify download success
        if not temp_zip.exists():
            raise RuntimeError("Download failed: No file was created")

        # Check file size
        file_size = temp_zip.stat().st_size
        logger.info(f"Downloaded file size: {file_size} bytes")

        if file_size < 1000:
            # Very small files are likely error messages
            with open(temp_zip, 'r', errors='ignore') as f:
                content = f.read(1000)
                logger.error(f"Download appears to be an error message: {content[:200]}...")
            raise RuntimeError(f"Downloaded file is too small ({file_size} bytes)")

        # Extract the NetCDF file from the zip
        extracted_file = extract_netcdf_from_zip(temp_zip, temp_dir)

        # Move extracted file to final destination
        shutil.move(str(extracted_file), str(output_file))

        # Clean up temporary files
        if temp_zip.exists():
            temp_zip.unlink()

        # Try to remove any remaining files in temp_dir (except the directory itself)
        for item in temp_dir.iterdir():
            if item.is_file():
                try:
                    item.unlink()
                except Exception as e:
                    logger.warning(f"Could not remove temporary file {item}: {e}")

        logger.info(f"Successfully downloaded data to {output_file}")
        return output_file

    except Exception as e:
        logger.error(f"Failed to download data: {str(e)}")

        # Clean up any temporary files
        temp_zip = temp_dir / f"temp_{output_filename}.zip"
        if temp_zip.exists():
            temp_zip.unlink()

        # Clean up failed output file if it exists
        if output_file.exists() and output_file.stat().st_size == 0:
            output_file.unlink()

        raise RuntimeError(f"Data download failed: {str(e)}") from e


def retrieve_all_data_for_point(
        latitude: float,
        longitude: float,
        output_dir: Union[str, Path],
        variables: List[str] = None,
        experiments: List[str] = None
) -> Dict[str, Path]:
    """
    Retrieve all requested data types for a specific point location.

    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        output_dir: Directory to store downloaded data
        variables: List of variables to download (defaults to all)
        experiments: List of experiments to download (defaults to all)

    Returns:
        Dict[str, Path]: Dictionary mapping data types to downloaded file paths
    """
    if variables is None:
        variables = list(VARIABLES.keys())

    if experiments is None:
        experiments = list(EXPERIMENTS.keys())

    results = {}

    for experiment in experiments:
        for variable in variables:
            key = f"{experiment}_{variable}"
            try:
                file_path = retrieve_cmip6_data(
                    variable=variable,
                    experiment=experiment,
                    latitude=latitude,
                    longitude=longitude,
                    output_dir=output_dir
                )
                results[key] = file_path
            except Exception as e:
                logger.error(f"Failed to download {key}: {str(e)}")
                results[key] = None

    return results
