#!/usr/bin/env python
"""
CMIP6 Data Download Script

This script downloads CMIP6 climate data for specific point locations using the
CDS API. It supports retrieving historical and projection data for temperature
and precipitation variables.

Example usage:
    python download_data.py --lat 43.5 --lon 10.2 --variables temperature precipitation
    python download_data.py --lat 43.5 --lon 10.2 --experiments historical ssp245
    python download_data.py --coordinates coordinates.txt
"""

import sys
import time
import logging
import argparse
from pathlib import Path
from typing import List, Tuple
# Import project modules
from src.data.retrieval import retrieve_all_data_for_point, VARIABLES, EXPERIMENTS

# Add project root to path for importing project modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).with_suffix(".log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("download_data")


def parse_coordinate_file(file_path: Path) -> List[Tuple[float, float]]:
    """
    Parse a file containing coordinates (one pair per line).

    Args:
        file_path: Path to coordinate file

    Returns:
        List of (latitude, longitude) tuples

    Raises:
        ValueError: If file format is invalid
    """
    coordinates = []

    with open(file_path, 'r') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            try:
                parts = line.split()
                if len(parts) != 2:
                    logger.warning(f"Line {i} has invalid format: {line}")
                    continue

                lat, lon = float(parts[0]), float(parts[1])
                coordinates.append((lat, lon))
            except ValueError:
                logger.warning(f"Could not parse coordinates on line {i}: {line}")

    if not coordinates:
        raise ValueError(f"No valid coordinates found in {file_path}")

    return coordinates


def create_data_directories(project_root: Path) -> Path:
    """
    Create the necessary data directories if they don't exist.

    Args:
        project_root: Project root directory

    Returns:
        Path: Path to raw data directory
    """
    raw_dir = project_root / "data" / "raw"
    historical_dir = raw_dir / "historical"
    projections_dir = raw_dir / "projections"

    for directory in [raw_dir, historical_dir, projections_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")

    return raw_dir


def main():
    """Main execution function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Download CMIP6 climate data for specific locations")

    # Define coordinate input methods
    parser.add_argument("--coordinates", type=str, help="Path to file with coordinate pairs (lat lon)")
    parser.add_argument("--lat", type=float, help="Latitude value")
    parser.add_argument("--lon", type=float, help="Longitude value")

    # Define optional arguments
    parser.add_argument("--variables", nargs="+", choices=list(VARIABLES.keys()),
                        default=list(VARIABLES.keys()), help="Climate variables to download")
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS.keys()),
                        default=list(EXPERIMENTS.keys()), help="Experiments to download")
    parser.add_argument("--output-dir", type=str, help="Custom output directory")
    parser.add_argument("--retry", type=int, default=3,
                        help="Number of retry attempts for failed downloads")
    parser.add_argument("--retry-delay", type=int, default=60,
                        help="Delay between retry attempts in seconds")

    args = parser.parse_args()

    # Validate coordinates are provided in some form
    if not args.coordinates and (args.lat is None or args.lon is None):
        parser.error("Either --coordinates file or both --lat and --lon must be provided")

    # Determine project root directory
    project_root = Path(__file__).resolve().parents[1]

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = create_data_directories(project_root)

    # Get list of coordinates to process
    if args.coordinates:
        coord_file = Path(args.coordinates)
        if not coord_file.exists():
            logger.error(f"Coordinate file not found: {coord_file}")
            sys.exit(1)
        coordinates = parse_coordinate_file(coord_file)
        logger.info(f"Loaded {len(coordinates)} coordinate pairs from {coord_file}")
    else:
        coordinates = [(args.lat, args.lon)]

    # Process each coordinate
    logger.info(f"Starting download for {len(coordinates)} locations")
    for i, (lat, lon) in enumerate(coordinates, 1):
        logger.info(f"Processing coordinate {i}/{len(coordinates)}: lat={lat}, lon={lon}")
        logger.info(
            f"Area: [{round(lat+1.0)}, {round(lon-1.0)}, "
            f"{round(lat-1.0)}, {round(lon+1.0)}] (North, West, South, East)"
        )

        # Try with retries
        for attempt in range(1, args.retry + 1):
            try:
                results = retrieve_all_data_for_point(
                    latitude=lat,
                    longitude=lon,
                    output_dir=output_dir,
                    variables=args.variables,
                    experiments=args.experiments
                )

                # Log results
                successful = sum(1 for path in results.values() if path is not None)
                logger.info(f"Successfully downloaded {successful}/{len(results)} files for lat={lat}, lon={lon}")
                break

            except Exception as e:
                logger.error(f"Attempt {attempt}/{args.retry} failed: {str(e)}")
                if attempt < args.retry:
                    logger.info(f"Retrying in {args.retry_delay} seconds...")
                    time.sleep(args.retry_delay)
                else:
                    logger.error(f"Failed to download data for lat={lat}, lon={lon} after {args.retry} attempts")

    logger.info("Download process completed")


if __name__ == "__main__":
    main()
