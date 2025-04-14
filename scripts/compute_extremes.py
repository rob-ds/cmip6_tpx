#!/usr/bin/env python
"""
CMIP6 Climate Extremes Computation Script

This script computes climate extreme indices from CMIP6 data using an adapted
methodology based on the Expert Team on Climate Change Detection and Indices (ETCCDI).
It processes both temperature and precipitation variables together, applying:

1. Standard ETCCDI precipitation indices adapted for monthly analysis
2. Modified temperature indices appropriate for daily mean temperature data
3. Basic and advanced compound extreme indices capturing temperature-precipitation interactions

Example usage:
    python compute_extremes.py --experiment ssp245 --month 7
    python compute_extremes.py --experiment ssp585 --month 1 --temperature-threshold 90 --precipitation-threshold 95
    python compute_extremes.py --all-experiments --month 7 --heat-wave-min-duration 3
"""

import sys
import logging
import argparse
from pathlib import Path

# Import project modules
from src.analysis.extremes import ExtremesAnalyzer

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
logger = logging.getLogger("compute_extremes")


def main():
    """Main execution function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Compute standardized climate extreme indices from CMIP6 data (adapted ETCCDI methodology)"
    )

    # Define required arguments
    parser.add_argument("--experiment", type=str, choices=["ssp245", "ssp585"],
                        help="Experiment to analyze")
    parser.add_argument("--month", type=int, required=True, choices=range(1, 13),
                        help="Month to analyze (1-12)")

    # Define optional arguments
    parser.add_argument("--all-experiments", action="store_true",
                        help="Analyze all experiments (ssp245 and ssp585)")
    parser.add_argument("--input-dir", type=str,
                        help="Custom input directory")
    parser.add_argument("--output-dir", type=str,
                        help="Custom output directory")

    # Threshold parameters
    parser.add_argument("--temperature-threshold", type=float, default=90.0,
                        help="Percentile threshold for temperature extremes (default: 90.0)")
    parser.add_argument("--precipitation-threshold", type=float, default=95.0,
                        help="Percentile threshold for precipitation extremes (default: 95.0)")
    parser.add_argument("--wet-day-threshold", type=float, default=1.0,
                        help="Threshold for wet day identification (mm/day)")
    parser.add_argument("--dry-day-threshold", type=float, default=1.0,
                        help="Threshold for dry day identification in hot-dry analysis (mm/day)")

    # Heat wave parameters
    parser.add_argument("--heat-wave-min-duration", type=int, default=6,
                        help="Minimum number of consecutive days for warm/cold spell detection (default: 6 per ETCCDI)")

    args = parser.parse_args()

    # Validate arguments
    if not args.all_experiments and args.experiment is None:
        parser.error("Either --all-experiments or --experiment must be provided")

    # Determine project root directory
    project_root = Path(__file__).resolve().parents[1]

    # Determine input and output directories
    input_dir = Path(args.input_dir) if args.input_dir else project_root / "data"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "data" / "processed" / "extremes"

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define experiments to analyze
    experiments = ["ssp245", "ssp585"] if args.all_experiments else [args.experiment]

    logger.info(f"Analyzing {len(experiments)} experiments for month {args.month}")

    # Process each experiment
    for experiment in experiments:
        logger.info(f"Processing {experiment}, month {args.month}")

        # Create analyzer
        analyzer = ExtremesAnalyzer(
            experiment=experiment,
            month=args.month,
            input_dir=input_dir,
            output_dir=output_dir,
            temperature_threshold_percentile=args.temperature_threshold,
            precipitation_threshold_percentile=args.precipitation_threshold,
            wet_day_threshold=args.wet_day_threshold,
            dry_day_threshold=args.dry_day_threshold,
            heat_wave_min_duration=args.heat_wave_min_duration
        )

        # Compute extremes
        results = analyzer.compute()

        # Save results
        output_file = analyzer.save_results(results)
        logger.info(f"Results saved to {output_file}")

    logger.info("Extremes computation completed")


if __name__ == "__main__":
    main()
