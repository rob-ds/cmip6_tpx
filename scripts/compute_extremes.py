#!/usr/bin/env python
"""
CMIP6 Climate Extremes Computation Script

This script computes extreme event metrics from CMIP6 climate data for specific
point locations. It supports both temperature and precipitation variables, and
both ssp245 and ssp585 experiments, as well as compound extreme analyses.

Example usage:
    python compute_extremes.py --variable temperature --experiment ssp245 --month 7
    python compute_extremes.py --variable precipitation --experiment ssp585 --month 1 --threshold 90
    python compute_extremes.py --variable temperature --secondary-variable precipitation --experiment ssp245 --month 7
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
    parser = argparse.ArgumentParser(description="Compute climate extreme metrics from CMIP6 data")

    # Define required arguments
    parser.add_argument("--variable", type=str, choices=["temperature", "precipitation"],
                        help="Primary climate variable to analyze")
    parser.add_argument("--experiment", type=str, choices=["ssp245", "ssp585"],
                        help="Experiment to analyze")
    parser.add_argument("--month", type=int, required=True, choices=range(1, 13),
                        help="Month to analyze (1-12)")

    # Define compound analysis arguments
    parser.add_argument("--secondary-variable", type=str, choices=["temperature", "precipitation"],
                        help="Secondary variable for compound extreme analysis")

    # Define optional arguments
    parser.add_argument("--all-combinations", action="store_true",
                        help="Analyze all variable and experiment combinations")
    parser.add_argument("--input-dir", type=str,
                        help="Custom input directory")
    parser.add_argument("--output-dir", type=str,
                        help="Custom output directory")

    # Threshold parameters
    parser.add_argument("--threshold", type=float, default=95.0,
                        help="Percentile threshold for extreme event detection (default: 95.0)")
    parser.add_argument("--secondary-threshold", type=float, default=95.0,
                        help="Percentile threshold for secondary variable (default: 95.0)")
    parser.add_argument("--wet-day-threshold", type=float, default=1.0,
                        help="Threshold for wet day identification (mm/day)")
    parser.add_argument("--dry-day-threshold", type=float, default=1.0,
                        help="Threshold for dry day identification in hot-dry analysis (mm/day)")

    # Heat wave parameters
    parser.add_argument("--heat-wave-min-duration", type=int, default=3,
                        help="Minimum number of consecutive days for heat wave detection (default: 3)")

    args = parser.parse_args()

    # Validate arguments
    if not args.all_combinations and (args.variable is None or args.experiment is None):
        parser.error("Either --all-combinations or both --variable and --experiment must be provided")

    if args.secondary_variable and args.secondary_variable == args.variable:
        parser.error("Primary and secondary variables must be different")

    # Determine project root directory
    project_root = Path(__file__).resolve().parents[1]

    # Determine input and output directories
    input_dir = Path(args.input_dir) if args.input_dir else project_root / "data"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "data" / "processed" / "extremes"

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define combinations to analyze
    variables = ["temperature", "precipitation"]
    experiments = ["ssp245", "ssp585"]

    if args.all_combinations:
        if args.secondary_variable:
            logger.warning("--all-combinations flag ignores --secondary-variable")

        combinations = [(var, exp, None) for var in variables for exp in experiments]
    else:
        combinations = [(args.variable, args.experiment, args.secondary_variable)]

    logger.info(f"Analyzing {len(combinations)} combinations for month {args.month}")

    # Process each combination
    for variable, experiment, secondary_variable in combinations:
        logger.info(f"Processing {variable}, {experiment}, month {args.month}")
        if secondary_variable:
            logger.info(f"Secondary variable: {secondary_variable}")

        # Create analyzer
        analyzer = ExtremesAnalyzer(
            variable=variable,
            experiment=experiment,
            month=args.month,
            input_dir=input_dir,
            output_dir=output_dir,
            threshold_percentile=args.threshold,
            secondary_variable=secondary_variable,
            secondary_threshold_percentile=args.secondary_threshold,
            wet_day_threshold=args.wet_day_threshold,
            dry_day_threshold=args.dry_day_threshold,
            heat_wave_min_duration=args.heat_wave_min_duration,
        )

        # Compute extremes
        results = analyzer.compute()

        # Save results
        output_file = analyzer.save_results(results)

        logger.info(f"Results saved to {output_file}")

    logger.info("Extremes computation completed")


if __name__ == "__main__":
    main()
