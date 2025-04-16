#!/usr/bin/env python
"""
CMIP6 Multi-scale Anomaly Computation Script

This script computes multiscale anomalies from CMIP6 climate data for specific
point locations. It supports both temperature and precipitation variables, and
both ssp245 and ssp585 experiments across multiple CMIP6 models.

Example usage:
    python compute_anomalies.py --variable temperature --experiment ssp245 --month 7 --model ec_earth3_cc
    python compute_anomalies.py --variable precipitation --experiment ssp585 --month 1 --model hadgem3_gc31_mm
    python compute_anomalies.py --all-combinations --month 7 --model ec_earth3_cc
"""

import sys
import logging
import argparse
from pathlib import Path

# Import project modules
from src.analysis.variability import VariabilityAnalyzer
from src.data.retrieval import MODELS  # Import available models

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
logger = logging.getLogger("compute_anomalies")


def main():
    """Main execution function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Compute multi-scale anomalies from CMIP6 climate data")

    # Define required arguments
    parser.add_argument("--variable", type=str, choices=["temperature", "precipitation"],
                        help="Climate variable to analyze")
    parser.add_argument("--experiment", type=str, choices=["ssp245", "ssp585"],
                        help="Experiment to analyze")
    parser.add_argument("--month", type=int, required=True, choices=range(1, 13),
                        help="Month to analyze (1-12)")

    # Add model selection argument
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()),
                        help=f"CMIP6 model to use. Available models: {', '.join(MODELS.keys())}")

    # Define optional arguments
    parser.add_argument("--all-combinations", action="store_true",
                        help="Analyze all variable and experiment combinations")
    parser.add_argument("--input-dir", type=str,
                        help="Custom input directory")
    parser.add_argument("--output-dir", type=str,
                        help="Custom output directory")
    parser.add_argument("--highpass-cutoff", type=float, default=10.0,
                        help="Cutoff period for high-pass filter (years)")
    parser.add_argument("--lowpass-cutoff", type=float, default=20.0,
                        help="Cutoff period for low-pass filter (years)")
    parser.add_argument("--window-size", type=int, default=11,
                        help="Size of sliding window for standard deviation (years)")

    args = parser.parse_args()

    # Validate arguments
    if not args.all_combinations and (args.variable is None or args.experiment is None):
        parser.error("Either --all-combinations or both --variable and --experiment must be provided")

    # Determine project root directory
    project_root = Path(__file__).resolve().parents[1]

    # Determine input and output directories
    input_dir = Path(args.input_dir) if args.input_dir else project_root / "data"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "data" / "processed" / "anomalies"

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define combinations to analyze
    variables = ["temperature", "precipitation"]
    experiments = ["ssp245", "ssp585"]

    if args.all_combinations:
        combinations = [(var, exp) for var in variables for exp in experiments]
    else:
        combinations = [(args.variable, args.experiment)]

    logger.info(f"Using model {args.model} ({MODELS[args.model]})")
    logger.info(f"Analyzing {len(combinations)} combinations for month {args.month}")

    # Process each combination
    for variable, experiment in combinations:
        logger.info(f"Processing {variable}, {experiment}, month {args.month}, model {args.model}")

        # Create analyzer
        analyzer = VariabilityAnalyzer(
            variable=variable,
            experiment=experiment,
            month=args.month,
            input_dir=input_dir,
            output_dir=output_dir,
            model=args.model,  # Pass model parameter
            highpass_cutoff=args.highpass_cutoff,
            lowpass_cutoff=args.lowpass_cutoff,
            window_size=args.window_size
        )

        # Compute anomalies
        results = analyzer.compute()

        # Save results
        output_file = analyzer.save_results(results)

        logger.info(f"Results saved to {output_file}")

    logger.info("Anomaly computation completed")


if __name__ == "__main__":
    main()
