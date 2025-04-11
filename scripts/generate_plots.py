#!/usr/bin/env python
"""
CMIP6 Climate Visualization Script

This script creates visualizations from processed CMIP6 climate data. It supports
both anomaly decomposition and extreme event visualizations.

Example usage:
    python generate_plots.py --variable temperature --experiment ssp245 --month 7 --plot-type anomaly
    python generate_plots.py --variable temperature --experiment ssp585 --month 7 --plot-type heatwave
    python generate_plots.py --variable temperature --experiment ssp585 --plot-type heatmap
    python generate_plots.py --variable temperature --experiment ssp585 --month 7 --plot-type location
    python generate_plots.py --all-plots --variable temperature --experiment ssp585
"""

import sys
import logging
import argparse
from pathlib import Path

# Import project modules
from src.visualization.timeseries import TimeSeriesPlotter
from src.visualization.extreme_statistics import ExtremeStatisticsPlotter
from src.visualization.location_map import LocationMapPlotter
from src.visualization.export import export_figure

# Add project root to path for importing project modules
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).with_suffix(".log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("generate_plots")


def main():
    """Main execution function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate visualizations from CMIP6 climate data")

    # Define required arguments
    parser.add_argument("--variable", type=str, choices=["temperature", "precipitation"],
                        help="Climate variable to analyze")
    parser.add_argument("--experiment", type=str, choices=["ssp245", "ssp585"],
                        help="Experiment to analyze")
    parser.add_argument("--month", type=int, choices=range(1, 13),
                        help="Month to analyze (1-12)")

    # Plot type selection
    parser.add_argument("--plot-type", type=str,
                        choices=["anomaly", "heatwave", "heatmap", "location", "all"],
                        help="Type of plot to generate")
    parser.add_argument("--all-plots", action="store_true",
                        help="Generate all plot types")

    # Define optional arguments
    parser.add_argument("--input-dir", type=str,
                        help="Custom input directory")
    parser.add_argument("--output-dir", type=str,
                        help="Custom output directory")
    parser.add_argument("--formats", type=str, default="png,pdf",
                        help="Comma-separated list of output formats")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Resolution for raster formats")
    parser.add_argument("--figsize", type=str, default="12,10",
                        help="Figure size in inches (width,height)")

    args = parser.parse_args()

    # Validate arguments
    if args.all_plots:
        if not args.variable or not args.experiment:
            parser.error("--all-plots requires --variable and --experiment")
        # Set plot_type to "all"
        args.plot_type = "all"
    elif args.plot_type is None:
        parser.error("Either --all-plots or --plot-type must be specified")

    # Month is not needed for heatmap plot
    if args.plot_type not in ["heatmap", "all"] and args.month is None:
        parser.error("--month is required for plot types other than 'heatmap'")

    # Determine project root directory
    project_dir = Path(__file__).resolve().parents[1]

    # Determine input and output directories
    input_dir = Path(args.input_dir) if args.input_dir else project_dir / "data"
    output_dir = Path(args.output_dir) if args.output_dir else project_dir / "figures"

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse figure size
    try:
        figsize = tuple(map(float, args.figsize.split(',')))
    except ValueError:
        logger.warning(f"Invalid figsize format: {args.figsize}, using default (12,10)")
        figsize = (12, 10)

    # Parse output formats
    formats = args.formats.split(',')

    logger.info(f"Generating {args.plot_type} plot for {args.variable}, {args.experiment}")

    # Generate requested plots
    if args.plot_type in ["anomaly", "all"]:
        if args.month is not None:
            generate_anomaly_plot(
                variable=args.variable,
                experiment=args.experiment,
                month=args.month,
                input_dir=input_dir,
                output_dir=output_dir,
                formats=formats,
                dpi=args.dpi,
                figsize=figsize
            )

    if args.plot_type in ["heatwave", "all"]:
        if args.variable == "temperature" and args.month is not None:
            generate_heatwave_plot(
                variable=args.variable,
                experiment=args.experiment,
                month=args.month,
                input_dir=input_dir,
                output_dir=output_dir,
                formats=formats,
                dpi=args.dpi,
                figsize=figsize
            )
        elif args.plot_type == "heatwave":
            logger.warning("Heatwave plots are only available for temperature variable")

    if args.plot_type in ["heatmap", "all"]:
        if args.variable == "temperature":
            generate_heatmap_plot(
                variable=args.variable,
                experiment=args.experiment,
                input_dir=input_dir,
                output_dir=output_dir,
                formats=formats,
                dpi=args.dpi,
                figsize=figsize
            )
        elif args.plot_type == "heatmap":
            logger.warning("Heatmap plots are only available for temperature variable")

    if args.plot_type in ["location", "all"]:
        if args.month is not None:
            generate_location_plot(
                variable=args.variable,
                experiment=args.experiment,
                month=args.month,
                input_dir=input_dir,
                output_dir=output_dir,
                formats=formats,
                dpi=args.dpi,
                figsize=figsize
            )

    logger.info("Plot generation completed")


def generate_anomaly_plot(variable, experiment, month, input_dir, output_dir, formats, dpi, figsize):
    """Generate multi-scale anomaly decomposition plot."""
    logger.info(f"Generating anomaly plot for {variable}, {experiment}, month {month}")

    # Create plotter
    plotter = TimeSeriesPlotter(
        variable=variable,
        experiment=experiment,
        month=month,
        input_dir=input_dir,
        output_dir=output_dir,
        dpi=dpi,
        figsize=figsize
    )

    # Generate plot
    fig = plotter.plot_anomaly_decomposition()

    # Export figure
    filename = f"{variable}_{experiment}_month{month:02d}_anomaly_decomposition"
    metadata = {
        "Variable": variable,
        "Experiment": experiment,
        "Month": str(month),
        "Type": "Anomaly Decomposition"
    }

    export_figure(
        fig=fig,
        output_dir=output_dir,
        filename=filename,
        formats=formats,
        dpi=dpi,
        metadata=metadata
    )

    # Close figure
    plotter.close()

    logger.info(f"Anomaly plot saved to {output_dir}/{filename}")


def generate_heatwave_plot(variable, experiment, month, input_dir, output_dir, formats, dpi, figsize):
    """Generate heat wave metrics plot."""
    logger.info(f"Generating heat wave plot for {variable}, {experiment}, month {month}")

    # Create plotter
    plotter = ExtremeStatisticsPlotter(
        variable=variable,
        experiment=experiment,
        month=month,
        input_dir=input_dir,
        output_dir=output_dir,
        dpi=dpi,
        figsize=figsize
    )

    # Generate plot
    fig = plotter.plot_heat_wave_metrics()

    # Export figure
    filename = f"{variable}_{experiment}_month{month:02d}_heat_wave_metrics"
    metadata = {
        "Variable": variable,
        "Experiment": experiment,
        "Month": str(month),
        "Type": "Heat Wave Metrics"
    }

    export_figure(
        fig=fig,
        output_dir=output_dir,
        filename=filename,
        formats=formats,
        dpi=dpi,
        metadata=metadata
    )

    # Close figure
    plotter.close()

    logger.info(f"Heat wave plot saved to {output_dir}/{filename}")


def generate_heatmap_plot(variable, experiment, input_dir, output_dir, formats, dpi, figsize):
    """Generate monthly heat wave heatmap."""
    logger.info(f"Generating monthly heat wave heatmap for {variable}, {experiment}")

    # Create plotter
    plotter = ExtremeStatisticsPlotter(
        variable=variable,
        experiment=experiment,
        month=1,  # Doesn't matter for heatmap
        input_dir=input_dir,
        output_dir=output_dir,
        dpi=dpi,
        figsize=figsize
    )

    # Generate plot
    fig = plotter.create_monthly_heatmap()

    # Export figure
    filename = f"{variable}_{experiment}_heat_wave_monthly_heatmap"
    metadata = {
        "Variable": variable,
        "Experiment": experiment,
        "Type": "Heat Wave Monthly Heatmap"
    }

    export_figure(
        fig=fig,
        output_dir=output_dir,
        filename=filename,
        formats=formats,
        dpi=dpi,
        metadata=metadata
    )

    # Close figure
    plotter.close()

    logger.info(f"Monthly heatmap saved to {output_dir}/{filename}")


def generate_location_plot(variable, experiment, month, input_dir, output_dir, formats, dpi, figsize):
    """Generate location map."""
    logger.info(f"Generating location map for {variable}, {experiment}, month {month}")

    # Create plotter
    plotter = LocationMapPlotter(
        variable=variable,
        experiment=experiment,
        month=month,
        input_dir=input_dir,
        output_dir=output_dir,
        dpi=dpi,
        figsize=figsize
    )

    # Generate plot
    fig = plotter.plot_location_map()

    # Export figure
    filename = f"{variable}_{experiment}_location_map"

    # Get lat/lon from plotter
    metadata = {
        "Variable": variable,
        "Experiment": experiment,
        "Latitude": f"{plotter.latitude:.4f}",
        "Longitude": f"{plotter.longitude:.4f}",
        "Type": "Location Map"
    }

    export_figure(
        fig=fig,
        output_dir=output_dir,
        filename=filename,
        formats=formats,
        dpi=dpi,
        metadata=metadata
    )

    # Close figure
    plotter.close()

    logger.info(f"Location map saved to {output_dir}/{filename}")


if __name__ == "__main__":
    main()
