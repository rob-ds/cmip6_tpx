#!/usr/bin/env python
"""
CMIP6 Climate Visualization Script

This script creates visualizations from processed CMIP6 climate data with support for:
1. Multi-scale anomaly decomposition
2. Time series analysis of individual climate metrics
3. Monthly heatmaps for temporal patterns
4. Location maps for geographical context

Workflow:
- Select visualization type with --viz-type
- Specify base variable (temperature/precipitation) with --variable
- Choose experiment scenario (ssp245/ssp585) with --experiment
- Select CMIP6 model with --model
- For time series: specify month (--month) and metric (--metric)
- For heatmaps: specify metric (--metric) and optional month selection (--months)
- Optionally override colors with --color-scheme
- Use --consistent-range to create paired plots with matching y-axis scales

Example usage:
    # Temperature examples
    # Anomaly decomposition
    python generate_plots.py --viz-type anomaly --variable temperature --experiment ssp245 --month 7 --model ec_earth3_cc

    # Time series for temperature metric
    python generate_plots.py --viz-type timeseries --variable temperature --experiment ssp585 --month 7 --metric warm_spell_days --model noresm2_mm

    # Paired time series with consistent y-axis range (for direct SSP-4.5-SSP-8.5 scenario comparison)
    python generate_plots.py --viz-type timeseries --variable temperature --month 7 --metric tm_max --consistent-range --model hadgem3_gc31_ll

    # Heatmap for temperature metric across summer months
    python generate_plots.py --viz-type heatmap --variable temperature --experiment ssp585 --metric tm90p --months 6-8 --model ec_earth3_cc

    # Precipitation examples
    # Time series for precipitation metric
    python generate_plots.py --viz-type timeseries --variable precipitation --experiment ssp245 --month 1 --metric rx1day --model ec_earth3_cc

    # Paired time series with consistent y-axis (for direct SSP-4.5-SSP-8.5 scenario comparison)
    python generate_plots.py --viz-type timeseries --variable precipitation --month 10 --metric rx1day --consistent-range --model hadgem3_gc31_ll

    # Heatmap for precipitation metric across all months
    python generate_plots.py --viz-type heatmap --variable precipitation --experiment ssp585 --metric r95p --months all --model noresm2_mm

    # Compound metric examples
    # Time series for compound temperature-precipitation metric
    python generate_plots.py --viz-type timeseries --variable temperature --experiment ssp585 --month 8 --metric hot_dry_frequency --model ec_earth3_cc

    # Heatmap for compound metric with custom color scheme
    python generate_plots.py --viz-type heatmap --variable temperature --experiment ssp585 --metric hot_wet_frequency --months all --color-scheme purples --model hadgem3_gc31_ll

    # Location map
    python generate_plots.py --viz-type location --variable temperature --experiment ssp245 --month 7 --model ec_earth3_cc
"""

import sys
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import List

# Import project modules
from src.visualization.timeseries import TimeSeriesPlotter
from src.visualization.extreme_statistics import ExtremeStatisticsPlotter
from src.visualization.location_map import LocationMapPlotter
from src.visualization.export import export_figure
from src.data.retrieval import MODELS  # Import available models

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

# Available metrics list - used for validation
AVAILABLE_METRICS = [
    # Temperature metrics
    'tm_max', 'tm_min', 'tm90p', 'tm10p', 'warm_spell_days', 'cold_spell_days',

    # Precipitation metrics
    'r95p', 'prcptot', 'rx1day', 'rx5day', 'r10mm', 'r20mm', 'sdii', 'cdd', 'cwd',

    # Compound metrics
    'hot_dry_frequency', 'hot_wet_frequency', 'cold_wet_frequency', 'cwhd', 'wspi', 'wpd',

    # Legacy metrics
    'frequency', 'persistence', 'intensity', 'hw_count', 'hw_days', 'hw_max_duration', 'hw_mean_duration'
]

# Define color scheme options
COLOR_SCHEMES = ['reds', 'blues', 'purples']


def parse_months_argument(months_str: str) -> List[int]:
    """
    Parse the months argument to a list of integers.

    Args:
        months_str: String representation of months (e.g., "1,2,3", "1-6", "all")

    Returns:
        List of month integers
    """
    if months_str.lower() == 'all':
        return list(range(1, 13))

    # Check for range notation
    if '-' in months_str:
        try:
            start, end = map(int, months_str.split('-'))
            if 1 <= start <= 12 and 1 <= end <= 12:
                return list(range(start, end + 1))
            else:
                raise ValueError("Month range must be between 1 and 12")
        except ValueError:
            raise ValueError(f"Invalid month range format: {months_str}")

    # Check for comma-separated list
    if ',' in months_str:
        try:
            months = [int(m) for m in months_str.split(',')]
            if all(1 <= m <= 12 for m in months):
                return months
            else:
                raise ValueError("All months must be between 1 and 12")
        except ValueError:
            raise ValueError(f"Invalid month list format: {months_str}")

    # Try single month
    try:
        month = int(months_str)
        if 1 <= month <= 12:
            return [month]
        else:
            raise ValueError("Month must be between 1 and 12")
    except ValueError:
        raise ValueError(f"Invalid month format: {months_str}")


def generate_paired_metric_timeseries(variable, metric, month, model, input_dir, output_dir,
                                      color_scheme=None, formats=None, dpi=300, figsize=(12, 10)):
    """
    Generate time series plots for both SSP245 and SSP585 scenarios with consistent y-axis.

    Args:
        variable: Climate variable to analyze
        metric: Specific metric to visualize
        month: Month to analyze (1-12)
        model: CMIP6 model to use
        input_dir: Directory containing input data
        output_dir: Directory to store output figures
        color_scheme: Color scheme to use
        formats: Output file formats
        dpi: Resolution for raster formats
        figsize: Figure size (width, height) in inches
    """
    logger.info(
        f"Generating paired time series plots for {metric}, month {month}, model {model} with consistent y-axis")

    # Create temporary plotters to load data for both scenarios
    temp_plotter_ssp245 = ExtremeStatisticsPlotter(
        variable=variable,
        experiment='ssp245',
        month=month,
        input_dir=input_dir,
        output_dir=output_dir,
        model=model,
        metric=metric,
        color_scheme=color_scheme,
        dpi=dpi,
        figsize=figsize
    )

    temp_plotter_ssp585 = ExtremeStatisticsPlotter(
        variable=variable,
        experiment='ssp585',
        month=month,
        input_dir=input_dir,
        output_dir=output_dir,
        model=model,
        metric=metric,
        color_scheme=color_scheme,
        dpi=dpi,
        figsize=figsize
    )

    # Load data for both scenarios
    temp_plotter_ssp245.load_data(data_type='extremes')
    temp_plotter_ssp585.load_data(data_type='extremes')

    # Extract metric data to determine consistent y-axis limits
    data_ssp245 = temp_plotter_ssp245.data[metric].values
    data_ssp585 = temp_plotter_ssp585.data[metric].values

    # Handle timedelta data type if present for either dataset
    try:
        if hasattr(data_ssp245, 'dtype') and np.issubdtype(data_ssp245.dtype, np.timedelta64):
            data_ssp245 = data_ssp245.astype('timedelta64[D]').astype(float)
        if hasattr(data_ssp585, 'dtype') and np.issubdtype(data_ssp585.dtype, np.timedelta64):
            data_ssp585 = data_ssp585.astype('timedelta64[D]').astype(float)
    except (TypeError, ValueError):
        pass

    # Determine global min and max values with a small buffer (5%)
    all_data = np.concatenate([data_ssp245, data_ssp585])
    valid_data = all_data[~np.isnan(all_data)]

    if len(valid_data) > 0:
        global_min = np.min(valid_data)
        global_max = np.max(valid_data)

        # Add 5% buffer on both ends
        y_range = global_max - global_min
        y_min = global_min - 0.05 * y_range
        y_max = global_max + 0.05 * y_range

        logger.info(f"Determined consistent y-axis range: {y_min:.2f} to {y_max:.2f}")
    else:
        logger.warning("No valid data found, using default y-axis range")
        y_min = None
        y_max = None

    # Clean up temporary plotters
    temp_plotter_ssp245.close()
    temp_plotter_ssp585.close()

    # Now create the actual plots with consistent y-axis limits
    for experiment in ['ssp245', 'ssp585']:
        plotter = ExtremeStatisticsPlotter(
            variable=variable,
            experiment=experiment,
            month=month,
            input_dir=input_dir,
            output_dir=output_dir,
            model=model,
            metric=metric,
            color_scheme=color_scheme,
            dpi=dpi,
            figsize=figsize
        )

        # Generate plot with consistent y-axis limits
        fig = plotter.plot_metric_time_series(y_min=y_min, y_max=y_max)

        # Export figure
        filename = f"{variable}_{model}_{experiment}_month{month:02d}_{metric}_consistent_range"
        metadata = {
            "Variable": variable,
            "Experiment": experiment,
            "Month": str(month),
            "Metric": metric,
            "Model": model,
            "Type": "Time Series (Consistent Range)"
        }

        export_figure(
            fig=fig,
            output_dir=output_dir,
            filename=filename,
            formats=formats,
            dpi=dpi,
            metadata=None
        )

        # Close figure
        plotter.close()

    logger.info(f"Paired time series plots with consistent y-axis saved to {output_dir}")


def main():
    """Main execution function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate visualizations from CMIP6 climate data")

    # Define required arguments
    parser.add_argument("--variable", type=str, choices=["temperature", "precipitation"],
                        help="Base climate variable to analyze")
    parser.add_argument("--experiment", type=str, choices=["ssp245", "ssp585"],
                        help="Experiment to analyze")

    # Add model selection
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()),
                        help=f"CMIP6 model to use. Available models: {', '.join(MODELS.keys())}")

    # Visualization type selection
    parser.add_argument("--viz-type", type=str, required=True,
                        choices=["anomaly", "timeseries", "heatmap", "location"],
                        help="Type of visualization to generate")

    # Month specification
    parser.add_argument("--month", type=int, choices=range(1, 13),
                        help="Month to analyze for timeseries and anomaly (1-12)")

    parser.add_argument("--months", type=str,
                        help="Months to analyze for heatmap (comma-separated, range with dash, or 'all')")

    # Metric selection
    parser.add_argument("--metric", type=str, choices=AVAILABLE_METRICS,
                        help="Specific metric to visualize")

    # Color scheme
    parser.add_argument("--color-scheme", type=str, choices=COLOR_SCHEMES,
                        help="Color scheme to use (overrides default based on metric)")

    # Consistent y-axis range for paired metrics
    parser.add_argument("--consistent-range", action="store_true",
                        help="Generate plots for both SSP245 and SSP585 with consistent y-axis range")

    # Define optional arguments
    parser.add_argument("--input-dir", type=str,
                        help="Custom input directory")
    parser.add_argument("--output-dir", type=str,
                        help="Custom output directory")
    parser.add_argument("--formats", type=str, default="png",
                        help="Comma-separated list of output formats")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Resolution for raster formats")
    parser.add_argument("--figsize", type=str, default="12,10",
                        help="Figure size in inches (width,height)")

    args = parser.parse_args()

    # Validate arguments based on visualization type
    if args.viz_type == 'anomaly':
        if not args.variable or not args.experiment or not args.month:
            parser.error("--viz-type anomaly requires --variable, --experiment, and --month")

    elif args.viz_type == 'timeseries':
        if args.consistent_range:
            # When using consistent range, experiment should not be specified
            if not args.variable or not args.month or not args.metric:
                parser.error("--viz-type timeseries with --consistent-range requires --variable, --month, and --metric")
            if args.experiment:
                parser.error(
                    "--experiment should not be specified when using --consistent-range (plots for both scenarios will be generated)")
        else:
            # Original validation for single scenario plotting
            if not args.variable or not args.experiment or not args.month or not args.metric:
                parser.error("--viz-type timeseries requires --variable, --experiment, --month, and --metric")

    elif args.viz_type == 'heatmap':
        if not args.variable or not args.experiment or not args.metric:
            parser.error("--viz-type heatmap requires --variable, --experiment, and --metric")

    elif args.viz_type == 'location':
        if not args.variable or not args.experiment or not args.month:
            parser.error("--viz-type location requires --variable, --experiment, and --month")

    # Log model information
    logger.info(f"Using CMIP6 model: {args.model} ({MODELS[args.model]})")

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

    logger.info(f"Generating {args.viz_type} plot for {args.variable}, {args.experiment}")

    # Generate requested visualization
    if args.viz_type == "anomaly":
        generate_anomaly_plot(
            variable=args.variable,
            experiment=args.experiment,
            month=args.month,
            model=args.model,
            input_dir=input_dir,
            output_dir=output_dir,
            formats=formats,
            dpi=args.dpi,
            figsize=figsize
        )

    # Generate time series plot
    elif args.viz_type == "timeseries":
        if args.consistent_range:
            # Generate paired plots with consistent y-axis range
            generate_paired_metric_timeseries(
                variable=args.variable,
                metric=args.metric,
                month=args.month,
                model=args.model,
                input_dir=input_dir,
                output_dir=output_dir,
                color_scheme=args.color_scheme,
                formats=formats,
                dpi=args.dpi,
                figsize=figsize
            )
        else:
            # Generate single plot (unchanged)
            generate_metric_timeseries(
                variable=args.variable,
                experiment=args.experiment,
                month=args.month,
                metric=args.metric,
                model=args.model,
                input_dir=input_dir,
                output_dir=output_dir,
                color_scheme=args.color_scheme,
                formats=formats,
                dpi=args.dpi,
                figsize=figsize
            )

    elif args.viz_type == "heatmap":
        # Parse months argument for heatmap
        if args.months:
            try:
                months = parse_months_argument(args.months)
            except ValueError as e:
                logger.error(str(e))
                return
        else:
            months = list(range(1, 13))  # Default to all months

        generate_metric_heatmap(
            variable=args.variable,
            experiment=args.experiment,
            metric=args.metric,
            months=months,
            model=args.model,
            input_dir=input_dir,
            output_dir=output_dir,
            color_scheme=args.color_scheme,
            formats=formats,
            dpi=args.dpi,
            figsize=figsize
        )

    elif args.viz_type == "location":
        generate_location_plot(
            variable=args.variable,
            experiment=args.experiment,
            month=args.month,
            model=args.model,
            input_dir=input_dir,
            output_dir=output_dir,
            formats=formats,
            dpi=args.dpi,
            figsize=figsize
        )

    logger.info("Plot generation completed")


def generate_anomaly_plot(variable, experiment, month, model, input_dir, output_dir, formats, dpi, figsize):
    """Generate multi-scale anomaly decomposition plot."""
    logger.info(f"Generating anomaly plot for {variable}, {experiment}, month {month}, model {model}")

    # Create plotter
    plotter = TimeSeriesPlotter(
        variable=variable,
        experiment=experiment,
        month=month,
        model=model,
        input_dir=input_dir,
        output_dir=output_dir,
        dpi=dpi,
        figsize=figsize
    )

    # Generate plot
    fig = plotter.plot_anomaly_decomposition()

    # Export figure
    filename = f"{variable}_{model}_{experiment}_month{month:02d}_anomaly_decomposition"
    metadata = {
        "Variable": variable,
        "Experiment": experiment,
        "Month": str(month),
        "Model": model,
        "Type": "Anomaly Decomposition"
    }

    export_figure(
        fig=fig,
        output_dir=output_dir,
        filename=filename,
        formats=formats,
        dpi=dpi,
        metadata=None
    )

    # Close figure
    plotter.close()

    logger.info(f"Anomaly plot saved to {output_dir}/{filename}")


def generate_metric_timeseries(variable, experiment, month, metric, model, input_dir, output_dir,
                               color_scheme=None, formats=None, dpi=300, figsize=(12, 10)):
    """Generate time series plot for a specific metric."""
    logger.info(f"Generating time series plot for {metric}, {experiment}, month {month}, model {model}")

    # Create plotter
    plotter = ExtremeStatisticsPlotter(
        variable=variable,
        experiment=experiment,
        month=month,
        input_dir=input_dir,
        output_dir=output_dir,
        model=model,
        metric=metric,
        color_scheme=color_scheme,
        dpi=dpi,
        figsize=figsize
    )

    # Generate plot
    fig = plotter.plot_metric_time_series()

    # Export figure
    filename = f"{variable}_{model}_{experiment}_month{month:02d}_{metric}_timeseries"

    # Extract metric metadata if available
    plotter.load_data(data_type='extremes')
    if metric in plotter.data:
        metric_name = plotter.data[metric].attrs.get('long_name', metric)
    else:
        metric_name = metric

    metadata = {
        "Variable": variable,
        "Experiment": experiment,
        "Month": str(month),
        "Metric": metric_name,
        "Model": model,
        "Type": "Time Series"
    }

    export_figure(
        fig=fig,
        output_dir=output_dir,
        filename=filename,
        formats=formats,
        dpi=dpi,
        metadata=None
    )

    # Close figure
    plotter.close()

    logger.info(f"Time series plot saved to {output_dir}/{filename}")


def generate_metric_heatmap(variable, experiment, metric, months, model, input_dir, output_dir,
                            color_scheme=None, formats=None, dpi=300, figsize=(14, 8)):
    """Generate heatmap for a specific metric across multiple months."""
    logger.info(f"Generating heatmap for {metric}, {experiment}, months {months}, model {model}")

    # Create plotter with first month (will be updated for each month)
    plotter = ExtremeStatisticsPlotter(
        variable=variable,
        experiment=experiment,
        month=months[0],  # Temporary, will load data for all months
        input_dir=input_dir,
        output_dir=output_dir,
        model=model,
        metric=metric,
        color_scheme=color_scheme,
        dpi=dpi,
        figsize=figsize
    )

    # Generate plot
    fig = plotter.create_metric_heatmap(months=months)

    # Export figure
    month_str = "all" if len(months) == 12 else f"months{'_'.join([str(m) for m in months])}"
    filename = f"{variable}_{model}_{experiment}_{metric}_{month_str}_heatmap"

    # Extract metric metadata if available
    plotter.load_data(data_type='extremes')
    if metric in plotter.data:
        metric_name = plotter.data[metric].attrs.get('long_name', metric)
    else:
        metric_name = metric

    metadata = {
        "Variable": variable,
        "Experiment": experiment,
        "Months": ",".join(str(m) for m in months),
        "Metric": metric_name,
        "Model": model,
        "Type": "Heatmap"
    }

    export_figure(
        fig=fig,
        output_dir=output_dir,
        filename=filename,
        formats=formats,
        dpi=dpi,
        metadata=None
    )

    # Close figure
    plotter.close()

    logger.info(f"Heatmap saved to {output_dir}/{filename}")


def generate_location_plot(variable, experiment, month, model, input_dir, output_dir, formats, dpi, figsize):
    """Generate location map."""
    logger.info(f"Generating location map for {variable}, {experiment}, month {month}, model {model}")

    # Create plotter
    plotter = LocationMapPlotter(
        variable=variable,
        experiment=experiment,
        month=month,
        model=model,
        input_dir=input_dir,
        output_dir=output_dir,
        dpi=dpi,
        figsize=figsize
    )

    # Generate plot
    fig = plotter.plot_location_map()

    # Export figure
    filename = f"{variable}_{model}_{experiment}_location_map"

    # Get lat/lon from plotter
    metadata = {
        "Variable": variable,
        "Experiment": experiment,
        "Model": model,
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
        metadata=None
    )

    # Close figure
    plotter.close()

    logger.info(f"Location map saved to {output_dir}/{filename}")


if __name__ == "__main__":
    main()
