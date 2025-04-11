"""
Export Module

This module provides functions to export figures and data.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union, List

import matplotlib.pyplot as plt

# Configure logger
logger = logging.getLogger(__name__)


def export_figure(fig: plt.Figure,
                  output_dir: Union[str, Path],
                  filename: str,
                  formats: List[str] = None,
                  dpi: int = 300,
                  metadata: Optional[Dict[str, str]] = None,
                  close_after: bool = False) -> List[Path]:
    """
    Export a figure to disk in multiple formats.

    Args:
        fig: Matplotlib figure to export
        output_dir: Directory to save figure in
        filename: Base filename (without extension)
        formats: List of formats to save in
        dpi: Resolution for raster formats
        metadata: Metadata to include in the figure
        close_after: Whether to close the figure after saving

    Returns:
        List of saved file paths
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add metadata if provided
    if metadata:
        # Format metadata for inclusion in figure
        meta_string = ", ".join([f"{k}: {v}" for k, v in metadata.items()])
        fig.text(0.02, 0.01, meta_string, fontsize=6, alpha=0.7,
                 transform=fig.transFigure)

    saved_files = []

    # Use default formats if none provided
    if formats is None:
        formats = ['png', 'pdf']

    # Save in all requested formats
    for fmt in formats:
        output_path = output_dir / f"{filename}.{fmt}"
        logger.info(f"Saving figure to {output_path}")

        try:
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
                        metadata=metadata if fmt == 'pdf' else None)
            saved_files.append(output_path)
        except Exception as e:
            logger.error(f"Error saving figure to {output_path}: {e}")

    # Close figure if requested
    if close_after:
        plt.close(fig)

    return saved_files
