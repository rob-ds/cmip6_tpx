"""
Filters Module

This module provides implementations of digital filters for climate data analysis,
including Butterworth high-pass and low-pass filters.
"""

import logging
from typing import Union

import numpy as np
import xarray as xr
from scipy import signal

# Configure logger
logger = logging.getLogger(__name__)


def butterworth_filter(
        data: Union[np.ndarray, xr.DataArray],
        cutoff_freq: float,
        filter_type: str = 'lowpass',
        order: int = 4,
        sampling_freq: float = 1.0
) -> np.ndarray:
    """
    Apply a Butterworth filter to the data.

    Args:
        data: Input data to be filtered
        cutoff_freq: Cutoff frequency
        filter_type: Type of filter ('lowpass' or 'highpass')
        order: Filter order
        sampling_freq: Sampling frequency

    Returns:
        Filtered data
    """
    # Validate filter type
    if filter_type not in ['lowpass', 'highpass']:
        raise ValueError(f"Invalid filter type: {filter_type}. Must be 'lowpass' or 'highpass'")

    # Convert xarray DataArray to numpy array if needed
    if isinstance(data, xr.DataArray):
        data_values = data.values
    else:
        data_values = data

    # Create butter filter
    nyquist = 0.5 * sampling_freq
    normal_cutoff = cutoff_freq / nyquist

    # Use SOS (second-order sections) representation for numerical stability
    sos = signal.butter(order, normal_cutoff, btype=filter_type, analog=False, output='sos')

    # Apply filter to data using sosfiltfilt for zero-phase filtering
    filtered_data = signal.sosfiltfilt(sos, data_values)

    return filtered_data


def lowpass_filter(
        data: Union[np.ndarray, xr.DataArray],
        cutoff_period: float,
        order: int = 4,
        sampling_freq: float = 1.0
) -> np.ndarray:
    """
    Apply a low-pass Butterworth filter to the data.

    This filters out oscillations with periods shorter than the cutoff period.

    Args:
        data: Input data to be filtered
        cutoff_period: Cutoff period (in the same unit as the sampling rate)
        order: Filter order
        sampling_freq: Sampling frequency

    Returns:
        Filtered data
    """
    # Convert period to frequency
    cutoff_freq = 1.0 / cutoff_period

    logger.info(f"Applying low-pass filter with cutoff period {cutoff_period} years")

    return butterworth_filter(
        data=data,
        cutoff_freq=cutoff_freq,
        filter_type='lowpass',
        order=order,
        sampling_freq=sampling_freq
    )


def highpass_filter(
        data: Union[np.ndarray, xr.DataArray],
        cutoff_period: float,
        order: int = 4,
        sampling_freq: float = 1.0
) -> np.ndarray:
    """
    Apply a high-pass Butterworth filter to the data.

    This filters out oscillations with periods longer than the cutoff period.

    Args:
        data: Input data to be filtered
        cutoff_period: Cutoff period (in the same unit as the sampling rate)
        order: Filter order
        sampling_freq: Sampling frequency

    Returns:
        Filtered data
    """
    # Convert period to frequency
    cutoff_freq = 1.0 / cutoff_period

    logger.info(f"Applying high-pass filter with cutoff period {cutoff_period} years")

    return butterworth_filter(
        data=data,
        cutoff_freq=cutoff_freq,
        filter_type='highpass',
        order=order,
        sampling_freq=sampling_freq
    )
