"""
Analysis Module

This module provides tools for analyzing CMIP6 climate data, including
anomaly calculation, filtering, and variability metrics.

Classes:
    BaseAnalyzer: Abstract base class for all analyzers
    AnomalyAnalyzer: Analyzer for calculating climate anomalies
    VariabilityAnalyzer: Analyzer for multi-scale climate variability

Functions:
    butterworth_filter: Apply a Butterworth filter to the data
    lowpass_filter: Apply a low-pass Butterworth filter to the data
    highpass_filter: Apply a high-pass Butterworth filter to the data
"""

from .base_analyzer import BaseAnalyzer
from .anomalies import AnomalyAnalyzer
from .filters import butterworth_filter, lowpass_filter, highpass_filter
from .variability import VariabilityAnalyzer

__all__ = [
    'BaseAnalyzer',
    'AnomalyAnalyzer',
    'butterworth_filter',
    'lowpass_filter',
    'highpass_filter',
    'VariabilityAnalyzer'
]
