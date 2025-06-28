"""
Evaluation module for specialized NLP processing.

This module contains evaluation tools and metrics for assessing
the performance of specialized components.
"""

from .insight_generator import InsightGenerator
from .metrics_generator import MetricsGenerator

__all__ = [
    'InsightGenerator',
    'MetricsGenerator'
]
