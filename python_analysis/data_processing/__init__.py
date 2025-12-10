"""
Data Processing Package

This package provides utilities for loading, processing, and organizing
calcium imaging and behavioral data.
"""

# Import commonly used functions from learning_phase_mapping
from .learning_phase_mapping import (
    day_label_to_learning_phase,
    convert_session_labels_to_phases,
    get_ordered_phase_labels,
    convert_animal_data_to_phases
)

__all__ = [
    'day_label_to_learning_phase',
    'convert_session_labels_to_phases',
    'get_ordered_phase_labels',
    'convert_animal_data_to_phases',
]

