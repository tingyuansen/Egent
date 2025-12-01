"""EW Agent - Autonomous Equivalent Width measurement package."""

from .ew_tools import (
    load_spectrum,
    extract_region,
    set_continuum_method,
    fit_ew,
    direct_integration_ew,
    compare_with_catalog,
    flag_line,
    record_measurement,
    get_analysis_summary,
)

__all__ = [
    'load_spectrum',
    'extract_region',
    'set_continuum_method',
    'fit_ew',
    'direct_integration_ew',
    'compare_with_catalog',
    'flag_line',
    'record_measurement',
    'get_analysis_summary',
]

