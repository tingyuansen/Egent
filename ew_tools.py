"""
EW Agent Tools - Core Functions for Equivalent Width Measurement
=================================================================

This module provides the foundational tools for measuring stellar equivalent
widths (EW) from high-resolution spectra. It is designed to work with both
automated pipelines and LLM-based agentic systems.

ARCHITECTURE:
    The module uses a thread-local session pattern (`_get_session()`) to maintain
    state during multi-step fitting workflows. This allows:
    - Loading a spectrum once, then fitting multiple lines
    - Iterative refinement of continuum and line fits
    - Parallel processing with ThreadPoolExecutor (each thread has its own session)

KEY FUNCTIONS:
    1. load_spectrum(gaia_id)      - Load spectrum with barycentric correction
    2. extract_region(wavelength)  - Extract ±window Å around target line
    3. set_continuum_method()      - Configure continuum fitting algorithm
    4. set_continuum_regions()     - Manually specify continuum wavelengths
    5. fit_ew()                    - Fit multi-Voigt model to measure EW
    6. get_fit_plot()              - Generate diagnostic plot for visual inspection
    7. flag_line()                 - Mark line as unreliable (blend, no data, etc.)
    8. record_measurement()        - Record final EW measurement

FITTING APPROACH:
    1. Continuum Estimation: Iterative sigma-clipping or top percentile initialization
    2. Peak Finding: Locate absorption features using inverted flux peaks
    3. Multi-Voigt Fitting: Simultaneous fit of linear continuum + N Voigt profiles
    4. Target Matching: Identify fitted line closest to target wavelength
    5. EW Integration: Integrate (1 - F_norm) under matched Voigt profile

DIAGNOSTICS:
    The fit_ew() function returns rich diagnostics including:
    - fit_quality: 'good', 'acceptable', or 'poor'
    - quality_issues: List of specific problems detected
    - reduced_chi2: Chi-squared per degree of freedom
    - correlated_residuals: Metrics for blend/continuum detection
    - wavelength_dependent_residuals: Check for systematic continuum trends

Author: EW Agent Framework
"""

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import base64
import io
import json
import threading

# =============================================================================
# CONSTANTS
# =============================================================================
C_KMS = 299792.458  # Speed of light in km/s


# =============================================================================
# THREAD-LOCAL SESSION MANAGEMENT
# =============================================================================
# Each thread gets its own isolated session to enable parallel processing
# without race conditions. This is critical for ThreadPoolExecutor usage.

_thread_local = threading.local()


def _get_session():
    """
    Get the thread-local session dictionary, initializing if needed.
    
    The session stores all state for the current fitting workflow:
    - spectrum: Loaded pandas DataFrame
    - wave_col: Column name for wavelength (raw or helio-corrected)
    - current_region: Extracted wavelength region around target
    - last_fit: Results from most recent fit_ew() call
    - continuum_method: Algorithm for continuum fitting
    - etc.
    
    Returns:
        dict: Thread-local session state
    """
    if not hasattr(_thread_local, 'session'):
        _thread_local.session = {
            # Spectrum data
            'spectrum': None,           # Full spectrum DataFrame
            'wave_col': None,           # 'wavelength' or 'wavelength_helio'
            'gaia_id': None,            # Gaia source ID
            'bary_corr': 0.0,           # Applied barycentric correction (km/s)
            
            # Region and fit state
            'current_region': None,     # Extracted spectral region dict
            'last_fit': None,           # Most recent fit_ew() result
            'fit_history': [],          # History of all fits for this line
            'iteration_log': [],        # Log of parameter changes
            
            # Continuum configuration
            'continuum_method': 'iterative_linear',  # Fitting algorithm
            'continuum_order': 1,       # Polynomial order (if applicable)
            'sigma_clip': 2.5,          # Sigma threshold for clipping
            'window': 3.0,              # Default extraction window (Å)
            
            # Measurements and flags
            'flagged_lines': [],        # Lines marked as unreliable
            'measurements': [],         # Recorded EW measurements
            'current_line_iterations': [],  # Iterations for current line
        }
    return _thread_local.session


# =============================================================================
# CORE TOOLS
# =============================================================================

def load_spectrum(gaia_id: int, custom_file: str = None) -> dict:
    """
    Load a pre-corrected Magellan spectrum for analysis.
    
    Spectra must be pre-processed with preprocess_spectra.py to apply
    barycentric + empirical corrections. The 'wavelength' column is
    already in the rest frame.
    
    Parameters:
        gaia_id: Gaia source ID
        custom_file: Path to custom spectrum file (optional)
    
    Returns:
        Status and spectrum info
    """
    if custom_file:
        spec_file = Path(custom_file)
    else:
        spec_file = Path(__file__).parent / 'spectra_corrected' / f'{gaia_id}_magellan.csv'
    
    if not spec_file.exists():
        return {"success": False, "error": f"Spectrum not found: {gaia_id}. Run preprocess_spectra.py first."}
    
    spec = pd.read_csv(spec_file)
    
    _get_session()['spectrum'] = spec
    _get_session()['wave_col'] = 'wavelength'
    _get_session()['gaia_id'] = gaia_id
    
    return {
        "success": True,
        "gaia_id": gaia_id,
        "n_points": len(spec),
        "wavelength_range": [float(spec['wavelength'].min()), float(spec['wavelength'].max())],
        "median_snr": float(spec['snr'].median()),
    }


def extract_region(line_wavelength: float, window: float = None, echelle_order: str = None) -> dict:
    """
    Extract spectral region around a line for fitting.
    Uses single best echelle order by default, or specified order.
    
    Parameters:
        line_wavelength: Target wavelength in Angstroms
        window: Window size (default from session, typically 3.0 Å)
        echelle_order: Specific echelle order to use (default: best order with most points)
    
    Returns:
        Region info and diagnostic data
    """
    if _get_session()['spectrum'] is None:
        return {"success": False, "error": "No spectrum loaded"}
    
    spec = _get_session()['spectrum']
    wave_col = _get_session()['wave_col']
    window = window or _get_session()['window']
    
    mask = (spec[wave_col] >= line_wavelength - window) & \
           (spec[wave_col] <= line_wavelength + window)
    region = spec[mask]
    
    if len(region) == 0:
        return {"success": False, "error": f"No data in {line_wavelength-window:.1f}-{line_wavelength+window:.1f} Å"}
    
    # Get available echelle orders
    order_counts = region.groupby('echelle_order').size()
    available_orders = list(order_counts.index)
    
    # Select order
    if echelle_order is not None and echelle_order in available_orders:
        best_order = echelle_order
    else:
        best_order = order_counts.idxmax()
    
    region = region[region['echelle_order'] == best_order].copy()
    region = region.sort_values(wave_col)
    
    # Check if target wavelength is actually within the extracted data
    wave_arr = region[wave_col].values
    wave_min, wave_max = wave_arr.min(), wave_arr.max()
    target_in_data = wave_min <= line_wavelength <= wave_max
    
    if not target_in_data:
        return {
            "success": False, 
            "error": f"Target {line_wavelength:.2f} Å not in data range [{wave_min:.2f}, {wave_max:.2f}]",
            "flag_as": "no_data"
        }
    
    # Clear iteration log for new line
    if _get_session()['current_region'] is None or \
       abs(_get_session()['current_region'].get('target_wave', 0) - line_wavelength) > 0.5:
        _get_session()['current_line_iterations'] = []
    
    _get_session()['current_region'] = {
        'wave': region[wave_col].values,
        'flux': region['flux'].values,
        'flux_err': region['flux_error'].values,
        'snr': region['snr'].values,
        'order': best_order,
        'target_wave': line_wavelength,
        'window': window,
        'available_orders': available_orders,
    }
    _get_session()['window'] = window
    
    return {
        "success": True,
        "target_wavelength": line_wavelength,
        "window": window,
        "n_points": len(region),
        "echelle_order": str(best_order),
        "n_orders_available": len(available_orders),
        "available_orders": available_orders,
        "wavelength_coverage": [float(region[wave_col].min()), float(region[wave_col].max())],
        "median_snr": float(region['snr'].median()),
        "flux_range": [float(region['flux'].min()), float(region['flux'].max())],
    }


def set_continuum_method(method: str = 'iterative_linear', order: int = 1, sigma_clip: float = 2.5,
                         top_percentile: float = None) -> dict:
    """
    Configure continuum fitting method. Call this BEFORE fit_ew if you want to change defaults.
    
    Parameters:
        method: 'iterative_linear', 'iterative_poly', 'simple_linear', 'spline', 'manual_regions', 'top_percentile'
        order: Polynomial order (for poly method)
        sigma_clip: Sigma threshold for iterative clipping
        top_percentile: For 'top_percentile' method, use top X% of flux values (e.g., 85 means top 15%)
    
    Returns:
        Current continuum settings
    """
    valid_methods = ['iterative_linear', 'iterative_poly', 'simple_linear', 'spline', 'manual_regions', 'top_percentile']
    if method not in valid_methods:
        return {"success": False, "error": f"Invalid method. Choose from: {valid_methods}"}
    
    _get_session()['continuum_method'] = method
    _get_session()['continuum_order'] = order
    _get_session()['sigma_clip'] = sigma_clip
    _get_session()['top_percentile'] = top_percentile or 85  # Default: use top 15% of flux
    
    return {
        "success": True,
        "method": method,
        "order": order,
        "sigma_clip": sigma_clip,
        "top_percentile": _get_session()['top_percentile'],
        "message": f"Continuum will use {method} with order={order}, sigma_clip={sigma_clip}"
    }


def set_continuum_regions(regions: list) -> dict:
    """
    Manually specify wavelength ranges to use for continuum fitting.
    Use this when automatic continuum detection fails due to crowded line regions.
    Call this AFTER extract_region and BEFORE fit_ew.
    
    Parameters:
        regions: List of [start, end] wavelength pairs, e.g. [[5700, 5701], [5704, 5705]]
                These define regions where continuum pixels should be selected.
    
    Returns:
        Confirmation and info about selected pixels
    """
    if _get_session()['current_region'] is None:
        return {"success": False, "error": "No region extracted. Call extract_region first."}
    
    region = _get_session()['current_region']
    wave = region['wave']
    
    # Build mask for continuum pixels
    cont_mask = np.zeros(len(wave), dtype=bool)
    for r in regions:
        if len(r) == 2:
            mask = (wave >= r[0]) & (wave <= r[1])
            cont_mask |= mask
    
    n_selected = np.sum(cont_mask)
    
    if n_selected < 5:
        return {
            "success": False,
            "error": f"Only {n_selected} pixels selected. Need at least 5 for continuum fitting.",
            "suggestion": "Expand your wavelength regions or add more regions"
        }
    
    # Store manual regions in session
    _get_session()['continuum_regions'] = regions
    _get_session()['continuum_mask'] = cont_mask
    _get_session()['continuum_method'] = 'manual_regions'
    
    # Show what regions were selected
    selected_ranges = []
    for r in regions:
        n_in_range = np.sum((wave >= r[0]) & (wave <= r[1]))
        selected_ranges.append({
            "range": [float(r[0]), float(r[1])],
            "n_pixels": int(n_in_range)
        })
    
    return {
        "success": True,
        "n_continuum_pixels": int(n_selected),
        "selected_regions": selected_ranges,
        "message": f"Manually selected {n_selected} continuum pixels from {len(regions)} region(s). "
                   f"fit_ew will now use these pixels for continuum fitting."
    }


def fit_ew(min_peak_height: float = 0.02, min_prominence: float = 0.015, 
           expected_ew: float = None, additional_peaks: list = None) -> dict:
    """
    Fit equivalent width using multi-Voigt model with current continuum settings.
    Returns detailed diagnostics for agent reasoning.
    
    Parameters:
        min_peak_height: Minimum absorption depth for peak detection (default 0.02 = 2%)
        min_prominence: Minimum prominence for peak detection
        expected_ew: Expected EW in mÅ (hint) - if provided, will lower detection threshold 
                     for weak lines and warn if matched line differs significantly
        additional_peaks: List of wavelengths to add as extra Voigt components.
                         Use this when residuals show W-shaped patterns indicating
                         missed blends. The LLM can identify these from the plot.
    
    Returns:
        Detailed fit results and diagnostics for agent reasoning
    """
    # Auto-adjust detection threshold for expected weak lines
    if expected_ew is not None and expected_ew < 25:
        # For weak lines (<25 mÅ), use more sensitive detection
        min_peak_height = min(min_peak_height, 0.01)  # 1% depth threshold
        min_prominence = min(min_prominence, 0.008)
    if _get_session()['current_region'] is None:
        return {"success": False, "error": "No region extracted. Call extract_region first."}
    
    region = _get_session()['current_region']
    wave = region['wave']
    flux = region['flux']
    flux_err = region['flux_err']
    target_wave = region['target_wave']
    
    # Continuum fitting based on session settings
    method = _get_session()['continuum_method']
    order = _get_session()['continuum_order']
    sigma = _get_session()['sigma_clip']
    top_pct = _get_session().get('top_percentile', 85)
    
    cont_mask = np.ones(len(flux), dtype=bool)
    
    if method == 'manual_regions' and _get_session().get('continuum_mask') is not None:
        # Use manually specified continuum regions
        cont_mask = _get_session()['continuum_mask']
        if len(cont_mask) != len(wave):
            # Regenerate mask if region changed
            regions = _get_session().get('continuum_regions', [])
            cont_mask = np.zeros(len(wave), dtype=bool)
            for r in regions:
                if len(r) == 2:
                    cont_mask |= (wave >= r[0]) & (wave <= r[1])
        
        # Fallback to top_percentile if manual regions have too few points
        if np.sum(cont_mask) < 5:
            threshold = np.percentile(flux, top_pct)
            cont_mask = flux >= threshold
            poly_order = 1
            coef = np.polyfit(wave[cont_mask], flux[cont_mask], poly_order)
            continuum = np.polyval(coef, wave)
            continuum_info = {'method': 'fallback_top_percentile', 'iterations': 1, 
                            'points_used': int(np.sum(cont_mask)), 'percentile': top_pct,
                            'note': 'Manual regions had too few points, fell back to top percentile'}
        else:
            poly_order = 1  # Always use linear for manual regions
            coef = np.polyfit(wave[cont_mask], flux[cont_mask], poly_order)
            continuum = np.polyval(coef, wave)
            continuum_info = {'method': 'manual_regions', 'iterations': 1, 'points_used': int(np.sum(cont_mask)),
                            'regions': _get_session().get('continuum_regions', [])}
    
    elif method == 'top_percentile':
        # Use top percentile of flux values as continuum points
        threshold = np.percentile(flux, top_pct)
        cont_mask = flux >= threshold
        poly_order = 1
        coef = np.polyfit(wave[cont_mask], flux[cont_mask], poly_order)
        continuum = np.polyval(coef, wave)
        continuum_info = {'method': 'top_percentile', 'iterations': 1, 
                         'points_used': int(np.sum(cont_mask)), 'percentile': top_pct,
                         'threshold_flux': float(threshold)}
    
    elif method in ['iterative_linear', 'iterative_poly']:
        poly_order = 1 if method == 'iterative_linear' else order
        
        # IMPROVED: Initialize from top percentile, but ensure points from both ends
        threshold = np.percentile(flux, top_pct)
        cont_mask = flux >= threshold
        
        # Ensure we have points from left, middle, and right thirds of the region
        n = len(wave)
        third = n // 3
        left_mask = np.zeros(n, dtype=bool)
        left_mask[:third] = True
        right_mask = np.zeros(n, dtype=bool)
        right_mask[-third:] = True
        mid_mask = np.zeros(n, dtype=bool)
        mid_mask[third:-third] = True
        
        # If any section has no continuum points, add top points from that section
        for section_mask, section_name in [(left_mask, 'left'), (right_mask, 'right'), (mid_mask, 'mid')]:
            section_cont = cont_mask & section_mask
            if np.sum(section_cont) < 2:
                # Add top 20% of this section
                section_flux = flux.copy()
                section_flux[~section_mask] = -np.inf
                section_thresh = np.percentile(section_flux[section_mask], 80)
                cont_mask |= (section_mask & (flux >= section_thresh))
        
        # Then iterate with sigma clipping from this better starting point
        for iteration in range(5):
            if np.sum(cont_mask) < 5:
                # If too few points, reset
                cont_mask = flux >= threshold
                break
            coef = np.polyfit(wave[cont_mask], flux[cont_mask], poly_order)
            continuum = np.polyval(coef, wave)
            residuals = flux - continuum
            std = np.std(residuals[cont_mask])
            # Only reject points significantly BELOW continuum (absorption lines)
            new_mask = cont_mask & (residuals > -sigma * std)
            if np.sum(new_mask) == np.sum(cont_mask) or np.sum(new_mask) < 5:
                break
            cont_mask = new_mask
        
        continuum_info = {'method': method, 'iterations': iteration + 1, 
                         'points_used': int(np.sum(cont_mask)), 'init_percentile': top_pct,
                         'slope': float(coef[-2]) if len(coef) > 1 else 0}
    else:
        # Simple linear - also use top percentile for better initialization
        threshold = np.percentile(flux, top_pct)
        cont_mask = flux >= threshold
        coef = np.polyfit(wave[cont_mask], flux[cont_mask], 1)
        continuum = np.polyval(coef, wave)
        continuum_info = {'method': 'simple_linear_top', 'iterations': 1, 
                         'points_used': int(np.sum(cont_mask)), 'percentile': top_pct}
    
    # Check continuum quality
    cont_residuals = flux[cont_mask] - np.polyval(coef, wave[cont_mask])
    continuum_rms = float(np.std(cont_residuals))
    continuum_slope = float(coef[-2]) if len(coef) > 1 else 0.0
    
    # Store initial continuum coefficients for exact reconstruction
    # coef is [c1, c0] from np.polyfit (highest order first)
    # Store as [c0, c1, ...] (lowest order first) to match multi_voigt format
    continuum_info['init_continuum_coef'] = coef[::-1].tolist()  # Reverse to get [c0, c1, ...]
    continuum_info['wave_center'] = float(np.mean(wave))  # For centering in reconstruction
    
    # Normalize
    flux_norm = flux / continuum
    flux_norm_err = flux_err / continuum
    
    # Peak finding
    flux_inv = 1 - flux_norm
    peaks, peak_props = find_peaks(flux_inv, height=min_peak_height, distance=4, prominence=min_prominence)
    
    # Add manually specified peaks (for blends identified by LLM from W-shaped residuals)
    additional_peak_indices = []
    if additional_peaks:
        for add_wave in additional_peaks:
            # Find closest index to specified wavelength
            idx = np.argmin(np.abs(wave - add_wave))
            # Only add if not already detected and within the data range
            if idx not in peaks and 0 < idx < len(wave) - 1:
                additional_peak_indices.append(idx)
        
        if additional_peak_indices:
            peaks = np.concatenate([peaks, np.array(additional_peak_indices)])
            peaks = np.sort(peaks)
    
    if len(peaks) == 0:
        return {
            "success": False,
            "error": "No absorption peaks found",
            "diagnostics": {
                "continuum": continuum_info,
                "continuum_rms": continuum_rms,
                "flux_norm_range": [float(flux_norm.min()), float(flux_norm.max())],
                "suggestion": "Try lowering min_peak_height or check if line is too weak"
            }
        }
    
    # Get continuum order from session (for polynomial support)
    cont_order = _get_session().get('continuum_order', 1)
    n_cont_params = cont_order + 1  # order 1 = 2 params, order 2 = 3 params, etc.
    
    # Multi-Voigt model with polynomial continuum
    def multi_voigt(wavelength, *params):
        # First n_cont_params are continuum coefficients
        cont_coeffs = params[:n_cont_params]
        w_centered = wavelength - wavelength.mean()
        
        # Polynomial continuum: c0 + c1*x + c2*x^2 + ...
        cont = np.zeros_like(wavelength)
        for i, c in enumerate(cont_coeffs):
            cont += c * (w_centered ** i)
        
        result = cont.copy()
        n_lines = (len(params) - n_cont_params) // 4
        for i in range(n_lines):
            idx = n_cont_params + i * 4
            amp, center, sig, gam = params[idx:idx+4]
            v = voigt_profile(wavelength - center, sig, gam)
            if v.max() > 0:
                v = v / v.max()
            result -= amp * v
        return result
    
    # Setup fit parameters based on continuum order
    n_lines = len(peaks)
    
    # Continuum initial guesses and bounds
    p0 = [1.0] + [0.0] * cont_order  # [1.0, 0.0] for linear, [1.0, 0.0, 0.0] for quadratic
    lower = [0.9] + [-0.05] * cont_order
    upper = [1.1] + [0.05] * cont_order
    
    detected_peaks = []
    for p in peaks:
        depth = max(min_peak_height, flux_inv[p])
        detected_peaks.append({'index': int(p), 'wavelength': float(wave[p]), 'depth': float(depth)})
        p0.extend([depth, wave[p], 0.04, 0.015])
        # Tighter gamma bounds: 0.001-0.08 based on empirical analysis
        # 95th percentile gamma is ~0.1, but most lines are << 0.08
        lower.extend([0.005, wave[p]-0.5, 0.003, 0.001])
        upper.extend([0.9, wave[p]+0.5, 0.25, 0.08])
    
    try:
        popt, pcov = curve_fit(multi_voigt, wave, flux_norm, p0=p0,
                               sigma=flux_norm_err, absolute_sigma=True,
                               bounds=(lower, upper), maxfev=20000)
        fit_success = True
        
        # =================================================================
        # CENTER DISTANCE VALIDATION: Merge lines closer than 0.2 Å
        # Two very close lines likely represent overfitting of a single feature
        # =================================================================
        MIN_CENTER_DISTANCE = 0.2  # Angstroms
        
        # Extract fitted centers
        fitted_centers = []
        for i in range(n_lines):
            idx = n_cont_params + i * 4 + 1  # center is at position 1 within each line's 4 params
            fitted_centers.append(popt[idx])
        
        # Check for pairs too close together
        centers_to_merge = []
        for i in range(len(fitted_centers)):
            for j in range(i+1, len(fitted_centers)):
                if abs(fitted_centers[i] - fitted_centers[j]) < MIN_CENTER_DISTANCE:
                    centers_to_merge.append((i, j))
        
        # If we found lines too close together, refit with merged peaks
        if centers_to_merge:
            # Keep only one peak from each pair (the one closer to target)
            peaks_to_remove = set()
            for i, j in centers_to_merge:
                # Keep the one closer to target
                dist_i = abs(fitted_centers[i] - target_wave)
                dist_j = abs(fitted_centers[j] - target_wave)
                if dist_i < dist_j:
                    peaks_to_remove.add(j)
                else:
                    peaks_to_remove.add(i)
            
            # Rebuild with fewer peaks
            new_peaks = [peaks[i] for i in range(len(peaks)) if i not in peaks_to_remove]
            if len(new_peaks) > 0:
                # Rebuild p0, lower, upper
                p0 = [1.0] + [0.0] * cont_order
                lower = [0.9] + [-0.05] * cont_order
                upper = [1.1] + [0.05] * cont_order
                
                for p in new_peaks:
                    depth = max(min_peak_height, flux_inv[p])
                    p0.extend([depth, wave[p], 0.04, 0.015])
                    lower.extend([0.005, wave[p]-0.5, 0.003, 0.001])
                    upper.extend([0.9, wave[p]+0.5, 0.25, 0.08])
                
                # Refit
                popt, pcov = curve_fit(multi_voigt, wave, flux_norm, p0=p0,
                                      sigma=flux_norm_err, absolute_sigma=True,
                                      bounds=(lower, upper), maxfev=20000)
                peaks = new_peaks
                n_lines = len(peaks)
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Curve fit failed: {str(e)}",
            "diagnostics": {
                "continuum": continuum_info,
                "n_peaks_found": n_lines,
                "detected_peaks": detected_peaks,
                "suggestion": "Fit failed - try adjusting window or continuum method"
            }
        }
    
    # Calculate fit quality metrics
    flux_fit = multi_voigt(wave, *popt)
    residuals = flux_norm - flux_fit
    
    # RMS normalized by flux uncertainty (more meaningful for varying SNR)
    # This is the square root of reduced chi-squared, essentially
    if np.any(flux_norm_err > 0) and np.median(flux_norm_err) > 0:
        normalized_residuals = residuals / flux_norm_err
        fit_rms = float(np.std(normalized_residuals))  # Should be ~1 for a good fit
    else:
        # Fallback: use raw RMS (for backward compatibility)
        fit_rms = float(np.std(residuals))
    
    # CENTRAL REGION RMS - focus on ±1.5Å around target (most important for EW accuracy)
    # This avoids penalizing fits for edge effects that don't affect the measurement
    central_mask = np.abs(wave - target_wave) < 1.5
    if np.sum(central_mask) > 10:
        central_residuals = residuals[central_mask]
        central_rms_raw = float(np.std(central_residuals))
        if np.any(flux_norm_err > 0) and np.median(flux_norm_err[central_mask]) > 0:
            central_norm_resid = central_residuals / flux_norm_err[central_mask]
            central_rms = float(np.std(central_norm_resid))
        else:
            central_rms = central_rms_raw / 0.01  # Fallback
    else:
        central_rms = fit_rms  # Fallback to full RMS
        central_rms_raw = float(np.std(residuals))
    
    max_residual = float(np.max(np.abs(residuals)))
    
    # Check for wavelength-dependent residuals (signals continuum problem!)
    # Fit a line to residuals vs wavelength
    resid_slope, resid_intercept = np.polyfit(wave, residuals, 1)
    resid_trend = np.polyval([resid_slope, resid_intercept], wave)
    resid_trend_strength = np.std(resid_trend)  # How much trend explains variance
    
    # NORMALIZED residual slope (σ/Å) - more interpretable
    if np.any(flux_norm_err > 0) and np.median(flux_norm_err) > 0:
        # Fit line to normalized residuals
        w_centered = wave - np.mean(wave)
        norm_resid_temp = residuals / flux_norm_err
        resid_slope_norm, _ = np.polyfit(w_centered, norm_resid_temp, 1)
    else:
        resid_slope_norm = resid_slope / 0.01  # Fallback
    
    # Chi-squared calculation (same as before)
    chi2 = np.sum((residuals / flux_norm_err)**2) if np.any(flux_norm_err > 0) else np.sum(residuals**2) / 0.01**2
    dof = len(residuals) - len(popt)
    reduced_chi2 = chi2 / dof if dof > 0 else chi2
    
    # Correlated residuals detection - normalized by uncertainty
    # Look for consecutive points > 2σ which signals unmodeled structure (blend/bad continuum)
    if np.any(flux_norm_err > 0):
        norm_resid = residuals / flux_norm_err  # Normalized residuals
    else:
        norm_resid = residuals / 0.01  # Fallback
    
    # Count runs of consecutive points with |norm_resid| > 2
    high_resid = np.abs(norm_resid) > 2
    max_consecutive = 0
    current_run = 0
    n_high_resid_runs = 0  # Count of runs with 3+ consecutive points
    
    for hr in high_resid:
        if hr:
            current_run += 1
            max_consecutive = max(max_consecutive, current_run)
        else:
            if current_run >= 3:
                n_high_resid_runs += 1
            current_run = 0
    if current_run >= 3:
        n_high_resid_runs += 1
    
    # Also check for systematic positive or negative runs (continuum offset)
    positive_run = 0
    negative_run = 0
    max_pos_run = 0
    max_neg_run = 0
    for nr in norm_resid:
        if nr > 1:
            positive_run += 1
            max_pos_run = max(max_pos_run, positive_run)
            negative_run = 0
        elif nr < -1:
            negative_run += 1
            max_neg_run = max(max_neg_run, negative_run)
            positive_run = 0
        else:
            positive_run = 0
            negative_run = 0
    
    correlated_resid_info = {
        'max_consecutive_2sigma': int(max_consecutive),
        'n_runs_3plus': int(n_high_resid_runs),
        'max_positive_run': int(max_pos_run),
        'max_negative_run': int(max_neg_run),
    }
    
    # Extract fitted continuum coefficients (important for exact reproduction!)
    fitted_cont_coeffs = popt[:n_cont_params].tolist()
    continuum_info['fitted_continuum_coeffs'] = fitted_cont_coeffs
    
    # Extract all fitted lines (n_cont_params continuum coeffs, then 4 params per Voigt)
    fitted_lines = []
    
    for i in range(n_lines):
        idx = n_cont_params + i * 4
        amp, center, sig, gam = popt[idx:idx+4]
        
        # Calculate EW
        fwhm = 2.355 * sig
        half_range = max(0.5, 2.5 * fwhm)
        wave_fine = np.linspace(center - half_range, center + half_range, 2000)
        v = voigt_profile(wave_fine - center, sig, gam)
        v = v / v.max() * amp
        ew = trapezoid(v, wave_fine) * 1000  # mÅ
        
        # EW uncertainty
        try:
            perr = np.sqrt(np.diag(pcov))
            rel_err = np.sqrt((perr[idx]/amp)**2 + (perr[idx+2]/sig)**2)
            ew_err = ew * min(rel_err, 0.5)
        except:
            ew_err = ew * 0.15
        
        line_info = {
            'center': float(center),
            'amplitude': float(amp),
            'sigma': float(sig),
            'gamma': float(gam),
            'fwhm': float(fwhm),
            'ew_mA': float(ew),
            'ew_err_mA': float(ew_err),
            'distance_from_target': float(abs(center - target_wave)),
        }
        fitted_lines.append(line_info)
    
    # Select target line: prioritize CLOSEST, but prefer deeper if essentially same position
    target_line = None
    candidates = [l for l in fitted_lines if l['distance_from_target'] < 0.5]  # Within 0.5Å
    
    if candidates:
        # First, find the closest candidate
        closest = min(candidates, key=lambda x: x['distance_from_target'])
        
        # Only prefer a deeper line if it's essentially at the same position (within 0.1Å)
        # This handles cases where a blend has two components at nearly the same wavelength
        very_close = [l for l in candidates if l['distance_from_target'] < closest['distance_from_target'] + 0.1]
        
        if len(very_close) > 1:
            # Multiple lines at essentially the same position - pick the deeper one
            target_line = max(very_close, key=lambda x: x['ew_mA'])
        else:
            # Just one candidate that's close - use the closest
            target_line = closest
    else:
        # Fallback: just pick closest
        if fitted_lines:
            target_line = min(fitted_lines, key=lambda x: x['distance_from_target'])
    
    # Store fit for later use (including diagnostics for plotting)
    _get_session()['last_fit'] = {
        'popt': popt.tolist(),
        'pcov': pcov.tolist(),
        'wave': wave.tolist(),
        'flux_norm': flux_norm.tolist(),
        'flux_norm_err': flux_norm_err.tolist(),  # For normalized RMS in plots
        'flux_fit': flux_fit.tolist(),
        'residuals': residuals.tolist(),
        'continuum': continuum.tolist(),
        'target_line': target_line,
        'all_lines': fitted_lines,
        'fit_rms': fit_rms,  # Normalized RMS (should be ~1 for good fit)
        'central_rms': central_rms,  # Normalized RMS in ±1.5Å around target
    }
    _get_session()['fit_history'].append({
        'target_wave': target_wave,
        'ew': target_line['ew_mA'] if target_line else None,
        'rms': fit_rms,
        'method': method,
    })
    
    # Quality assessment for agent reasoning
    quality_issues = []
    quality_warnings = []
    
    # Calculate minimum detectable EW based on SNR (physical constraint, no catalog needed)
    median_snr = np.median(region['snr']) if region.get('snr') is not None else 100
    # Detection limit: ~3σ in depth × typical line width (0.1 Å) × 1000 mÅ/Å
    min_detectable_ew = (3.0 / median_snr) * 0.1 * 1000  # in mÅ
    
    if target_line is None:
        quality_issues.append("No line found near target wavelength")
        quality_issues.append(f"Minimum detectable EW at this SNR ({median_snr:.0f}): ~{min_detectable_ew:.1f} mÅ")
    else:
        # Check wavelength offset - be lenient since heliocentric correction is applied
        offset = target_line['distance_from_target']
        if offset > 0.30:  # 300 mÅ threshold - very large offset
            quality_issues.append(f"Large offset: Matched line is {offset*1000:.0f} mÅ away from target")
        elif offset > 0.15:  # 150 mÅ warning
            quality_warnings.append(f"Moderate offset from target: {offset*1000:.0f} mÅ")
        
        # Check if any line exists closer to target (might be undetected weak line)
        if offset > 0.15:
            # List all fitted lines sorted by distance to target
            lines_by_dist = sorted(fitted_lines, key=lambda x: x['distance_from_target'])
            if len(lines_by_dist) > 1:
                closest = lines_by_dist[0]
                second = lines_by_dist[1]
                quality_warnings.append(
                    f"Closest line: {closest['center']:.2f}Å (EW={closest['ew_mA']:.1f}mÅ), "
                    f"Next: {second['center']:.2f}Å (EW={second['ew_mA']:.1f}mÅ)"
                )
        
        # Detection limit warning (physical, no catalog)
        if target_line['ew_mA'] < min_detectable_ew:
            quality_warnings.append(f"Warning: Measured EW ({target_line['ew_mA']:.1f} mÅ) is near detection limit ({min_detectable_ew:.1f} mÅ)")
        
        # NOTE: We do NOT use expected_ew for flagging decisions - that would be cheating!
        # Flagging must be based purely on fit diagnostics (chi2, residuals, etc.)
        # expected_ew is only used to adjust detection threshold for very weak lines
    
    # RMS is now normalized by flux uncertainty (should be ~1.0 for a good fit)
    # RMS > 2.0 means residuals are 2x larger than expected from noise
    if fit_rms > 3.0:
        quality_issues.append(f"High normalized RMS: {fit_rms:.2f} (expect ~1.0)")
    elif fit_rms > 2.0:
        quality_warnings.append(f"Elevated normalized RMS: {fit_rms:.2f} (expect ~1.0)")
    
    # CRITICAL: Check for wavelength-dependent residuals (signals BAD CONTINUUM!)
    if resid_trend_strength > 0.015:
        quality_issues.append(f"WAVELENGTH-DEPENDENT RESIDUALS: Trend strength {resid_trend_strength:.4f}")
        quality_issues.append("This indicates the continuum slope is WRONG - try different continuum method!")
    elif resid_trend_strength > 0.008:
        quality_warnings.append(f"Slight wavelength trend in residuals: {resid_trend_strength:.4f}")
    
    # Check reduced chi-squared
    if reduced_chi2 > 5:
        quality_issues.append(f"VERY HIGH χ²/dof = {reduced_chi2:.1f} - fit is poor, consider NUCLEAR OPTION")
    elif reduced_chi2 > 2:
        quality_warnings.append(f"Elevated χ²/dof = {reduced_chi2:.1f}")
    
    # Check for correlated residuals (signal unmodeled structure: blend or bad continuum)
    if correlated_resid_info['max_consecutive_2sigma'] >= 5:
        quality_issues.append(f"CORRELATED RESIDUALS: {correlated_resid_info['max_consecutive_2sigma']} consecutive >2σ points - likely BLEND or BAD CONTINUUM")
    elif correlated_resid_info['max_consecutive_2sigma'] >= 3:
        quality_warnings.append(f"Correlated residuals: {correlated_resid_info['max_consecutive_2sigma']} consecutive >2σ points")
    
    if correlated_resid_info['n_runs_3plus'] >= 2:
        quality_issues.append(f"MULTIPLE UNMODELED FEATURES: {correlated_resid_info['n_runs_3plus']} runs of >2σ residuals")
    
    # Check for systematic offset (continuum too high or low)
    if correlated_resid_info['max_positive_run'] >= 8 or correlated_resid_info['max_negative_run'] >= 8:
        quality_issues.append(f"SYSTEMATIC CONTINUUM OFFSET: {max(correlated_resid_info['max_positive_run'], correlated_resid_info['max_negative_run'])} consecutive same-sign residuals")
    
    if abs(continuum_slope) > 0.01:
        quality_warnings.append(f"Steep continuum slope: {continuum_slope:.4f}")
    
    if n_lines > 5:
        quality_warnings.append(f"Many lines ({n_lines}) in window - consider narrower window")
    
    # Check for potential line confusion (alternative lines nearby)
    alternative_lines = []
    if target_line:
        for line in fitted_lines:
            if line == target_line:
                continue
            dist = abs(line['center'] - target_wave)
            if dist < 1.5:  # Within 1.5 Å of target
                alternative_lines.append({
                    'center': line['center'],
                    'ew_mA': line['ew_mA'],
                    'distance': dist
                })
        
        if alternative_lines:
            # Sort by distance
            alternative_lines.sort(key=lambda x: x['distance'])
            quality_warnings.append(
                f"Alternative lines within 1.5Å: {len(alternative_lines)} "
                f"(closest: {alternative_lines[0]['center']:.2f}Å, EW={alternative_lines[0]['ew_mA']:.1f}mÅ)"
            )
    
    quality = "poor" if quality_issues else ("acceptable" if quality_warnings else "good")
    
    # Log this iteration for analysis
    iteration_record = {
        'target_wavelength': target_wave,
        'iteration': len(_get_session()['current_line_iterations']) + 1,
        'method': 'voigt_fit',
        'continuum_method': _get_session()['continuum_method'],
        'continuum_order': _get_session()['continuum_order'],
        'window': _get_session()['window'],
        'echelle_order': region['order'],
        'ew_mA': target_line['ew_mA'] if target_line else None,
        'ew_err_mA': target_line['ew_err_mA'] if target_line else None,
        'fit_quality': quality,
        'fit_rms': fit_rms,
        'n_lines_fitted': n_lines,
        'issues': quality_issues,
        'warnings': quality_warnings,
    }
    _get_session()['current_line_iterations'].append(iteration_record)
    _get_session()['iteration_log'].append(iteration_record)
    
    # Build region_info for exact plot reconstruction
    region_info = {
        'echelle_order': region['order'],
        'window': region.get('window', 3.0),
        'wave_range': [float(wave[0]), float(wave[-1])],
    }
    
    return {
        "success": True,
        "target_wavelength": target_wave,
        "target_line": target_line,
        "fit_quality": quality,
        "fit_rms": fit_rms,
        "reduced_chi2": reduced_chi2,
        "quality_issues": quality_issues,
        "quality_warnings": quality_warnings,
        "iteration_number": len(_get_session()['current_line_iterations']),
        "region_info": region_info,       # For plot reconstruction
        "continuum_info": continuum_info, # For plot reconstruction
        "diagnostics": {
            "continuum": continuum_info,
            "continuum_rms": continuum_rms,
            "continuum_slope": continuum_slope,
            "fit_rms": fit_rms,
            "central_rms": central_rms,  # RMS in ±1.5Å around target (most important)
            "central_rms_raw": central_rms_raw,  # Raw RMS in central region
            "max_residual": max_residual,
            "reduced_chi2": reduced_chi2,
            "resid_wavelength_trend": resid_trend_strength,
            "resid_slope_norm": float(resid_slope_norm),  # σ/Å - normalized slope
            "correlated_residuals": correlated_resid_info,
            "n_lines_fitted": n_lines,
            "all_fitted_lines": fitted_lines,
            "alternative_lines_nearby": alternative_lines,
        },
        "suggestions": _generate_suggestions(quality_issues, quality_warnings, fit_rms, n_lines, target_line, 
                                            alternative_lines, reduced_chi2, resid_trend_strength,
                                            resid_slope_norm=resid_slope_norm,
                                            wave_range=[wave.min(), wave.max()] if len(wave) > 0 else None)
    }


def _generate_suggestions(issues, warnings, fit_rms, n_lines, target_line, alternative_lines=None,
                          reduced_chi2=None, resid_trend=None, resid_slope_norm=None, wave_range=None):
    """Generate actionable suggestions for agent with tiered approach."""
    suggestions = []
    
    # TIER 1: Simple fixes
    if "High residual RMS" in str(issues):
        suggestions.append("TIER 1: Try set_continuum_method('iterative_poly', order=2) for curved continuum")
    
    if n_lines > 4:
        suggestions.append(f"TIER 1: Window has {n_lines} lines - try extract_region with window=2.0")
    
    # TIER 2: Manual intervention
    if n_lines > 6:
        suggestions.append("TIER 2: CROWDED - use set_continuum_regions with line-free ranges from plot")
    
    if target_line and target_line['distance_from_target'] > 0.12:
        suggestions.append("TIER 2: Large offset - check plot if feature exists at target wavelength")
    
    # TIER 3: HIGH RESIDUAL SLOPE - USE MANUAL CONTINUUM REGIONS
    # Generate SPECIFIC suggested wavelength ranges based on current window
    if resid_slope_norm is not None and abs(resid_slope_norm) > 1.0 and wave_range:
        w_min, w_max = wave_range
        w_mid = (w_min + w_max) / 2
        # Suggest two regions: one in first quarter, one in last quarter of window
        region1 = [round(w_min + 0.15 * (w_max - w_min), 2), round(w_min + 0.25 * (w_max - w_min), 2)]
        region2 = [round(w_min + 0.75 * (w_max - w_min), 2), round(w_min + 0.85 * (w_max - w_min), 2)]
        
        suggestions.insert(0, f"⚠️ CRITICAL: Residual slope = {resid_slope_norm:.2f} σ/Å - MUST use set_continuum_regions!")
        suggestions.insert(1, f"   → COPY THIS: set_continuum_regions([[{region1[0]}, {region1[1]}], [{region2[0]}, {region2[1]}]])")
        suggestions.insert(2, "   → Adjust ranges if needed: look for FLAT y~1.0 regions in plot, avoiding absorptions")
    elif resid_slope_norm is not None and abs(resid_slope_norm) > 1.0:
        suggestions.insert(0, f"⚠️ CRITICAL: Residual slope = {resid_slope_norm:.2f} σ/Å!")
        suggestions.insert(1, "   → USE set_continuum_regions to specify clean continuum pixels")
    elif resid_slope_norm is not None and abs(resid_slope_norm) > 0.5:
        suggestions.append(f"TIER 2: Residual slope = {resid_slope_norm:.2f} σ/Å - try narrower window or manual continuum")
    
    # Legacy check
    if resid_trend is not None and resid_trend > 0.015:
        suggestions.append("⚠️ NUCLEAR: Wavelength-dependent residuals! Continuum slope is WRONG.")
        suggestions.append("   Look at plot, find flat continuum regions, use set_continuum_regions")
    
    if reduced_chi2 is not None and reduced_chi2 > 5:
        suggestions.append(f"⚠️ NUCLEAR: χ²/dof={reduced_chi2:.1f} is very high!")
        suggestions.append("   Try: different window, set_continuum_method('top_percentile', top_percentile=90)")
    
    # Line confusion
    if alternative_lines and len(alternative_lines) > 0:
        alt = alternative_lines[0]
        if alt['distance'] < 0.5:
            suggestions.append(f"Alternative line at {alt['center']:.2f}Å ({alt['ew_mA']:.1f}mÅ) nearby")
    
    if not suggestions:
        suggestions.append("Fit looks good - proceed with measurement")
    
    return suggestions


def get_fit_plot(format: str = 'base64') -> dict:
    """
    Generate a diagnostic plot of the current fit.
    Returns base64 image that agent can view for reasoning.
    
    Parameters:
        format: 'base64' (default) or 'save' (saves to file)
    
    Returns:
        Plot as base64 string or file path
    """
    if _get_session()['last_fit'] is None:
        return {"success": False, "error": "No fit available. Run fit_ew first."}
    
    fit = _get_session()['last_fit']
    wave = np.array(fit['wave'])
    flux_norm = np.array(fit['flux_norm'])
    flux_fit = np.array(fit['flux_fit'])
    residuals = np.array(fit['residuals'])
    target_line = fit['target_line']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[3, 1], sharex=True)
    
    # Main plot
    ax1.plot(wave, flux_norm, 'k-', lw=0.8, alpha=0.7, label='Data')
    ax1.plot(wave, flux_fit, 'r-', lw=1.5, label='Fit')
    ax1.axhline(1, color='gray', ls=':', alpha=0.5)
    
    # Mark all fitted lines
    for line in fit['all_lines']:
        color = 'green' if line == target_line else 'orange'
        ax1.axvline(line['center'], color=color, ls='--', alpha=0.5, lw=1)
        ax1.text(line['center'], 0.45, f"{line['ew_mA']:.1f}", fontsize=8, 
                ha='center', color=color)
    
    # Mark target
    target_wave = _get_session()['current_region']['target_wave']
    ax1.axvline(target_wave, color='blue', ls=':', alpha=0.7, lw=2, label=f'Target: {target_wave:.2f} Å')
    
    ax1.set_ylabel('Normalized Flux')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_ylim(0.4, 1.15)
    
    if target_line:
        ax1.set_title(f'Target: {target_wave:.2f} Å | EW={target_line["ew_mA"]:.1f}±{target_line["ew_err_mA"]:.1f} mÅ | '
                     f'Offset={target_line["distance_from_target"]*1000:.1f} mÅ')
    
    # Get flux errors from session for normalized residuals
    flux_norm_err = fit.get('flux_norm_err', None)
    if flux_norm_err is not None and len(flux_norm_err) == len(residuals):
        flux_norm_err = np.array(flux_norm_err)
        if np.any(flux_norm_err > 0) and np.median(flux_norm_err) > 0:
            residuals_norm = residuals / flux_norm_err
        else:
            residuals_norm = residuals / 0.01  # Fallback
    else:
        residuals_norm = residuals / 0.01  # Fallback
    
    # Normalized residuals with sigma bands
    ax2.axhspan(-1, 1, alpha=0.2, color='lightgreen', zorder=1, label='±1σ')
    ax2.axhspan(-2, 2, alpha=0.1, color='lightyellow', zorder=1, label='±2σ')
    ax2.plot(wave, residuals_norm, 'k-', lw=0.8, zorder=3)
    ax2.axhline(0, color='gray', ls='-', alpha=0.5, zorder=2)
    ax2.axhline(1, color='green', ls=':', alpha=0.4, zorder=2)
    ax2.axhline(-1, color='green', ls=':', alpha=0.4, zorder=2)
    ax2.axhline(2, color='orange', ls=':', alpha=0.3, zorder=2)
    ax2.axhline(-2, color='orange', ls=':', alpha=0.3, zorder=2)
    
    # Fit linear trend to normalized residuals and show if significant
    w_centered = wave - np.mean(wave)
    slope_coef = np.polyfit(w_centered, residuals_norm, 1)
    resid_slope_norm = slope_coef[0]  # σ/Å
    if abs(resid_slope_norm) > 0.3:  # Show if > 0.3 σ/Å
        linear_trend = np.polyval(slope_coef, w_centered)
        ax2.plot(wave, linear_trend, 'r--', lw=1.5, alpha=0.8, zorder=4, 
                 label=f'Trend: {resid_slope_norm:+.2f} σ/Å')
        ax2.legend(loc='upper right', fontsize=7)
    
    ax2.set_xlabel('Wavelength (Å)')
    ax2.set_ylabel('Residuals (σ)')
    ax2.set_ylim(-5, 5)
    
    # Calculate normalized RMS
    rms_norm = float(np.std(residuals_norm))
    
    # Central region RMS (±1.5Å around target) - most important for EW
    central_mask = np.abs(wave - target_wave) < 1.5
    if np.sum(central_mask) > 5:
        central_resid_norm = residuals_norm[central_mask]
        rms_central_norm = float(np.std(central_resid_norm))
    else:
        rms_central_norm = rms_norm
    
    # Display normalized RMS (should be ~1 for good fit)
    ax2.text(0.02, 0.95, f'RMS={rms_norm:.2f}σ | Central={rms_central_norm:.2f}σ', 
             transform=ax2.transAxes, fontsize=9, va='top')
    
    fig.tight_layout()
    
    if format == 'base64':
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        # Include trend info if significant
        if abs(resid_slope_norm) > 0.3:
            trend_msg = f" RESIDUAL TREND={resid_slope_norm:+.2f} σ/Å (continuum slope problem!)."
        else:
            trend_msg = ""
        
        return {
            "success": True,
            "image_base64": img_base64,
            "description": f"Fit plot for {target_wave:.2f} Å. Green=target, Orange=other lines. "
                          f"RMS(norm)={rms_norm:.2f}σ, Central={rms_central_norm:.2f}σ.{trend_msg} "
                          f"Check if fit matches data, look for un-fitted lines at edges."
        }
    else:
        filename = f"fit_{target_wave:.2f}.png"
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return {"success": True, "file": filename}


def flag_line(line_wavelength: float, reason: str, exclude_from_analysis: bool = True) -> dict:
    """
    Flag a line as unreliable and optionally exclude from final analysis.
    Use this when a line has severe blending, poor fit quality, or other issues.
    
    Parameters:
        line_wavelength: Wavelength of the problematic line
        reason: Why the line is being flagged (e.g., "severe_blend", "poor_fit", "low_snr")
        exclude_from_analysis: Whether to exclude this line from final statistics (default True)
    
    Returns:
        Confirmation of flagging
    """
    flag_entry = {
        'wavelength': line_wavelength,
        'reason': reason,
        'excluded': exclude_from_analysis,
    }
    
    # Check if already flagged
    existing = [f for f in _get_session()['flagged_lines'] if abs(f['wavelength'] - line_wavelength) < 0.1]
    if existing:
        # Update existing flag
        for f in _get_session()['flagged_lines']:
            if abs(f['wavelength'] - line_wavelength) < 0.1:
                f['reason'] = reason
                f['excluded'] = exclude_from_analysis
        action = "updated"
    else:
        _get_session()['flagged_lines'].append(flag_entry)
        action = "added"
    
    return {
        "success": True,
        "action": action,
        "line_wavelength": line_wavelength,
        "reason": reason,
        "excluded": exclude_from_analysis,
        "total_flagged": len(_get_session()['flagged_lines']),
        "message": f"Line {line_wavelength:.2f} Å flagged as '{reason}' and {'excluded from' if exclude_from_analysis else 'included in'} analysis"
    }


def record_measurement(line_wavelength: float, ew_mA: float, ew_err_mA: float, 
                       method: str, quality: str, catalog_ew: float = None) -> dict:
    """
    Record a successful EW measurement for final analysis.
    
    Parameters:
        line_wavelength: Wavelength measured
        ew_mA: Measured EW in mÅ
        ew_err_mA: Measurement uncertainty
        method: Method used (voigt_fit, direct_integration)
        quality: Fit quality (good, acceptable, poor)
        catalog_ew: Catalog value for comparison (optional)
    
    Returns:
        Measurement record
    """
    measurement = {
        'wavelength': line_wavelength,
        'ew_mA': ew_mA,
        'ew_err_mA': ew_err_mA,
        'method': method,
        'quality': quality,
        'catalog_ew': catalog_ew,
        'diff_pct': ((ew_mA / catalog_ew) - 1) * 100 if catalog_ew else None,
    }
    
    # Check if line is flagged
    is_flagged = any(abs(f['wavelength'] - line_wavelength) < 0.1 and f['excluded'] 
                     for f in _get_session()['flagged_lines'])
    measurement['flagged'] = is_flagged
    
    # Update or add measurement
    existing_idx = None
    for i, m in enumerate(_get_session()['measurements']):
        if abs(m['wavelength'] - line_wavelength) < 0.1:
            existing_idx = i
            break
    
    if existing_idx is not None:
        _get_session()['measurements'][existing_idx] = measurement
    else:
        _get_session()['measurements'].append(measurement)
    
    return {
        "success": True,
        "measurement": measurement,
        "is_flagged": is_flagged,
        "total_measurements": len(_get_session()['measurements']),
    }



