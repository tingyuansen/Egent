"""
EW Tools - Core Functions for Equivalent Width Measurement
==========================================================

This module provides tools for measuring stellar equivalent widths (EW) 
from high-resolution spectra using multi-Voigt profile fitting.

ARCHITECTURE:
    Uses a thread-local session pattern for state during multi-step fitting.
    This enables parallel processing with ThreadPoolExecutor.

KEY FUNCTIONS (LLM Tools):
    1. load_spectrum(spectrum_file)     - Load rest-frame spectrum
    2. extract_region(wavelength)       - Extract ±window Å around target
    3. set_continuum_method()           - Configure continuum fitting
    4. set_continuum_regions()          - Manually specify continuum wavelengths
    5. fit_ew()                         - Fit multi-Voigt model to measure EW
    6. get_fit_plot()                   - Generate diagnostic plot
    7. flag_line()                      - Mark line as unreliable
    8. record_measurement()             - Record final EW measurement

FITTING APPROACH:
    1. Continuum: Iterative sigma-clipping or manual regions
    2. Peak Finding: Locate absorption features in inverted flux
    3. Multi-Voigt: Simultaneous fit of polynomial continuum + N Voigt profiles
    4. EW Integration: Integrate (1 - F_norm) under target Voigt profile
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from scipy.integrate import trapezoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
import threading


# =============================================================================
# THREAD-LOCAL SESSION MANAGEMENT
# =============================================================================

_thread_local = threading.local()


def _get_session():
    """
    Get the thread-local session dictionary.
    
    The session stores state for the current fitting workflow:
    - spectrum: Loaded DataFrame with wavelength, flux, flux_error
    - current_region: Extracted wavelength region around target
    - last_fit: Results from most recent fit_ew() call
    - continuum_method: Algorithm for continuum fitting
    """
    if not hasattr(_thread_local, 'session'):
        _thread_local.session = {
            # Spectrum data
            'spectrum': None,
            'spectrum_file': None,
            
            # Region and fit state
            'current_region': None,
            'last_fit': None,
            'fit_history': [],
            'current_line_iterations': [],
            
            # Continuum configuration
            'continuum_method': 'iterative_linear',
            'continuum_order': 1,
            'sigma_clip': 2.5,
            'top_percentile': 85,
            'window': 3.0,
            
            # Measurements and flags
            'flagged_lines': [],
            'measurements': [],
        }
    return _thread_local.session


def _reset_session():
    """Reset the session for a new spectrum."""
    if hasattr(_thread_local, 'session'):
        _thread_local.session = None


# =============================================================================
# CORE TOOLS
# =============================================================================

def load_spectrum(spectrum_file: str) -> dict:
    """
    Load a rest-frame spectrum for analysis.
    
    Expected CSV format:
        wavelength,flux,flux_error
        5000.0,1.05,0.02
        5000.1,0.98,0.02
        ...
    
    The spectrum should already be in the stellar rest frame.
    
    Parameters:
        spectrum_file: Path to CSV file with columns: wavelength, flux, flux_error
    
    Returns:
        Status and spectrum info
    """
    spec_file = Path(spectrum_file)
    
    if not spec_file.exists():
        return {"success": False, "error": f"Spectrum file not found: {spectrum_file}"}
    
    try:
        spec = pd.read_csv(spec_file)
    except Exception as e:
        return {"success": False, "error": f"Failed to read spectrum: {e}"}
    
    # Validate required columns
    required_cols = ['wavelength', 'flux', 'flux_error']
    missing = [c for c in required_cols if c not in spec.columns]
    if missing:
        return {"success": False, "error": f"Missing columns: {missing}. Required: {required_cols}"}
    
    # Sort by wavelength
    spec = spec.sort_values('wavelength').reset_index(drop=True)
    
    # Calculate SNR if not present
    if 'snr' not in spec.columns:
        spec['snr'] = spec['flux'] / spec['flux_error'].replace(0, np.nan)
        spec['snr'] = spec['snr'].fillna(1.0)
    
    _get_session()['spectrum'] = spec
    _get_session()['spectrum_file'] = str(spec_file)
    
    return {
        "success": True,
        "file": str(spec_file),
        "n_points": len(spec),
        "wavelength_range": [float(spec['wavelength'].min()), float(spec['wavelength'].max())],
        "median_snr": float(spec['snr'].median()),
    }


def extract_region(line_wavelength: float, window: float = None) -> dict:
    """
    Extract spectral region around a target line for fitting.
    
    Parameters:
        line_wavelength: Target wavelength in Angstroms
        window: Half-width window in Angstroms (default: 3.0 Å)
    
    Returns:
        Region info and diagnostic data
    """
    if _get_session()['spectrum'] is None:
        return {"success": False, "error": "No spectrum loaded. Call load_spectrum first."}
    
    spec = _get_session()['spectrum']
    window = window or _get_session()['window']
    
    # Extract region around target
    mask = (spec['wavelength'] >= line_wavelength - window) & \
           (spec['wavelength'] <= line_wavelength + window)
    region = spec[mask].copy()
    
    if len(region) == 0:
        return {
            "success": False, 
            "error": f"No data in range {line_wavelength-window:.1f}-{line_wavelength+window:.1f} Å",
            "flag_as": "no_data"
        }
    
    region = region.sort_values('wavelength')
    
    # Check if target wavelength is within the data
    wave_arr = region['wavelength'].values
    wave_min, wave_max = wave_arr.min(), wave_arr.max()
    
    if not (wave_min <= line_wavelength <= wave_max):
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
        'wave': region['wavelength'].values,
        'flux': region['flux'].values,
        'flux_err': region['flux_error'].values,
        'snr': region['snr'].values,
        'target_wave': line_wavelength,
        'window': window,
    }
    _get_session()['window'] = window
    
    return {
        "success": True,
        "target_wavelength": line_wavelength,
        "window": window,
        "n_points": len(region),
        "wavelength_coverage": [float(wave_min), float(wave_max)],
        "median_snr": float(region['snr'].median()),
    }


def set_continuum_method(method: str = 'iterative_linear', order: int = 1, 
                         sigma_clip: float = 2.5, top_percentile: float = None) -> dict:
    """
    Configure continuum fitting method.
    
    Parameters:
        method: 'iterative_linear', 'iterative_poly', 'top_percentile', 'manual_regions'
        order: Polynomial order (for iterative_poly)
        sigma_clip: Sigma threshold for iterative clipping
        top_percentile: Use top X% of flux values (default 85)
    
    Returns:
        Current continuum settings
    """
    valid_methods = ['iterative_linear', 'iterative_poly', 'top_percentile', 'manual_regions']
    if method not in valid_methods:
        return {"success": False, "error": f"Invalid method. Choose from: {valid_methods}"}
    
    _get_session()['continuum_method'] = method
    _get_session()['continuum_order'] = order
    _get_session()['sigma_clip'] = sigma_clip
    _get_session()['top_percentile'] = top_percentile or 85
    
    return {
        "success": True,
        "method": method,
        "order": order,
        "sigma_clip": sigma_clip,
        "top_percentile": _get_session()['top_percentile'],
    }


def set_continuum_regions(regions: list) -> dict:
    """
    Manually specify wavelength regions for continuum fitting.
    Use when automatic detection fails in crowded spectral regions.
    
    Parameters:
        regions: List of [start, end] wavelength pairs
                 e.g., [[5700, 5701], [5704, 5705]]
    
    Returns:
        Confirmation and pixel count
    """
    if _get_session()['current_region'] is None:
        return {"success": False, "error": "No region extracted. Call extract_region first."}
    
    region = _get_session()['current_region']
    wave = region['wave']
    
    # Build mask for continuum pixels
    cont_mask = np.zeros(len(wave), dtype=bool)
    for r in regions:
        if len(r) == 2:
            cont_mask |= (wave >= r[0]) & (wave <= r[1])
    
    n_selected = np.sum(cont_mask)
    
    if n_selected < 5:
        return {
            "success": False,
            "error": f"Only {n_selected} pixels selected. Need at least 5.",
        }
    
    _get_session()['continuum_regions'] = regions
    _get_session()['continuum_mask'] = cont_mask
    _get_session()['continuum_method'] = 'manual_regions'
    
    return {
        "success": True,
        "n_continuum_pixels": int(n_selected),
        "regions": regions,
    }


def fit_ew(min_peak_height: float = 0.02, min_prominence: float = 0.015,
           additional_peaks: list = None) -> dict:
    """
    Fit equivalent width using multi-Voigt model.
    
    Parameters:
        min_peak_height: Minimum absorption depth for peak detection (default 2%)
        min_prominence: Minimum prominence for peak detection
        additional_peaks: Wavelengths to add as extra Voigt components for blends
    
    Returns:
        Fit results and diagnostics
    """
    if _get_session()['current_region'] is None:
        return {"success": False, "error": "No region extracted. Call extract_region first."}
    
    region = _get_session()['current_region']
    wave = region['wave']
    flux = region['flux']
    flux_err = region['flux_err']
    target_wave = region['target_wave']
    
    # === CONTINUUM FITTING ===
    method = _get_session()['continuum_method']
    order = _get_session()['continuum_order']
    sigma = _get_session()['sigma_clip']
    top_pct = _get_session()['top_percentile']
    
    cont_mask = np.ones(len(flux), dtype=bool)
    
    if method == 'manual_regions' and _get_session().get('continuum_mask') is not None:
        cont_mask = _get_session()['continuum_mask']
        if len(cont_mask) != len(wave):
            # Regenerate if region changed
            regions = _get_session().get('continuum_regions', [])
            cont_mask = np.zeros(len(wave), dtype=bool)
            for r in regions:
                if len(r) == 2:
                    cont_mask |= (wave >= r[0]) & (wave <= r[1])
        
        if np.sum(cont_mask) < 5:
            # Fallback to top percentile
            threshold = np.percentile(flux, top_pct)
            cont_mask = flux >= threshold
        
        coef = np.polyfit(wave[cont_mask], flux[cont_mask], 1)
        continuum = np.polyval(coef, wave)
        continuum_info = {'method': 'manual_regions', 'points_used': int(np.sum(cont_mask))}
    
    elif method == 'top_percentile':
        threshold = np.percentile(flux, top_pct)
        cont_mask = flux >= threshold
        coef = np.polyfit(wave[cont_mask], flux[cont_mask], 1)
        continuum = np.polyval(coef, wave)
        continuum_info = {'method': 'top_percentile', 'percentile': top_pct}
    
    else:  # iterative_linear or iterative_poly
        poly_order = 1 if method == 'iterative_linear' else order
        
        # Initialize from top percentile
        threshold = np.percentile(flux, top_pct)
        cont_mask = flux >= threshold
        
        # Ensure points from left/right thirds
        n = len(wave)
        third = n // 3
        for section_mask in [np.arange(n) < third, np.arange(n) >= n - third]:
            if np.sum(cont_mask & section_mask) < 2:
                section_thresh = np.percentile(flux[section_mask], 80)
                cont_mask |= (section_mask & (flux >= section_thresh))
        
        # Iterate with sigma clipping
        for iteration in range(5):
            if np.sum(cont_mask) < 5:
                cont_mask = flux >= threshold
                break
            coef = np.polyfit(wave[cont_mask], flux[cont_mask], poly_order)
            continuum = np.polyval(coef, wave)
            residuals = flux - continuum
            std = np.std(residuals[cont_mask])
            new_mask = cont_mask & (residuals > -sigma * std)
            if np.sum(new_mask) == np.sum(cont_mask) or np.sum(new_mask) < 5:
                break
            cont_mask = new_mask
        
        continuum_info = {'method': method, 'iterations': iteration + 1}
    
    continuum_info['init_continuum_coef'] = coef[::-1].tolist()
    continuum_info['wave_center'] = float(np.mean(wave))
    
    # Normalize
    flux_norm = flux / continuum
    flux_norm_err = flux_err / continuum
    
    # === PEAK FINDING ===
    flux_inv = 1 - flux_norm
    peaks, _ = find_peaks(flux_inv, height=min_peak_height, distance=4, prominence=min_prominence)
    
    # Add manually specified peaks
    if additional_peaks:
        for add_wave in additional_peaks:
            idx = np.argmin(np.abs(wave - add_wave))
            if idx not in peaks and 0 < idx < len(wave) - 1:
                peaks = np.concatenate([peaks, [idx]])
        peaks = np.sort(peaks)
    
    if len(peaks) == 0:
        return {
            "success": False,
            "error": "No absorption peaks found",
            "diagnostics": {"continuum": continuum_info},
        }
    
    # === MULTI-VOIGT FITTING ===
    cont_order = _get_session()['continuum_order']
    n_cont_params = cont_order + 1
    
    def multi_voigt(wavelength, *params):
        cont_coeffs = params[:n_cont_params]
        w_centered = wavelength - wavelength.mean()
        
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
    
    n_lines = len(peaks)
    
    # Initial guesses and bounds
    p0 = [1.0] + [0.0] * cont_order
    lower = [0.9] + [-0.05] * cont_order
    upper = [1.1] + [0.05] * cont_order
    
    for p in peaks:
        depth = max(min_peak_height, flux_inv[p])
        p0.extend([depth, wave[p], 0.04, 0.015])
        lower.extend([0.005, wave[p]-0.5, 0.003, 0.001])
        upper.extend([0.9, wave[p]+0.5, 0.25, 0.08])
    
    try:
        popt, pcov = curve_fit(multi_voigt, wave, flux_norm, p0=p0,
                               sigma=flux_norm_err, absolute_sigma=True,
                               bounds=(lower, upper), maxfev=20000)
        
        # Check for merged lines (< 0.2 Å apart)
        fitted_centers = [popt[n_cont_params + i * 4 + 1] for i in range(n_lines)]
        centers_to_merge = []
        for i in range(len(fitted_centers)):
            for j in range(i+1, len(fitted_centers)):
                if abs(fitted_centers[i] - fitted_centers[j]) < 0.2:
                    centers_to_merge.append((i, j))
        
        if centers_to_merge:
            # Remove duplicates, refit
            peaks_to_remove = set()
            for i, j in centers_to_merge:
                dist_i = abs(fitted_centers[i] - target_wave)
                dist_j = abs(fitted_centers[j] - target_wave)
                peaks_to_remove.add(j if dist_i < dist_j else i)
            
            new_peaks = [peaks[i] for i in range(len(peaks)) if i not in peaks_to_remove]
            if new_peaks:
                p0 = [1.0] + [0.0] * cont_order
                lower = [0.9] + [-0.05] * cont_order
                upper = [1.1] + [0.05] * cont_order
                for p in new_peaks:
                    depth = max(min_peak_height, flux_inv[p])
                    p0.extend([depth, wave[p], 0.04, 0.015])
                    lower.extend([0.005, wave[p]-0.5, 0.003, 0.001])
                    upper.extend([0.9, wave[p]+0.5, 0.25, 0.08])
                popt, pcov = curve_fit(multi_voigt, wave, flux_norm, p0=p0,
                                      sigma=flux_norm_err, absolute_sigma=True,
                                      bounds=(lower, upper), maxfev=20000)
                peaks = new_peaks
                n_lines = len(peaks)
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Curve fit failed: {e}",
            "diagnostics": {"continuum": continuum_info, "n_peaks": n_lines},
        }
    
    # === DIAGNOSTICS ===
    flux_fit = multi_voigt(wave, *popt)
    residuals = flux_norm - flux_fit
    
    # Normalized RMS
    if np.any(flux_norm_err > 0):
        normalized_residuals = residuals / flux_norm_err
        fit_rms = float(np.std(normalized_residuals))
    else:
        fit_rms = float(np.std(residuals)) / 0.01
    
    # Central region RMS (±1.5Å around target)
    central_mask = np.abs(wave - target_wave) < 1.5
    if np.sum(central_mask) > 10:
        central_resid = residuals[central_mask] / flux_norm_err[central_mask]
        central_rms = float(np.std(central_resid))
    else:
        central_rms = fit_rms
    
    # Chi-squared
    chi2 = np.sum((residuals / flux_norm_err)**2)
    dof = len(residuals) - len(popt)
    reduced_chi2 = chi2 / dof if dof > 0 else chi2
    
    # Residual slope (continuum problem indicator)
    w_centered = wave - np.mean(wave)
    norm_resid = residuals / flux_norm_err
    resid_slope_norm, _ = np.polyfit(w_centered, norm_resid, 1)
    
    # Correlated residuals
    high_resid = np.abs(norm_resid) > 2
    max_consecutive = max((sum(1 for _ in g) for k, g in __import__('itertools').groupby(high_resid) if k), default=0)
    
    # Store continuum coefficients
    continuum_info['fitted_continuum_coeffs'] = popt[:n_cont_params].tolist()
    
    # === EXTRACT LINE PARAMETERS ===
    fitted_lines = []
    for i in range(n_lines):
        idx = n_cont_params + i * 4
        amp, center, sig, gam = popt[idx:idx+4]
        
        fwhm = 2.355 * sig
        half_range = max(0.5, 2.5 * fwhm)
        wave_fine = np.linspace(center - half_range, center + half_range, 2000)
        v = voigt_profile(wave_fine - center, sig, gam)
        v = v / v.max() * amp
        ew = trapezoid(v, wave_fine) * 1000  # mÅ
        
        try:
            perr = np.sqrt(np.diag(pcov))
            rel_err = np.sqrt((perr[idx]/amp)**2 + (perr[idx+2]/sig)**2)
            ew_err = ew * min(rel_err, 0.5)
        except:
            ew_err = ew * 0.15
        
        fitted_lines.append({
            'center': float(center),
            'amplitude': float(amp),
            'sigma': float(sig),
            'gamma': float(gam),
            'fwhm': float(fwhm),
            'ew_mA': float(ew),
            'ew_err_mA': float(ew_err),
            'distance_from_target': float(abs(center - target_wave)),
        })
    
    # Select target line (closest to target)
    target_line = None
    candidates = [l for l in fitted_lines if l['distance_from_target'] < 0.5]
    if candidates:
        closest = min(candidates, key=lambda x: x['distance_from_target'])
        very_close = [l for l in candidates if l['distance_from_target'] < closest['distance_from_target'] + 0.1]
        target_line = max(very_close, key=lambda x: x['ew_mA']) if len(very_close) > 1 else closest
    elif fitted_lines:
        target_line = min(fitted_lines, key=lambda x: x['distance_from_target'])
    
    # Store fit results
    _get_session()['last_fit'] = {
        'popt': popt.tolist(),
        'pcov': pcov.tolist(),
        'wave': wave.tolist(),
        'flux_norm': flux_norm.tolist(),
        'flux_norm_err': flux_norm_err.tolist(),
        'flux_fit': flux_fit.tolist(),
        'residuals': residuals.tolist(),
        'continuum': continuum.tolist(),
        'target_line': target_line,
        'all_lines': fitted_lines,
        'fit_rms': fit_rms,
        'central_rms': central_rms,
    }
    
    # === QUALITY ASSESSMENT ===
    quality_issues = []
    quality_warnings = []
    
    if target_line is None:
        quality_issues.append("No line found near target wavelength")
    else:
        if target_line['distance_from_target'] > 0.30:
            quality_issues.append(f"Large offset: {target_line['distance_from_target']*1000:.0f} mÅ from target")
        elif target_line['distance_from_target'] > 0.15:
            quality_warnings.append(f"Offset from target: {target_line['distance_from_target']*1000:.0f} mÅ")
    
    if fit_rms > 3.0:
        quality_issues.append(f"High RMS: {fit_rms:.2f}σ (expect ~1.0)")
    elif fit_rms > 2.0:
        quality_warnings.append(f"Elevated RMS: {fit_rms:.2f}σ")
    
    if abs(resid_slope_norm) > 0.5:
        quality_issues.append(f"Residual slope: {resid_slope_norm:.2f} σ/Å (continuum problem)")
    
    if reduced_chi2 > 5:
        quality_issues.append(f"High χ²/dof = {reduced_chi2:.1f}")
    
    if max_consecutive >= 5:
        quality_issues.append(f"Correlated residuals: {max_consecutive} consecutive >2σ points")
    
    if n_lines > 5:
        quality_warnings.append(f"Many lines ({n_lines}) in window")
    
    quality = "poor" if quality_issues else ("acceptable" if quality_warnings else "good")
    
    return {
        "success": True,
        "target_wavelength": target_wave,
        "target_line": target_line,
        "fit_quality": quality,
        "fit_rms": fit_rms,
        "reduced_chi2": reduced_chi2,
        "quality_issues": quality_issues,
        "quality_warnings": quality_warnings,
        "region_info": {'window': region['window'], 'wave_range': [float(wave[0]), float(wave[-1])]},
        "continuum_info": continuum_info,
        "diagnostics": {
            "n_lines_fitted": n_lines,
            "all_fitted_lines": fitted_lines,
            "central_rms": central_rms,
            "resid_slope_norm": float(resid_slope_norm),
        },
    }


def get_fit_plot(format: str = 'base64') -> dict:
    """
    Generate diagnostic plot of the current fit for visual inspection.
    
    Returns base64 image that the LLM can view for reasoning.
    
    Returns:
        Plot as base64 string
    """
    if _get_session()['last_fit'] is None:
        return {"success": False, "error": "No fit available. Run fit_ew first."}
    
    fit = _get_session()['last_fit']
    wave = np.array(fit['wave'])
    flux_norm = np.array(fit['flux_norm'])
    flux_fit = np.array(fit['flux_fit'])
    residuals = np.array(fit['residuals'])
    flux_norm_err = np.array(fit.get('flux_norm_err', [0.01]*len(wave)))
    target_line = fit['target_line']
    target_wave = _get_session()['current_region']['target_wave']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[3, 1], sharex=True)
    
    # Main panel: data and fit
    ax1.plot(wave, flux_norm, 'k-', lw=0.8, alpha=0.7, label='Data')
    ax1.plot(wave, flux_fit, 'r-', lw=1.5, label='Fit')
    ax1.axhline(1, color='gray', ls=':', alpha=0.5)
    
    # Mark all fitted lines
    for line in fit['all_lines']:
        color = 'green' if line == target_line else 'orange'
        ax1.axvline(line['center'], color=color, ls='--', alpha=0.5, lw=1)
        ax1.text(line['center'], 0.45, f"{line['ew_mA']:.1f}", fontsize=8, ha='center', color=color)
    
    # Mark target
    ax1.axvline(target_wave, color='blue', ls=':', alpha=0.7, lw=2, label=f'Target: {target_wave:.2f} Å')
    
    ax1.set_ylabel('Normalized Flux')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_ylim(0.4, 1.15)
    
    if target_line:
        ax1.set_title(f'Target: {target_wave:.2f} Å | EW={target_line["ew_mA"]:.1f}±{target_line["ew_err_mA"]:.1f} mÅ')
    
    # Residuals panel
    residuals_norm = residuals / flux_norm_err if np.any(flux_norm_err > 0) else residuals / 0.01
    
    ax2.axhspan(-1, 1, alpha=0.2, color='lightgreen', zorder=1, label='±1σ')
    ax2.axhspan(-2, 2, alpha=0.1, color='lightyellow', zorder=1, label='±2σ')
    ax2.plot(wave, residuals_norm, 'k-', lw=0.8, zorder=3)
    ax2.axhline(0, color='gray', ls='-', alpha=0.5, zorder=2)
    ax2.axhline(1, color='green', ls=':', alpha=0.4, zorder=2)
    ax2.axhline(-1, color='green', ls=':', alpha=0.4, zorder=2)
    ax2.axhline(2, color='orange', ls=':', alpha=0.3, zorder=2)
    ax2.axhline(-2, color='orange', ls=':', alpha=0.3, zorder=2)
    
    # Show trend line if significant
    w_centered = wave - np.mean(wave)
    slope_coef = np.polyfit(w_centered, residuals_norm, 1)
    if abs(slope_coef[0]) > 0.3:
        ax2.plot(wave, np.polyval(slope_coef, w_centered), 'r--', lw=1.5, alpha=0.8,
                 label=f'Trend: {slope_coef[0]:+.2f} σ/Å')
        ax2.legend(loc='upper right', fontsize=7)
    
    ax2.set_xlabel('Wavelength (Å)')
    ax2.set_ylabel('Residuals (σ)')
    ax2.set_ylim(-5, 5)
    
    rms_norm = float(np.std(residuals_norm))
    ax2.text(0.02, 0.95, f'RMS={rms_norm:.2f}σ', transform=ax2.transAxes, fontsize=9, va='top')
    
    fig.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return {
        "success": True,
        "image_base64": img_base64,
        "description": f"Fit plot for {target_wave:.2f} Å. RMS={rms_norm:.2f}σ.",
    }


def flag_line(line_wavelength: float, reason: str, exclude_from_analysis: bool = True) -> dict:
    """
    Flag a line as unreliable.
    
    Parameters:
        line_wavelength: Wavelength of the problematic line
        reason: Why flagged (no_data, severe_blend, fit_failed, bad_continuum, wrong_line)
        exclude_from_analysis: Whether to exclude from final analysis
    
    Returns:
        Confirmation
    """
    flag_entry = {
        'wavelength': line_wavelength,
        'reason': reason,
        'excluded': exclude_from_analysis,
    }
    
    # Check if already flagged
    existing = [f for f in _get_session()['flagged_lines'] if abs(f['wavelength'] - line_wavelength) < 0.1]
    if existing:
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
        "message": f"Line {line_wavelength:.2f} Å flagged as '{reason}'"
    }


def record_measurement(line_wavelength: float, ew_mA: float, ew_err_mA: float,
                       method: str, quality: str) -> dict:
    """
    Record a successful EW measurement.
    
    Parameters:
        line_wavelength: Wavelength measured
        ew_mA: Measured EW in milli-Angstroms
        ew_err_mA: Measurement uncertainty
        method: Method used (voigt_fit)
        quality: Fit quality (good, acceptable, poor)
    
    Returns:
        Measurement record
    """
    measurement = {
        'wavelength': line_wavelength,
        'ew_mA': ew_mA,
        'ew_err_mA': ew_err_mA,
        'method': method,
        'quality': quality,
    }
    
    # Check if flagged
    is_flagged = any(abs(f['wavelength'] - line_wavelength) < 0.1 and f['excluded']
                     for f in _get_session()['flagged_lines'])
    measurement['flagged'] = is_flagged
    
    # Update or add
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
        "total_measurements": len(_get_session()['measurements']),
    }
