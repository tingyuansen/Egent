#!/usr/bin/env python3
"""
Utility Functions for EW Analysis
==================================

Plotting utilities and helper functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import voigt_profile
from typing import Dict, Any, Optional, Tuple
import base64
import io


# =============================================================================
# GAIA ID HELPERS
# =============================================================================

def get_pair_info_from_gaia_id(gaia_id: int) -> Optional[Dict[str, Any]]:
    """
    Look up pair_id and component from Gaia ID using the calibration file.
    
    Returns:
        {'pair_id': int, 'component': str, 'ew_col': str} or None if not found
    """
    cal_file = Path(__file__).parent / 'data' / 'spectra_good_calibration.csv'
    if not cal_file.exists():
        return None
    
    df_cal = pd.read_csv(cal_file)
    match = df_cal[df_cal['gaia_id'] == gaia_id]
    
    if len(match) == 0:
        return None
    
    row = match.iloc[0]
    return {
        'pair_id': int(row['pair_id']),
        'component': row['component'],
        'ew_col': f"EW_Pair_{int(row['pair_id'])}{row['component']}"
    }


# =============================================================================
# UNIFIED LINE PLOTTING
# =============================================================================

def plot_line_fit(
    gaia_id: int, 
    line_result: dict, 
    output_dir: Path = None,
    spectrum_data: dict = None,
    return_base64: bool = False,
    save_file: bool = True,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Unified line fit plotting function.
    
    Used by both:
    - LLM reasoning (return_base64=True for vision)
    - Replot/post-analysis (save_file=True)
    
    Args:
        gaia_id: Gaia source ID
        line_result: Result dictionary for this line
        output_dir: Output directory for plots (if saving)
        spectrum_data: Pre-loaded spectrum {'df': DataFrame}
        return_base64: Return base64-encoded image for LLM vision
        save_file: Save plot to disk
        
    Returns:
        (file_path, base64_string) - either may be None based on args
    """
    from ew_tools import load_spectrum, _get_session
    
    wave = line_result.get('line') or line_result.get('wavelength')
    if wave is None:
        return None, None
    
    # Load spectrum if not provided
    if spectrum_data is None:
        load_spectrum(gaia_id)
        session = _get_session()
        spec = session['spectrum']
    else:
        spec = spectrum_data['df']
    
    wave_col = 'wavelength'
    
    # Get region info
    region_info = line_result.get('region_info') or {}
    wave_range = region_info.get('wave_range')
    stored_order = region_info.get('echelle_order')
    
    if wave_range and len(wave_range) == 2:
        nearby = spec[(spec[wave_col] >= wave_range[0]) & (spec[wave_col] <= wave_range[1])]
    else:
        nearby = spec[(spec[wave_col] >= wave - 3) & (spec[wave_col] <= wave + 3)]
    
    if len(nearby) == 0:
        return None, None
    
    # Handle echelle orders
    if 'echelle_order' in nearby.columns:
        if stored_order and stored_order in nearby['echelle_order'].values:
            nearby = nearby[nearby['echelle_order'] == stored_order]
        else:
            orders = nearby['echelle_order'].unique()
            if len(orders) > 1:
                order_counts = nearby.groupby('echelle_order').size()
                nearby = nearby[nearby['echelle_order'] == order_counts.idxmax()]
    
    if wave_range and len(wave_range) == 2:
        nearby = nearby[(nearby[wave_col] >= wave_range[0]) & (nearby[wave_col] <= wave_range[1])]
        if len(nearby) == 0:
            return None, None
    
    w = nearby[wave_col].values
    f = nearby['flux'].values
    sort_idx = np.argsort(w)
    w, f = w[sort_idx], f[sort_idx]
    
    # Get Voigt parameters
    if line_result.get('used_llm') and line_result.get('iterations'):
        all_lines = line_result['iterations'][-1].get('all_lines', [])
    else:
        voigt_params = line_result.get('direct_voigt_params') or {}
        all_lines = voigt_params.get('all_lines', [])
    
    # Continuum normalization
    continuum_info = line_result.get('continuum_info') or {}
    init_coef = continuum_info.get('init_continuum_coef')
    fitted_coeffs = continuum_info.get('fitted_continuum_coeffs') or continuum_info.get('coeffs')
    
    if init_coef:
        continuum_init = np.polyval(init_coef[::-1], w)
    else:
        threshold = np.percentile(f, 85)
        cont_mask = f >= threshold
        if np.sum(cont_mask) >= 2:
            coef_init = np.polyfit(w[cont_mask], f[cont_mask], 1)
            continuum_init = np.polyval(coef_init, w)
        else:
            continuum_init = np.percentile(f, 95)
    
    f_prenorm = f / continuum_init
    
    if fitted_coeffs and len(fitted_coeffs) > 0:
        w_centered = w - np.mean(w)
        continuum_fine = sum(c * (w_centered ** i) for i, c in enumerate(fitted_coeffs))
        f_norm = f_prenorm / continuum_fine
    else:
        # Fallback: use pre-normalized
        f_norm = f_prenorm
    
    f_norm_err = np.maximum(f_norm / 100, 0.001)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[3, 1], sharex=True)
    
    ax1.plot(w, f_norm, 'k-', lw=1.0, alpha=0.9, label='Data')
    ax1.axhline(1, color='gray', ls=':', alpha=0.5)
    ax1.axvline(wave, color='blue', ls=':', lw=2, alpha=0.7, label='Target')
    
    # Build model from Voigt components
    model = np.ones_like(w)
    if all_lines:
        for line in all_lines:
            center = line.get('center', 0)
            amp = line.get('amplitude', 0)
            sig = line.get('sigma', 0.05)
            gam = line.get('gamma', 0.05)
            
            if amp > 0 and sig > 0:
                v = voigt_profile(w - center, sig, gam)
                if v.max() > 0:
                    v = v / v.max()
                model -= amp * v
                
                color = 'green' if abs(center - wave) < 0.3 else 'orange'
                ax1.axvline(center, color=color, ls='--', alpha=0.5, lw=1)
        
        ax1.plot(w, model, 'r-', lw=1.5, label='Voigt fit')
        residuals = f_norm - model
    else:
        residuals = f_norm - 1.0
    
    # Labels
    sp = line_result.get('species', '?')
    meas_ew = line_result.get('measured_ew', 0)
    diff_pct = line_result.get('diff_pct', 0) or 0
    flagged = line_result.get('flagged', False)
    used_llm = line_result.get('used_llm', False)
    
    status = "FLAGGED" if flagged else ("LLM" if used_llm else "DIRECT")
    title = f'{sp} {wave:.2f} Å | {status} | EW={meas_ew:.1f} mÅ | Δ={diff_pct:+.1f}%'
    ax1.set_title(title, fontsize=11, weight='bold', color='red' if flagged else 'black')
    ax1.set_ylabel('Normalized Flux')
    ax1.set_ylim(0.3, 1.15)
    ax1.legend(loc='lower right', fontsize=9)
    
    # Residuals panel
    residuals_norm = residuals / f_norm_err
    rms_norm = np.std(residuals_norm)
    
    ax2.axhspan(-1, 1, alpha=0.2, color='lightgreen', zorder=1)
    ax2.axhspan(-2, 2, alpha=0.1, color='lightyellow', zorder=1)
    ax2.plot(w, residuals_norm, 'k-', lw=0.8, zorder=3)
    ax2.axhline(0, color='gray', ls='-', alpha=0.5)
    ax2.axhline(1, color='green', ls=':', alpha=0.4)
    ax2.axhline(-1, color='green', ls=':', alpha=0.4)
    ax2.axhline(2, color='orange', ls=':', alpha=0.3)
    ax2.axhline(-2, color='orange', ls=':', alpha=0.3)
    ax2.set_xlabel('Wavelength (Å)')
    ax2.set_ylabel('Residuals (σ)')
    ax2.set_ylim(-4, 4)
    ax2.text(0.02, 0.95, f'RMS={rms_norm:.2f}σ', transform=ax2.transAxes, fontsize=10, va='top')
    
    fig.tight_layout()
    
    file_path = None
    b64_string = None
    
    try:
        # Return base64 for LLM vision
        if return_base64:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            buf.seek(0)
            b64_string = base64.b64encode(buf.read()).decode('utf-8')
        
        # Save to disk
        if save_file and output_dir:
            # Determine subfolder based on result type
            # Error cases: LLM timeout, API errors, or no data
            flag_reason = line_result.get('flag_reason', '')
            has_error = (
                line_result.get('error') or 
                line_result.get('llm_error') or 
                line_result.get('llm_timeout') or
                'no_data' in str(flag_reason)
            )
            if has_error:
                subdir = output_dir / 'error'
            elif flagged:
                subdir = output_dir / 'flagged'
            elif used_llm:
                subdir = output_dir / 'llm'
            else:
                subdir = output_dir / 'direct'
            subdir.mkdir(parents=True, exist_ok=True)
            
            outfile = subdir / f'{wave:.2f}_{sp}.png'
            fig.savefig(outfile, dpi=120, bbox_inches='tight')
            file_path = str(outfile)
    finally:
        plt.close(fig)
        # Periodically close all to prevent memory buildup in long runs
        if hasattr(plt, '_pylab_helpers'):
            import gc
            gc.collect()
    
    return file_path, b64_string


# Alias for backwards compatibility
def plot_single_line(gaia_id: int, line_result: dict, output_dir: Path,
                     spectrum_data: dict = None) -> str:
    """Backwards-compatible alias for plot_line_fit."""
    file_path, _ = plot_line_fit(gaia_id, line_result, output_dir, spectrum_data,
                                  return_base64=False, save_file=True)
    return file_path


# =============================================================================
# COMPARISON PLOTS
# =============================================================================

def generate_all_plots(results: list, gaia_id: int, timestamp: str, out_dir: Path = None):
    """Generate all diagnostic plots for an analysis run."""
    if out_dir is None:
        out_dir = Path(__file__).parent / 'output'
    out_dir.mkdir(exist_ok=True)
    
    prefix = f"gaia{gaia_id}_"
    generate_comparison_plot(results, gaia_id, timestamp, out_dir, prefix)
    
    print(f"  Comparison plot saved: {prefix}comparison_{timestamp}.png")
    print(f"  For individual line plots, run: python replot_results.py <json_file> --all-lines")


def generate_comparison_plot(results: list, gaia_id: int, timestamp: str, 
                             out_dir: Path, prefix: str = ""):
    """
    Generate 1-to-1 comparison plot (catalog EW vs measured EW).
    """
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # Separate flagged and unflagged
    unflagged = [r for r in results if r.get('catalog_ew') and r.get('measured_ew') 
                 and r['catalog_ew'] > 0 and not r.get('flagged')]
    flagged = [r for r in results if r.get('catalog_ew') and r.get('measured_ew') 
               and r['catalog_ew'] > 0 and r.get('flagged')]
    
    if not unflagged and not flagged:
        print("  No valid measurements for plot")
        plt.close(fig)
        return
    
    # Extract EW arrays
    cat_ews = np.array([r['catalog_ew'] for r in unflagged]) if unflagged else np.array([])
    direct_ews = np.array([r.get('direct_ew', r.get('measured_ew', 0)) for r in unflagged]) if unflagged else np.array([])
    final_ews = np.array([r['measured_ew'] for r in unflagged]) if unflagged else np.array([])
    
    # Set axis limits
    all_vals = list(cat_ews) + list(final_ews) + [r['catalog_ew'] for r in flagged] + [r['measured_ew'] for r in flagged]
    lim = [0, max(all_vals) * 1.1] if all_vals else [0, 150]
    ax.plot(lim, lim, 'k--', lw=1.5, alpha=0.7, label='1:1')
    
    # Plot direct baseline (gray x)
    if len(cat_ews) > 0:
        ax.scatter(cat_ews, direct_ews, c='gray', s=20, marker='x', alpha=0.5, 
                   label='Direct', zorder=1)
    
    # Plot flagged as hollow circles
    for r in flagged:
        ax.scatter(r['catalog_ew'], r['measured_ew'], facecolors='none', edgecolors='gray', 
                  s=80, lw=2, alpha=0.6, zorder=2, marker='o')
    
    # Plot unflagged colored by deviation
    for i, r in enumerate(unflagged):
        diff = abs(final_ews[i]/cat_ews[i] - 1) * 100
        color = 'green' if diff < 10 else 'blue' if diff < 15 else 'orange' if diff < 20 else 'red'
        ax.scatter(cat_ews[i], final_ews[i], c=color, s=60, alpha=0.8, edgecolor='k', lw=0.5, zorder=3)
        
        # Arrow from direct to final if LLM was used
        if r.get('used_llm') and direct_ews[i] > 0:
            ax.annotate('', xy=(cat_ews[i], final_ews[i]), xytext=(cat_ews[i], direct_ews[i]),
                       arrowprops=dict(arrowstyle='->', color='purple', lw=1.5, alpha=0.6), zorder=2)
    
    # Statistics
    if len(cat_ews) > 0:
        final_diffs = np.abs(final_ews/cat_ews - 1) * 100
        direct_diffs = np.abs(direct_ews/cat_ews - 1) * 100
        within15_final = np.sum(final_diffs < 15)
        within15_direct = np.sum(direct_diffs < 15)
    else:
        final_diffs = direct_diffs = np.array([])
        within15_final = within15_direct = 0
    
    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='green', edgecolor='k', label='<10%%: %d' % (np.sum(final_diffs < 10) if len(final_diffs) else 0)),
        Patch(facecolor='blue', edgecolor='k', label='10-15%%: %d' % (np.sum((final_diffs >= 10) & (final_diffs < 15)) if len(final_diffs) else 0)),
        Patch(facecolor='orange', edgecolor='k', label='15-20%%: %d' % (np.sum((final_diffs >= 15) & (final_diffs < 20)) if len(final_diffs) else 0)),
        Patch(facecolor='red', edgecolor='k', label='>20%%: %d' % (np.sum(final_diffs >= 20) if len(final_diffs) else 0)),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='gray', 
               markersize=10, markeredgewidth=2, label='Flagged: %d' % len(flagged)),
        Line2D([0], [0], marker='x', color='gray', markersize=8, linestyle='', label='Direct'),
        Line2D([0], [0], color='purple', lw=2, label='LLM correction'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    ax.set_xlabel('Catalog EW (mÅ)', fontsize=12)
    ax.set_ylabel('Measured EW (mÅ)', fontsize=12)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    if len(final_diffs) > 0:
        title = f'Gaia {gaia_id}\nDirect: {within15_direct}/{len(direct_diffs)} ({100*within15_direct/len(direct_diffs):.0f}%) | Final: {within15_final}/{len(final_diffs)} ({100*within15_final/len(final_diffs):.0f}%) within 15%\nMean|Δ|: Direct={np.mean(direct_diffs):.1f}%, Final={np.mean(final_diffs):.1f}% | Flagged: {len(flagged)}'
    else:
        title = f'Gaia {gaia_id}\nNo unflagged results | Flagged: {len(flagged)}'
    ax.set_title(title, fontsize=11)
    
    fig.tight_layout()
    fig.savefig(out_dir / f'{prefix}comparison_{timestamp}.png', dpi=150)
    plt.close(fig)
    print(f"  1-to-1 plot: {prefix}comparison_{timestamp}.png")


def cleanup_temp_plots(plots_dir: Path = None):
    """Clean up temporary plot files used for LLM vision."""
    if plots_dir is None:
        plots_dir = Path(__file__).parent / 'plots'
    
    if plots_dir.exists():
        import shutil
        shutil.rmtree(plots_dir)
        plots_dir.mkdir()
        print(f"  Cleaned up temporary plots in {plots_dir}")
