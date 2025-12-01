#!/usr/bin/env python3
"""
Egent: LLM-Powered Equivalent Width Measurement
================================================

Autonomous EW measurement using LLM-guided multi-Voigt fitting.

Pipeline:
    1. Direct Voigt Fitting: Fast deterministic fitting with quality metrics
    2. LLM Visual Inspection: For borderline cases, LLM inspects plots and
       can adjust continuum, window size, add blend peaks, or flag lines

Usage:
    python run_ew.py --spectrum spectrum.csv --lines linelist.csv
    python run_ew.py --spectrum spectrum.csv --lines linelist.csv --workers 5

Input Files:
    spectrum.csv: wavelength,flux,flux_error (rest-frame)
    linelist.csv: wavelength (one line per row)

Output:
    results_<timestamp>.json - Full results with Voigt parameters
    fits/ - Diagnostic plots for each line
"""

import json
import time
import base64
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import get_config
from llm_client import LLMClient


# =============================================================================
# TOOL DEFINITIONS FOR LLM
# =============================================================================

TOOLS = [
    {"type": "function", "function": {
        "name": "load_spectrum",
        "description": "Load spectrum from file (already done - use to verify)",
        "parameters": {"type": "object", "properties": {
            "spectrum_file": {"type": "string", "description": "Path to spectrum CSV"}
        }, "required": ["spectrum_file"]}}},

    {"type": "function", "function": {
        "name": "extract_region",
        "description": "Extract spectral region around target line. window=5.0 for crowded, window=2.0 for isolated.",
        "parameters": {"type": "object", "properties": {
            "line_wavelength": {"type": "number", "description": "Target wavelength in Angstroms"},
            "window": {"type": "number", "description": "Half-width in Angstroms (default 3.0)"}
        }, "required": ["line_wavelength"]}}},

    {"type": "function", "function": {
        "name": "set_continuum_method",
        "description": "Configure continuum fitting. 'iterative_linear' (default), 'iterative_poly' (order=2 for curved), 'top_percentile'.",
        "parameters": {"type": "object", "properties": {
            "method": {"type": "string", "enum": ["iterative_linear", "iterative_poly", "top_percentile"]},
            "order": {"type": "integer", "description": "Polynomial order (for iterative_poly)"},
            "top_percentile": {"type": "number", "description": "Use top X% of flux (default 85)"}
        }, "required": ["method"]}}},

    {"type": "function", "function": {
        "name": "set_continuum_regions",
        "description": "Manually specify line-free wavelength regions for continuum. Example: [[5700, 5701], [5704, 5705]]",
        "parameters": {"type": "object", "properties": {
            "regions": {"type": "array", "items": {"type": "array", "items": {"type": "number"}},
                       "description": "List of [start, end] wavelength pairs"}
        }, "required": ["regions"]}}},

    {"type": "function", "function": {
        "name": "fit_ew",
        "description": "Fit multi-Voigt model. Use additional_peaks for missed blends showing W-shaped residuals.",
        "parameters": {"type": "object", "properties": {
            "additional_peaks": {"type": "array", "items": {"type": "number"},
                               "description": "Wavelengths to add as extra Voigt components"}
        }, "required": []}}},

    {"type": "function", "function": {
        "name": "get_fit_plot",
        "description": "Generate diagnostic plot for visual inspection.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},

    {"type": "function", "function": {
        "name": "flag_line",
        "description": "Flag line as unreliable. Reasons: no_data, severe_blend, fit_failed, bad_continuum",
        "parameters": {"type": "object", "properties": {
            "line_wavelength": {"type": "number"},
            "reason": {"type": "string", "enum": ["no_data", "severe_blend", "fit_failed", "bad_continuum"]}
        }, "required": ["line_wavelength", "reason"]}}},

    {"type": "function", "function": {
        "name": "record_measurement",
        "description": "Record final EW measurement",
        "parameters": {"type": "object", "properties": {
            "line_wavelength": {"type": "number"},
            "ew_mA": {"type": "number", "description": "Equivalent width in milli-Angstroms"},
            "ew_err_mA": {"type": "number", "description": "Uncertainty in milli-Angstroms"},
            "quality": {"type": "string", "description": "good, acceptable, or poor"}
        }, "required": ["line_wavelength", "ew_mA", "ew_err_mA", "quality"]}}},
]


# =============================================================================
# LLM SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are an expert reviewer for stellar EW measurements. You're called for BORDERLINE cases where automated fitting is uncertain.

YOUR ROLE: Use visual inspection to make judgment calls that algorithms can't.

WORKFLOW:
1. The fit is already done. Call get_fit_plot to VISUALLY INSPECT.
2. Make a judgment: ACCEPT, IMPROVE, or FLAG.

=== CRITICAL: FOCUS ONLY ON TARGET REGION ===

The TARGET LINE is marked with a BLUE DASHED vertical line.
ONLY care about fit quality within ±0.5Å of this target.

**EDGE vs TARGET:**
- IGNORE: Spikes/noise at window edges (>1.5Å from target)
- CHECK: Residuals within ±0.5Å of the blue target line
- If target region is clean but edges are messy → ACCEPT
- If target region has W-pattern or blend → FIX or FLAG

=== RMS INTERPRETATION ===

**RMS < 1.5σ: LIKELY GOOD** - still check for W-shaped residuals at target
**RMS 1.5-2.0σ: GOOD** - check target region only, accept if clean
**RMS 2.0-2.5σ: MARGINAL** - try to improve, accept if target is clean after
**RMS > 2.5σ: POOR** - needs work or FLAG

=== BLEND DETECTION ===

Look for these patterns in residuals NEAR THE TARGET (±1Å):

1. **W-SHAPED residuals** at target = MISSED BLEND
   - Fix: fit_ew(additional_peaks=[wavelength_of_blend])

2. **ASYMMETRIC wing** on target line
   - Fix: add the wing as additional_peak

3. **Systematic offset** at target center
   - May need continuum adjustment

=== DECISIONS ===

**ACCEPT** (call record_measurement) if:
- RMS < 1.5σ (excellent)
- OR: RMS < 2.0σ AND target region residuals within ±2σ

**IMPROVE** by trying:
1. Add missing peaks: fit_ew(additional_peaks=[...])
2. Polynomial continuum: set_continuum_method('iterative_poly', order=2)
3. Window adjustment: extract_region with window=2.0 or 5.0
4. Manual continuum: set_continuum_regions([[start, end], ...])

**FLAG** (use sparingly!) only when:
- Problem is AT THE TARGET (±0.5Å), not at edges
- RMS > 2.5σ that cannot be improved
- Target line is completely blended or absent"""


# =============================================================================
# SPECTRUM FILE STORAGE (for parallel workers)
# =============================================================================

_SPECTRUM_FILE = None


# =============================================================================
# FITTING FUNCTIONS
# =============================================================================

def direct_fit(spectrum_file: str, line_wave: float, add_noise: bool = False) -> dict:
    """
    Direct Voigt fitting without LLM intervention.
    
    Args:
        spectrum_file: Path to spectrum CSV
        line_wave: Target wavelength
        add_noise: Add stochasticity for retry diversity
    """
    import random
    from ew_tools import load_spectrum, extract_region, set_continuum_method, fit_ew, _get_session

    load_spectrum(spectrum_file)
    
    if add_noise:
        window = 3.0 + random.uniform(-0.5, 0.5)
        top_percentile = 85 + random.randint(-5, 5)
    else:
        window = 3.0
        top_percentile = 85
    
    region_result = extract_region(line_wave, window=window)

    if not region_result.get('success'):
        return {
            'success': False,
            'flagged': True,
            'flag_reason': region_result.get('flag_as', 'no_data'),
            'error': region_result.get('error', 'No data')
        }

    set_continuum_method('iterative_linear', order=1, top_percentile=top_percentile)
    result = fit_ew()

    if result['success'] and result.get('target_line'):
        ew = result['target_line']['ew_mA']
        err = result['target_line']['ew_err_mA']
        rms = result.get('fit_rms', 1.0)
        central_rms = result.get('diagnostics', {}).get('central_rms', rms)
        quality = result.get('fit_quality', 'poor')
        reduced_chi2 = result.get('reduced_chi2', 1.0)

        needs_improvement = False
        improvement_reason = None
        n_lines = result.get('diagnostics', {}).get('n_lines_fitted', 0)

        if quality == 'poor':
            needs_improvement = True
            improvement_reason = 'poor_quality'
        elif n_lines >= 10:
            needs_improvement = True
            improvement_reason = 'crowded_region'
        elif reduced_chi2 > 15:
            needs_improvement = True
            improvement_reason = 'elevated_chi2'
        elif central_rms > 2.5:
            needs_improvement = True
            improvement_reason = 'elevated_central_rms'

        resid_slope = result.get('diagnostics', {}).get('resid_slope_norm', 0)
        if abs(resid_slope) > 0.5:
            needs_improvement = True
            improvement_reason = f'residual_slope_{resid_slope:.2f}'

        session = _get_session()
        region = session.get('current_region', {})

        return {
            'success': True,
            'measured_ew': ew,
            'ew_err': err,
            'fit_rms': rms,
            'fit_quality': quality,
            'reduced_chi2': reduced_chi2,
            'n_lines': n_lines,
            'needs_improvement': needs_improvement,
            'improvement_reason': improvement_reason,
            'region_info': result.get('region_info'),
            'continuum_info': result.get('continuum_info'),
            'voigt_params': {
                'target': result['target_line'],
                'all_lines': result.get('diagnostics', {}).get('all_fitted_lines', []),
            },
        }

    return {'success': False, 'needs_improvement': True, 'improvement_reason': 'fit_failed'}


def llm_measure_with_vision(
    spectrum_file: str,
    line_wave: float,
    direct_result: dict = None,
    output_dir: str = None,
) -> dict:
    """Use LLM with vision to review and improve the EW measurement."""
    from ew_tools import (
        load_spectrum, extract_region, set_continuum_method, set_continuum_regions,
        fit_ew, get_fit_plot, flag_line, record_measurement
    )

    def execute_tool(name, inputs):
        if name == "load_spectrum":
            return load_spectrum(inputs["spectrum_file"])
        elif name == "extract_region":
            return extract_region(inputs["line_wavelength"], inputs.get("window"))
        elif name == "set_continuum_method":
            return set_continuum_method(inputs["method"], inputs.get("order", 1),
                                       top_percentile=inputs.get("top_percentile", 85))
        elif name == "set_continuum_regions":
            return set_continuum_regions(inputs["regions"])
        elif name == "fit_ew":
            return fit_ew(additional_peaks=inputs.get("additional_peaks"))
        elif name == "get_fit_plot":
            return get_fit_plot()
        elif name == "flag_line":
            return flag_line(inputs["line_wavelength"], inputs["reason"], True)
        elif name == "record_measurement":
            return record_measurement(inputs["line_wavelength"], inputs["ew_mA"],
                                      inputs["ew_err_mA"], "voigt_fit", inputs["quality"])
        return {"error": f"Unknown tool: {name}"}

    client = LLMClient()

    context = ""
    if direct_result and direct_result.get('success'):
        context = f"\nDirect fit: {direct_result['measured_ew']:.1f}±{direct_result['ew_err']:.1f} mÅ"
        context += f", RMS={direct_result.get('fit_rms', 0):.2f}σ, quality={direct_result['fit_quality']}"
        context += "\nPlease visually inspect the fit and improve if needed."

    prompt = f"Measure EW for line at {line_wave:.2f} Å.{context}"
    messages = [{"role": "user", "content": prompt}]

    start_time = time.time()
    MAX_LLM_TIME = 300
    final_ew, final_err, final_quality = None, None, None
    flagged, flag_reason = False, None
    iterations = []

    for turn in range(10):
        if time.time() - start_time > MAX_LLM_TIME:
            break

        try:
            response = client.chat(messages, tools=TOOLS, system_prompt=SYSTEM_PROMPT, timeout=90)
        except Exception as e:
            error_str = str(e).lower()
            is_fatal = not any(x in error_str for x in ['rate', 'limit', 'timeout', '429', '503', '500'])
            if is_fatal:
                return {'success': False, 'error': str(e), 'method': 'llm_error'}
            return {'success': False, 'error': f'Rate limit: {e}', 'method': 'llm_rate_limit'}

        message = response.choices[0].message

        if message.tool_calls is None or len(message.tool_calls) == 0:
            break

        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in message.tool_calls
            ]
        })

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            inputs = json.loads(tool_call.function.arguments)
            result = execute_tool(name, inputs)

            if name == "fit_ew" and result.get("success"):
                target = result.get('target_line', {})
                all_lines = result.get('diagnostics', {}).get('all_fitted_lines', [])
                iterations.append({
                    'ew': target.get('ew_mA'),
                    'ew_err': target.get('ew_err_mA'),
                    'quality': result.get('fit_quality'),
                    'rms': result.get('fit_rms'),
                    'target_voigt': target,
                    'all_lines': all_lines,
                    'region_info': result.get('region_info'),
                    'continuum_info': result.get('continuum_info'),
                })
                if target:
                    final_ew = target['ew_mA']
                    final_err = target['ew_err_mA']
                    final_quality = result.get('fit_quality')

            if name == "record_measurement":
                final_ew = inputs["ew_mA"]
                final_err = inputs["ew_err_mA"]
                final_quality = inputs["quality"]

            if name == "flag_line":
                flagged = True
                flag_reason = inputs.get("reason")

            if name == "get_fit_plot" and result.get("image_base64"):
                image_b64 = result["image_base64"]

                # Save plot
                if output_dir:
                    plot_dir = Path(output_dir) / "plots"
                    plot_dir.mkdir(exist_ok=True, parents=True)
                    plot_file = plot_dir / f"llm_{line_wave:.2f}_{int(time.time())}.png"
                    with open(plot_file, 'wb') as f:
                        f.write(base64.b64decode(image_b64))

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": "Plot generated. Sending for visual inspection..."
                })

                current_rms = iterations[-1].get('rms', 0) if iterations else 0
                rms_warning = ""
                if current_rms > 5.0:
                    rms_warning = f"\n\n⚠️ CRITICAL: RMS={current_rms:.1f}σ is very high.\n"
                elif current_rms > 2.5:
                    rms_warning = f"\n\n⚠️ WARNING: RMS={current_rms:.1f}σ is elevated.\n"

                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": "high"}},
                        {"type": "text", "text": f"""Inspect this fit:{rms_warning}

1. Does RED fit match BLACK data at the target?
2. Check for W-shaped residuals (missed blends)
3. Check for continuum issues (tilted residuals)

**Decisions based on RMS:**
- RMS < 1.5: GOOD - can ACCEPT
- RMS 1.5-2.5: ACCEPTABLE
- RMS > 2.5: POOR - needs improvement
- RMS > 5: CATASTROPHIC - FLAG or major fixes

Record measurement and quality."""}
                    ]
                })
            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, default=str)
                })

    return {
        'success': (final_ew is not None) or flagged,
        'measured_ew': final_ew,
        'ew_err': final_err,
        'fit_quality': final_quality,
        'flagged': flagged,
        'flag_reason': flag_reason,
        'n_iterations': len(iterations),
        'iterations': iterations,
        'conversation': messages,
        'method': 'llm_vision',
        'time_sec': time.time() - start_time,
    }


# =============================================================================
# LINE PROCESSING
# =============================================================================

MAX_LINE_RETRIES = 3


def process_line(args) -> dict:
    """Process a single spectral line with retry logic."""
    spectrum_file, line_wave, idx, total, output_dir = args
    start = time.time()

    direct_result = direct_fit(spectrum_file, line_wave)

    d_ew = direct_result.get('measured_ew')
    d_qual = direct_result.get('fit_quality', '?')
    needs_llm = direct_result.get('needs_improvement', False)
    reason = direct_result.get('improvement_reason', '')

    # Explain why LLM is being invoked
    reason_explanations = {
        'poor_quality': 'RMS>2.5σ, LLM will try to improve continuum/blends',
        'crowded_region': '≥10 lines detected, LLM will check for missed blends',
        'elevated_chi2': 'χ²>15, LLM will examine residual patterns',
        'elevated_central_rms': 'central RMS>2.5σ, LLM will refine target line fit',
        'fit_failed': 'fitting failed, LLM will try alternative approach',
    }
    # Handle residual_slope_X.XX format
    if reason and reason.startswith('residual_slope_'):
        slope_val = reason.replace('residual_slope_', '')
        reason_text = f'tilted residuals ({slope_val}), LLM will adjust continuum'
    else:
        reason_text = reason_explanations.get(reason, reason) if reason else 'needs review'
    
    if needs_llm:
        print(f"    [{idx+1}/{total}] {line_wave:.2f}Å: Direct={d_ew:.1f}mÅ → LLM review ({reason_text})", flush=True)

    result = {
        'wavelength': line_wave,
        'direct_ew': d_ew,
        'direct_quality': d_qual,
        'direct_voigt_params': direct_result.get('voigt_params'),
        'region_info': direct_result.get('region_info'),
        'continuum_info': direct_result.get('continuum_info'),
    }

    if direct_result.get('flagged'):
        result.update({
            'success': False,
            'flagged': True,
            'flag_reason': direct_result.get('flag_reason', 'no_data'),
            'method': 'direct',
            'used_llm': False,
            'time_sec': time.time() - start,
        })
        save_line_plot(spectrum_file, result, output_dir, 'flagged')
        return result

    if direct_result.get('success') and not needs_llm:
        result.update({
            'success': True,
            'measured_ew': d_ew,
            'ew_err': direct_result['ew_err'],
            'fit_quality': d_qual,
            'method': 'direct',
            'used_llm': False,
            'flagged': False,
            'time_sec': time.time() - start,
        })
        save_line_plot(spectrum_file, result, output_dir, 'direct')
        return result

    # LLM processing with retry
    llm_result = None
    llm_attempts = 0
    flag_retries = 0
    
    while True:
        llm_attempts += 1
        llm_result = llm_measure_with_vision(spectrum_file, line_wave, direct_result, output_dir)
        
        if llm_result.get('success') and not llm_result.get('flagged'):
            break
        
        if llm_result.get('flagged'):
            flag_retries += 1
            if flag_retries < MAX_LINE_RETRIES:
                direct_result = direct_fit(spectrum_file, line_wave, add_noise=True)
                print(f"    🔄 Flag retry {flag_retries}/{MAX_LINE_RETRIES}...", flush=True)
                continue
            break
        
        error_str = str(llm_result.get('error', '')).lower()
        is_retryable = any(x in error_str for x in ['timeout', 'rate', '429', 'retry'])
        
        if not is_retryable or llm_attempts >= MAX_LINE_RETRIES:
            break
        
        wait_time = llm_attempts * 5
        print(f"    ⚠️ Retry {llm_attempts+1}/{MAX_LINE_RETRIES} after {wait_time}s...", flush=True)
        time.sleep(wait_time)

    if not llm_result.get('success') and not llm_result.get('flagged'):
        result.update({
            'success': False,
            'llm_timeout': True,
            'llm_error': llm_result.get('error'),
            'method': 'llm_timeout',
            'used_llm': True,
            'flagged': False,
            'time_sec': time.time() - start,
        })
        save_line_plot(spectrum_file, result, output_dir, 'error')
        return result

    final_region = result.get('region_info')
    final_continuum = result.get('continuum_info')
    if llm_result.get('iterations'):
        last_iter = llm_result['iterations'][-1]
        if last_iter.get('region_info'):
            final_region = last_iter['region_info']
        if last_iter.get('continuum_info'):
            final_continuum = last_iter['continuum_info']

    result.update({
        'success': llm_result.get('success', False),
        'measured_ew': llm_result.get('measured_ew'),
        'ew_err': llm_result.get('ew_err'),
        'fit_quality': llm_result.get('fit_quality'),
        'flagged': llm_result.get('flagged', False),
        'flag_reason': llm_result.get('flag_reason'),
        'method': llm_result.get('method', 'llm'),
        'n_iterations': llm_result.get('n_iterations', 0),
        'iterations': llm_result.get('iterations', []),
        'used_llm': True,
        'time_sec': time.time() - start,
        'region_info': final_region,
        'continuum_info': final_continuum,
    })

    # Auto-flag high RMS
    if not result.get('flagged') and result.get('iterations'):
        final_rms = result['iterations'][-1].get('rms', 0)
        if final_rms > 3.0:
            result['flagged'] = True
            result['flag_reason'] = f'auto_high_rms_{final_rms:.1f}σ'

    subdir = 'flagged' if result.get('flagged') else 'llm'
    save_line_plot(spectrum_file, result, output_dir, subdir)

    return result


def save_line_plot(spectrum_file: str, result: dict, output_dir: str, subdir: str):
    """Save diagnostic plot for a line."""
    from ew_tools import load_spectrum, extract_region, fit_ew, get_fit_plot, _get_session
    from scipy.special import voigt_profile
    
    wave = result.get('wavelength')
    if not wave or not output_dir:
        return
    
    try:
        load_spectrum(spectrum_file)
        region_info = result.get('region_info') or {}
        window = region_info.get('window', 3.0)
        extract_region(wave, window=window)
        
        # Get voigt params
        if result.get('used_llm') and result.get('iterations'):
            all_lines = result['iterations'][-1].get('all_lines', [])
        else:
            all_lines = (result.get('direct_voigt_params') or {}).get('all_lines', [])
        
        session = _get_session()
        region = session['current_region']
        w = region['wave']
        f = region['flux']
        f_err = region['flux_err']
        
        # Normalize
        threshold = np.percentile(f, 85)
        cont_mask = f >= threshold
        if np.sum(cont_mask) >= 2:
            coef = np.polyfit(w[cont_mask], f[cont_mask], 1)
            continuum = np.polyval(coef, w)
        else:
            continuum = np.percentile(f, 95)
        f_norm = f / continuum
        f_norm_err = f_err / continuum
        
        # Build model
        model = np.ones_like(w)
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
        
        residuals = f_norm - model
        residuals_norm = residuals / f_norm_err if np.any(f_norm_err > 0) else residuals / 0.01
        rms = float(np.std(residuals_norm))
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[3, 1], sharex=True)
        
        ax1.plot(w, f_norm, 'k-', lw=1.0, alpha=0.9, label='Data')
        ax1.axhline(1, color='gray', ls=':', alpha=0.5)
        ax1.axvline(wave, color='blue', ls=':', lw=2, alpha=0.7, label='Target')
        
        if all_lines:
            ax1.plot(w, model, 'r-', lw=1.5, label='Fit')
            for line in all_lines:
                color = 'green' if abs(line['center'] - wave) < 0.3 else 'orange'
                ax1.axvline(line['center'], color=color, ls='--', alpha=0.5, lw=1)
        
        meas_ew = result.get('measured_ew', 0)
        flagged = result.get('flagged', False)
        used_llm = result.get('used_llm', False)
        status = "FLAGGED" if flagged else ("LLM" if used_llm else "DIRECT")
        ax1.set_title(f'{wave:.2f} Å | {status} | EW={meas_ew:.1f} mÅ', fontsize=11,
                     weight='bold', color='red' if flagged else 'black')
        ax1.set_ylabel('Normalized Flux')
        ax1.set_ylim(0.3, 1.15)
        ax1.legend(loc='lower right', fontsize=9)
        
        ax2.axhspan(-1, 1, alpha=0.2, color='lightgreen')
        ax2.axhspan(-2, 2, alpha=0.1, color='lightyellow')
        ax2.plot(w, residuals_norm, 'k-', lw=0.8)
        ax2.axhline(0, color='gray', ls='-', alpha=0.5)
        ax2.set_xlabel('Wavelength (Å)')
        ax2.set_ylabel('Residuals (σ)')
        ax2.set_ylim(-4, 4)
        ax2.text(0.02, 0.95, f'RMS={rms:.2f}σ', transform=ax2.transAxes, fontsize=10, va='top')
        
        fig.tight_layout()
        
        out_path = Path(output_dir) / 'fits' / subdir
        out_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path / f'{wave:.2f}.png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        
    except Exception as e:
        pass


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_ew_analysis(
    spectrum_file: str,
    linelist_file: str,
    n_workers: int = None,
    output_dir: str = None,
    clean_plots: bool = True,
):
    """
    Run complete EW analysis for all lines in the line list.

    Args:
        spectrum_file: Path to spectrum CSV (wavelength, flux, flux_error)
        linelist_file: Path to line list CSV (wavelength column)
        n_workers: Number of parallel workers (default: 10)
        output_dir: Output directory (default: ~/Egent_output)
        clean_plots: Remove temporary LLM working plots after completion (default: True)

    Returns:
        List of result dictionaries
    """
    from ew_tools import load_spectrum

    config = get_config()
    config.validate()

    # Load and validate spectrum
    spec_result = load_spectrum(spectrum_file)
    if not spec_result['success']:
        print(f"Error: {spec_result.get('error')}")
        return []
    
    print(f"Spectrum: {spectrum_file}")
    print(f"  {spec_result['n_points']} points, {spec_result['wavelength_range'][0]:.1f}-{spec_result['wavelength_range'][1]:.1f} Å")
    print(f"  Median SNR: {spec_result['median_snr']:.0f}")

    # Load line list
    linelist_path = Path(linelist_file)
    if not linelist_path.exists():
        print(f"Error: Line list not found: {linelist_file}")
        return []
    
    lines_df = pd.read_csv(linelist_path)
    if 'wavelength' not in lines_df.columns:
        # Try first column
        lines_df.columns = ['wavelength'] + list(lines_df.columns[1:])
    
    line_waves = lines_df['wavelength'].dropna().values.tolist()
    print(f"\nLine list: {linelist_file}")
    print(f"  {len(line_waves)} lines to measure")

    # Output directory
    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = config.output_dir
    out_dir.mkdir(exist_ok=True, parents=True)
    out_dir_str = str(out_dir)

    # Workers
    if n_workers is None:
        n_workers = config.default_workers

    # Build argument list
    args_list = [(spectrum_file, wave, i, len(line_waves), out_dir_str)
                 for i, wave in enumerate(line_waves)]

    # Header
    print(f"\n{'='*70}")
    print(f"EGENT EW ANALYSIS: {len(line_waves)} lines")
    print(f"Model: {config.model_id} | Workers: {n_workers}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    results = []
    stats = {'direct': 0, 'llm': 0, 'flagged': 0, 'timeout': 0, 'failed': 0}

    # Process lines in parallel
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for args in args_list:
            time.sleep(0.05)
            futures[executor.submit(process_line, args)] = args[1]

        for i, future in enumerate(as_completed(futures)):
            wave = futures[future]
            try:
                result = future.result()
                results.append(result)

                if result.get('flagged'):
                    stats['flagged'] += 1
                    symbol = '🚩'
                elif result.get('llm_timeout'):
                    stats['timeout'] += 1
                    symbol = '⏱️'
                elif not result.get('success'):
                    stats['failed'] += 1
                    symbol = '✗'
                elif not result.get('used_llm'):
                    stats['direct'] += 1
                    symbol = '✓'
                else:
                    stats['llm'] += 1
                    symbol = '🔄'

                ew = result.get('measured_ew')
                t = result.get('time_sec', 0)

                if result.get('flagged'):
                    status = f"{symbol} FLAGGED ({result.get('flag_reason', '?')}) {t:.1f}s"
                elif result.get('llm_timeout'):
                    status = f"{symbol} TIMEOUT {t:.1f}s"
                elif ew:
                    status = f"{symbol} {ew:.1f}±{result.get('ew_err', 0):.1f} mÅ {t:.1f}s"
                else:
                    status = f"{symbol} failed {t:.1f}s"

                print(f"[{i+1:3d}/{len(line_waves)}] {wave:8.2f} Å | {status}", flush=True)

            except Exception as e:
                print(f"[{i+1:3d}/{len(line_waves)}] {wave:8.2f} Å | ERROR: {e}", flush=True)
                results.append({'wavelength': wave, 'success': False, 'error': str(e)})
                stats['failed'] += 1

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"  Direct accepted: {stats['direct']} (no LLM needed)")
    print(f"  LLM refined: {stats['llm']}")
    print(f"  Flagged: {stats['flagged']}")
    print(f"  Timeout: {stats['timeout']}")
    print(f"  Failed: {stats['failed']}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = out_dir / f"results_{timestamp}.json"

    output = {
        'metadata': {
            'spectrum_file': spectrum_file,
            'linelist_file': linelist_file,
            'timestamp': timestamp,
            'n_lines': len(line_waves),
            'n_workers': n_workers,
            'model': config.model_id,
        },
        'summary': stats,
        'results': results,
    }

    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved: {out_file}")
    print(f"Plots saved: {out_dir}/fits/")

    # Clean up temporary plots directory if requested
    if clean_plots:
        plots_dir = Path(__file__).parent / 'plots'
        if plots_dir.exists():
            import shutil
            shutil.rmtree(plots_dir)
            print(f"Cleaned up temporary plots directory")

    print(f"{'='*70}")

    return results


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Egent: LLM-Powered Equivalent Width Measurement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_ew.py --spectrum spectrum.csv --lines linelist.csv
    python run_ew.py --spectrum spectrum.csv --lines linelist.csv --workers 5
    python run_ew.py --spectrum spectrum.csv --lines linelist.csv --output-dir ./output

Input Files:
    spectrum.csv: Must have columns: wavelength, flux, flux_error
    linelist.csv: Must have column: wavelength (or first column is wavelength)

Environment:
    OPENAI_API_KEY: Your OpenAI API key (required)
    EGENT_MODEL: Model to use (default: gpt-5-mini)
        """
    )
    parser.add_argument("--spectrum", type=str, required=True,
                        help="Path to spectrum CSV file")
    parser.add_argument("--lines", type=str, required=True,
                        help="Path to line list CSV file")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: 10)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: ~/Egent_output)")
    parser.add_argument("--keep-plots", action="store_true",
                        help="Keep temporary LLM working plots (default: clean up)")

    args = parser.parse_args()

    run_ew_analysis(
        spectrum_file=args.spectrum,
        linelist_file=args.lines,
        n_workers=args.workers,
        output_dir=args.output_dir,
        clean_plots=not args.keep_plots,
    )
