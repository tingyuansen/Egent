#!/usr/bin/env python3
"""
Egent EW Analysis Pipeline
==========================

Autonomous equivalent width (EW) measurement using LLM-guided Voigt fitting.

This script implements a two-stage EW measurement pipeline:

STAGE 1: Direct Voigt Fitting 
    - Fast deterministic fitting with optimized continuum settings
    - Automatically accepts fits that pass quality thresholds
    - Flags obvious failures (no data, severe blends)

STAGE 2: LLM Visual Inspection 
    - Only invoked for borderline cases where direct fit is uncertain
    - Uses vision capability to visually inspect fit plots
    - Can adjust continuum, window size, add peaks, or flag as unreliable
    - Includes quality gate: must verify fit is acceptable before recording

Usage:
    # Just provide Gaia ID (pair info auto-detected)
    python run_ew.py --gaia-id 4287378367995943680

    # Using mini model (faster, cheaper)
    python run_ew.py --gaia-id 4287378367995943680 --mini

Author: Egent Framework
"""

import json
import time
import base64
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import voigt_profile

from config import get_config
from llm_client import LLMClient


# =============================================================================
# TOOL DEFINITIONS FOR LLM
# =============================================================================

TOOLS = [
    {"type": "function", "function": {
        "name": "load_spectrum",
        "description": "Load spectrum by Gaia ID with barycentric correction applied",
        "parameters": {"type": "object", "properties": {
            "gaia_id": {"type": "integer", "description": "Gaia DR3 source ID"}
        }, "required": ["gaia_id"]}}},

    {"type": "function", "function": {
        "name": "extract_region",
        "description": "Extract spectral region around target line. Use window=5.0 for crowded regions, window=2.0 for isolated weak lines.",
        "parameters": {"type": "object", "properties": {
            "line_wavelength": {"type": "number", "description": "Target wavelength in Angstroms"},
            "window": {"type": "number", "description": "Half-width in Angstroms (default 3.0)"}
        }, "required": ["line_wavelength"]}}},

    {"type": "function", "function": {
        "name": "set_continuum_method",
        "description": "Configure continuum fitting method. Options: 'iterative_linear' (default), 'iterative_poly' (order=2 for curved continuum), 'top_percentile' (use top 85-95% for crowded regions).",
        "parameters": {"type": "object", "properties": {
            "method": {"type": "string", "enum": ["iterative_linear", "iterative_poly", "top_percentile"]},
            "order": {"type": "integer", "description": "Polynomial order (for iterative_poly)"},
            "top_percentile": {"type": "number", "description": "Use top X% of flux values (default 85)"}
        }, "required": ["method"]}}},

    {"type": "function", "function": {
        "name": "set_continuum_regions",
        "description": "Manually specify wavelength regions for continuum fitting. Use when automatic detection fails in crowded regions. Example: [[5700, 5701], [5704, 5705]]",
        "parameters": {"type": "object", "properties": {
            "regions": {"type": "array", "items": {"type": "array", "items": {"type": "number"}},
                       "description": "List of [start, end] wavelength pairs"}
        }, "required": ["regions"]}}},

    {"type": "function", "function": {
        "name": "fit_ew",
        "description": "Fit multi-Voigt model to measure EW. Returns diagnostics: fit_quality, quality_issues, correlated_residuals. If residuals show W-shaped patterns (indicating missed blends), use additional_peaks to add Voigt components.",
        "parameters": {"type": "object", "properties": {
            "additional_peaks": {"type": "array", "items": {"type": "number"},
                               "description": "Wavelengths to add as extra Voigt components for missed blends."}
        }, "required": []}}},

    {"type": "function", "function": {
        "name": "get_fit_plot",
        "description": "Generate diagnostic plot for visual inspection. Shows data, fit, residuals, and line identification.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},

    {"type": "function", "function": {
        "name": "flag_line",
        "description": "Flag line as unreliable. Reasons: no_data, severe_blend, fit_failed, bad_continuum, wrong_line",
        "parameters": {"type": "object", "properties": {
            "line_wavelength": {"type": "number"},
            "reason": {"type": "string", "enum": ["no_data", "severe_blend", "fit_failed", "bad_continuum", "wrong_line"]}
        }, "required": ["line_wavelength", "reason"]}}},

    {"type": "function", "function": {
        "name": "record_measurement",
        "description": "Record final EW measurement with uncertainty and quality assessment",
        "parameters": {"type": "object", "properties": {
            "line_wavelength": {"type": "number"},
            "ew_mA": {"type": "number", "description": "Equivalent width in milli-Angstroms"},
            "ew_err_mA": {"type": "number", "description": "Uncertainty in milli-Angstroms"},
            "quality": {"type": "string", "description": "good, acceptable, or poor"}
        }, "required": ["line_wavelength", "ew_mA", "ew_err_mA", "quality"]}}},
]


# =============================================================================
# LLM SYSTEM PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are an expert reviewer for stellar EW measurements. You're called for BORDERLINE cases where automated fitting is uncertain.

YOUR ROLE: Use visual inspection to make judgment calls that algorithms can't.

WORKFLOW:
1. The fit is already done. Call get_fit_plot to VISUALLY INSPECT.
2. Make a judgment: ACCEPT, IMPROVE, or FLAG.

=== ⚠️ CRITICAL: FOCUS ONLY ON TARGET REGION ⚠️ ===

The TARGET LINE is marked with a BLUE DASHED vertical line.
ONLY care about fit quality within ±0.5Å of this target.

**EDGE vs TARGET distinction:**
- IGNORE: Spikes/noise at window edges (>1.5Å from target) - these are normal
- CHECK: Residuals within ±0.5Å of the blue target line
- If target region is clean but edges are messy → ACCEPT
- If target region has W-pattern or blend → FIX or FLAG

=== RMS INTERPRETATION ===

**RMS < 1.5σ: LIKELY GOOD but still check target region**
  - Low RMS is good, but STILL look for W-shaped residuals at target
  - If residuals at target ±0.5Å are flat → ACCEPT
  - If W-pattern at target (dip-peak-dip) → add_blend or FLAG

**RMS 1.5-2.0σ: GOOD → Check target region only**
  - If target ±0.5Å looks clean → ACCEPT
  - Only flag if there's a problem AT the target line itself

**RMS 2.0-2.5σ: MARGINAL → Try to improve**
  - Try polynomial continuum or adding peaks
  - Accept if target region is clean after improvement

**RMS > 2.5σ: POOR → Needs work or FLAG**
  - Try improvements first
  - Flag only if cannot improve

=== ⚠️ BLEND DETECTION (CHECK THIS EVEN WITH LOW RMS) ===

Look for these patterns in residuals NEAR THE TARGET (±1Å):

1. **W-SHAPED residuals** at target = MISSED BLEND
   - dip, peak, dip pattern = two lines overlapping
   - Fix: fit_ew(additional_peaks=[wavelength_of_blend])

2. **ASYMMETRIC wing/shoulder** on target line
   - One side of absorption drops faster than model
   - Look for extra dip adjacent to target (±0.5Å)
   - Fix: add the wing as additional_peak

3. **Systematic offset** at target center
   - Residuals consistently positive or negative at target
   - May need to adjust continuum or add nearby blend

Even if overall RMS is low, these patterns at target = bad fit → FIX or FLAG

=== DECISIONS ===

**ACCEPT** (call record_measurement) if:
- RMS < 1.5σ (excellent fit - accept without hesitation)
- OR: RMS < 2.0σ AND target region residuals within ±2σ
- Red fit follows black data at the TARGET wavelength

**IMPROVE** by trying:
1. Add missing peaks near target: fit_ew(additional_peaks=[...])
2. Polynomial continuum: set_continuum_method('iterative_poly', order=2)
3. Window adjustment: extract_region with window=2.0 or 5.0

**FLAG** (use sparingly!) only when:
- Problem is AT THE TARGET (±0.5Å), not at edges
- RMS > 2.5σ that cannot be improved after trying fixes
- Target line is completely blended or absent
- W-pattern at target that cannot be resolved with additional_peaks

**DO NOT FLAG** for:
- Edge spikes/noise far from target (>1.5Å away)
- Low RMS (<2.0σ) with clean target region
- Fit looks good visually (trust the fit, not external references)"""


# =============================================================================
# GLOBAL STATE
# =============================================================================

_CUSTOM_SPECTRUM_FILE = None


# =============================================================================
# FITTING FUNCTIONS
# =============================================================================

def direct_fit(gaia_id: int, line_wave: float, add_noise: bool = False) -> dict:
    """
    Optimized direct Voigt fitting without LLM intervention.
    
    Args:
        gaia_id: Gaia source ID
        line_wave: Target wavelength
        add_noise: If True, add stochasticity to fitting parameters for retry diversity
    """
    import random
    from ew_tools import load_spectrum, extract_region, set_continuum_method, fit_ew, _get_session

    load_spectrum(gaia_id, custom_file=_CUSTOM_SPECTRUM_FILE)
    
    # Add stochasticity for retry diversity
    if add_noise:
        window = 3.0 + random.uniform(-0.5, 0.5)  # 2.5-3.5 Å
        top_percentile = 85 + random.randint(-5, 5)  # 80-90%
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
        offset = result['target_line'].get('distance_from_target', 0)
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

        resid_slope_norm = result.get('diagnostics', {}).get('resid_slope_norm', 0)
        if abs(resid_slope_norm) > 0.5:
            needs_improvement = True
            improvement_reason = f'residual_slope_{resid_slope_norm:.2f}_sigma_per_A'

        all_lines = result.get('diagnostics', {}).get('all_fitted_lines', [])
        target_line = result.get('target_line', {})

        session = _get_session()
        region = session.get('current_region', {})
        last_fit = session.get('last_fit', {})

        popt = last_fit.get('popt', [])
        continuum_method = session.get('continuum_method', 'iterative_linear')
        continuum_order = session.get('continuum_order', 1)
        n_cont_params = continuum_order + 1
        continuum_coeffs = popt[:n_cont_params] if len(popt) >= n_cont_params else [1.0] + [0.0] * continuum_order

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
            'wavelength_offset': offset,
            'quality_issues': result.get('quality_issues', []),
            'region_info': {
                'echelle_order': region.get('order'),
                'window': region.get('window', 3.0),
                'wave_range': [region.get('wave', [0, 0])[0], region.get('wave', [0, 0])[-1]] if region.get('wave') is not None else None,
            },
            'continuum_info': {
                'method': continuum_method,
                'order': continuum_order,
                'coeffs': continuum_coeffs,
                'init_continuum_coef': result.get('diagnostics', {}).get('continuum', {}).get('init_continuum_coef'),
                'wave_center': result.get('diagnostics', {}).get('continuum', {}).get('wave_center'),
            },
            'voigt_params': {
                'target': {
                    'center': target_line.get('center'),
                    'amplitude': target_line.get('amplitude'),
                    'sigma': target_line.get('sigma'),
                    'gamma': target_line.get('gamma'),
                    'fwhm': target_line.get('fwhm'),
                    'ew_mA': target_line.get('ew_mA'),
                } if target_line else None,
                'all_lines': all_lines,
            },
        }

    return {'success': False, 'needs_improvement': True, 'improvement_reason': 'fit_failed'}


def llm_measure_with_vision(
    gaia_id: int,
    line_wave: float,
    species: float,
    direct_result: dict = None,
    output_dir: str = None,
    use_mini: bool = False
) -> dict:
    """Use LLM with vision to review and improve the EW measurement."""
    from ew_tools import (
        load_spectrum, extract_region, set_continuum_method, set_continuum_regions,
        fit_ew, get_fit_plot, flag_line, record_measurement
    )

    def execute_tool(name, inputs):
        if name == "load_spectrum":
            return load_spectrum(inputs["gaia_id"], custom_file=_CUSTOM_SPECTRUM_FILE)
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
                                      inputs["ew_err_mA"], "voigt_fit",
                                      inputs["quality"], None)
        return {"error": f"Unknown tool: {name}"}

    client = LLMClient(use_mini=use_mini)

    species_name = "Fe I" if species == 26.0 else "Fe II" if species == 26.1 else f"Species {species}"
    context = ""
    if direct_result and direct_result.get('success'):
        context = f"\nDirect fit: {direct_result['measured_ew']:.1f}±{direct_result['ew_err']:.1f} mÅ"
        context += f", RMS={direct_result.get('fit_rms', 0):.4f}, quality={direct_result['fit_quality']}"
        context += "\nPlease visually inspect the fit and improve if needed."

    prompt = f"Measure EW for {species_name} line at {line_wave:.2f} Å (Gaia ID {gaia_id}).{context}"
    messages = [{"role": "user", "content": prompt}]

    start_time = time.time()
    MAX_LLM_TIME = 300  # 5 minutes total budget (retry logic handles rate limits)
    final_ew, final_err, final_quality = None, None, None
    flagged, flag_reason = False, None
    iterations = []

    for turn in range(10):
        if time.time() - start_time > MAX_LLM_TIME:
            break

        try:
            # LLMClient has internal retry logic for rate limits/timeouts
            response = client.chat(messages, tools=TOOLS, system_prompt=SYSTEM_PROMPT, timeout=90)
        except Exception as e:
            # Only fail if the error is non-retryable AND all internal retries exhausted
            error_str = str(e).lower()
            is_fatal = not any(x in error_str for x in ['rate', 'limit', 'timeout', '429', '503', '500'])
            if is_fatal:
                return {
                    'success': False,
                    'error': str(e),
                    'method': 'llm_error',
                    'time_sec': time.time() - start_time,
                }
            # For rate limit errors that exhausted retries, return with error for outer retry
            return {
                'success': False,
                'error': f'Rate limit exhausted: {e}',
                'method': 'llm_rate_limit',
                'time_sec': time.time() - start_time,
            }

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
                    'reduced_chi2': result.get('reduced_chi2'),
                    'target_voigt': {
                        'center': target.get('center'),
                        'amplitude': target.get('amplitude'),
                        'sigma': target.get('sigma'),
                        'gamma': target.get('gamma'),
                        'fwhm': target.get('fwhm'),
                    } if target else None,
                    'all_lines': all_lines,
                    'n_lines': len(all_lines),
                    'region_info': result.get('region_info'),
                    'continuum_info': result.get('continuum_info'),
                })
                if result.get('target_line'):
                    final_ew = result['target_line']['ew_mA']
                    final_err = result['target_line']['ew_err_mA']
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

                if output_dir:
                    plot_dir = Path(output_dir) / "plots"
                else:
                    plot_dir = Path(__file__).parent / "plots"
                plot_dir.mkdir(exist_ok=True, parents=True)
                plot_file = plot_dir / f"llm_{line_wave:.2f}_{int(time.time())}.png"
                with open(plot_file, 'wb') as f:
                    f.write(base64.b64decode(image_b64))

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"Plot saved. Sending image for visual inspection..."
                })

                current_rms = iterations[-1].get('rms', 0) if iterations else 0
                rms_warning = ""
                if current_rms > 5.0:
                    rms_warning = f"\n\n⚠️ CRITICAL: Normalized RMS={current_rms:.1f} is very high (>5).\n"
                elif current_rms > 2.5:
                    rms_warning = f"\n\n⚠️ WARNING: Normalized RMS={current_rms:.1f} is elevated (>2.5).\n"

                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": "high"}},
                        {"type": "text", "text": f"""Inspect this fit carefully:{rms_warning}

**1. Does RED fit match BLACK data?**
   - Compare absorption depths
   - Check each absorption feature

**2. CHECK FOR MISSING LINES:**
   - Look for W-shaped residuals
   - Strong absorptions with no red component need additional_peaks

**3. CONTINUUM CHECK:**
   - If continuum is offset, try polynomial or wider window

**DECISIONS based on RMS:**
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

    elapsed = time.time() - start_time

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
        'time_sec': elapsed,
    }




# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

# Max retries for LLM at the line level (fresh restart each time)
MAX_LINE_RETRIES = 3


def process_line(args) -> dict:
    """Process a single spectral line with robust retry logic."""
    gaia_id, line_wave, catalog_ew, species, idx, total, output_dir, use_mini = args
    start = time.time()

    direct_result = direct_fit(gaia_id, line_wave)

    d_ew = direct_result.get('measured_ew')
    d_qual = direct_result.get('fit_quality', '?')
    needs_llm = direct_result.get('needs_improvement', False)
    reason = direct_result.get('improvement_reason', '')

    if needs_llm:
        print(f"    [{idx+1}/{total}] {line_wave:.2f}Å: Direct={d_ew:.1f}mA ({d_qual}) → LLM needed ({reason})", flush=True)

    result = {
        'line': line_wave,
        'species': species,
        'catalog_ew': catalog_ew,
        'direct_ew': d_ew,
        'direct_quality': d_qual,
        'direct_diagnostics': {
            'reduced_chi2': direct_result.get('reduced_chi2'),
            'fit_rms': direct_result.get('fit_rms'),
            'n_lines': direct_result.get('n_lines'),
            'wavelength_offset': direct_result.get('wavelength_offset'),
            'quality_issues': direct_result.get('quality_issues', []),
            'improvement_reason': direct_result.get('improvement_reason'),
        },
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
        # Save flagged plot to flagged/ subfolder
        plot_output_dir = Path(output_dir) / f'gaia{gaia_id}_fits'
        try:
            from utils import plot_line_fit
            plot_path, _ = plot_line_fit(gaia_id, result, plot_output_dir, save_file=True)
            if plot_path:
                result['plot_path'] = plot_path
        except Exception:
            pass
        return result

    if direct_result.get('success') and not needs_llm:
        diff = (d_ew / catalog_ew - 1) * 100 if catalog_ew else None

        result.update({
            'success': True,
            'measured_ew': d_ew,
            'ew_err': direct_result['ew_err'],
            'diff_pct': diff,
            'fit_quality': d_qual,
            'wavelength_offset': direct_result.get('wavelength_offset', 0),
            'method': 'direct',
            'used_llm': False,
            'flagged': False,
            'flag_reason': None,
            'time_sec': time.time() - start,
        })

        plot_output_dir = Path(output_dir) / f'gaia{gaia_id}_fits'
        try:
            from utils import plot_line_fit
            plot_path, _ = plot_line_fit(gaia_id, result, plot_output_dir, save_file=True)
            if plot_path:
                result['plot_path'] = plot_path
        except Exception:
            pass

        return result

    # LLM processing with retry logic - fresh restart on each attempt
    # Also retry when LLM wants to FLAG - give it more chances with fresh fits
    MAX_FLAG_RETRIES = 3
    llm_result = None
    llm_attempts = 0
    flag_retries = 0
    best_result = None  # Track best non-flagged result
    
    while True:
        llm_attempts += 1
        
        # Fresh LLM call (new conversation each time)
        llm_result = llm_measure_with_vision(
            gaia_id, line_wave, species, direct_result, output_dir, use_mini
        )
        
        # Track best non-flagged result
        if llm_result.get('success') and not llm_result.get('flagged'):
            if best_result is None or (llm_result.get('measured_ew') and 
                                       llm_result.get('iterations') and
                                       llm_result['iterations'][-1].get('rms', 10) < 
                                       (best_result.get('iterations', [{}])[-1].get('rms', 10) if best_result.get('iterations') else 10)):
                best_result = llm_result
            break  # Success = done
        
        # If LLM wants to FLAG, retry with fresh fitting (add stochasticity)
        if llm_result.get('flagged'):
            flag_retries += 1
            if flag_retries < MAX_FLAG_RETRIES:
                # Re-run direct fit with slight variation (uses random perturbation internally)
                direct_result = direct_fit(gaia_id, line_wave, add_noise=True)
                print(f"    🔄 Flag retry {flag_retries}/{MAX_FLAG_RETRIES} with fresh initialization...", flush=True)
                continue
            else:
                # Accept the flag after MAX_FLAG_RETRIES
                # But if we have a successful result from earlier, use that instead
                if best_result is not None:
                    print(f"    ✓ Using earlier successful fit instead of flag", flush=True)
                    llm_result = best_result
                break
        
        # Check if it's a retryable error (timeout, rate limit, etc.)
        error_str = str(llm_result.get('error', '')).lower()
        is_retryable = any(x in error_str for x in ['timeout', 'rate', '429', 'retry', 'connection'])
        
        if not is_retryable:
            # Non-retryable error (e.g., bad response format)
            break
        
        # Retryable error - wait and try again with fresh conversation
        if llm_attempts < MAX_LINE_RETRIES:
            wait_time = llm_attempts * 5  # 5s, 10s, 15s
            print(f"    ⚠️ LLM timeout/error, restarting fresh (attempt {llm_attempts+1}/{MAX_LINE_RETRIES}) after {wait_time}s...", flush=True)
            time.sleep(wait_time)
        else:
            break

    # After all retries, check result - if LLM failed, mark as timeout (don't fall back)
    if not llm_result.get('success') and not llm_result.get('flagged'):
        error_msg = llm_result.get('error', 'unknown')
        print(f"    ⏱️ LLM TIMEOUT after {llm_attempts} attempts: {error_msg[:50]}", flush=True)
        result.update({
            'success': False,
            'llm_timeout': True,  # Clear tag for exclusion from statistics
            'llm_error': error_msg,
            'llm_attempts': llm_attempts,
            'method': 'llm_timeout',
            'used_llm': True,
            'flagged': False,  # Not flagged - this is a timeout, different category
            'time_sec': time.time() - start,
        })
        # Save error plot to error/ subfolder
        plot_output_dir = Path(output_dir) / f'gaia{gaia_id}_fits'
        try:
            from utils import plot_line_fit
            plot_path, _ = plot_line_fit(gaia_id, result, plot_output_dir, save_file=True)
            if plot_path:
                result['plot_path'] = plot_path
        except Exception:
            pass
        return result

    diff = (llm_result['measured_ew'] / catalog_ew - 1) * 100 if llm_result.get('measured_ew') and catalog_ew else None

    final_region_info = result.get('region_info')
    final_continuum_info = result.get('continuum_info')
    if llm_result.get('iterations'):
        last_iter = llm_result['iterations'][-1]
        if last_iter.get('region_info'):
            final_region_info = last_iter['region_info']
        if last_iter.get('continuum_info'):
            final_continuum_info = last_iter['continuum_info']

    result.update({
        'success': llm_result.get('success', False),
        'measured_ew': llm_result.get('measured_ew'),
        'ew_err': llm_result.get('ew_err'),
        'diff_pct': diff,
        'fit_quality': llm_result.get('fit_quality'),
        'flagged': llm_result.get('flagged', False),
        'flag_reason': llm_result.get('flag_reason'),
        'method': llm_result.get('method', 'llm'),
        'n_iterations': llm_result.get('n_iterations', 0),
        'iterations': llm_result.get('iterations', []),
        'conversation': llm_result.get('conversation', []),
        'used_llm': True,
        'time_sec': time.time() - start,
        'direct_ew': d_ew,
        'direct_quality': d_qual,
        'region_info': final_region_info,
        'continuum_info': final_continuum_info,
    })

    # Get final RMS for quality decisions
    final_rms = 0
    if result.get('iterations'):
        last_iter = result['iterations'][-1]
        final_rms = last_iter.get('rms', 0)
    
    # Auto-flag based on quality metrics (safety net for LLM mistakes)
    if not result.get('flagged'):
        # Flag if RMS is too high (LLM should have flagged this)
        if final_rms > 3.0:
            result['flagged'] = True
            result['flag_reason'] = f'auto_high_rms_{final_rms:.1f}σ'
        # Note: We do NOT use catalog deviation for flagging - this is blind analysis

    plot_output_dir = Path(output_dir) / f'gaia{gaia_id}_fits'
    try:
        from utils import plot_line_fit
        plot_path, _ = plot_line_fit(gaia_id, result, plot_output_dir, save_file=True)
        if plot_path:
            result['plot_path'] = plot_path
    except Exception:
        pass

    return result


# =============================================================================
# MAIN ANALYSIS RUNNER
# =============================================================================

def run_ew_analysis(
    gaia_id: int,
    n_workers: int = None,
    light_only: bool = False,
    custom_spectrum: str = None,
    n_lines: int = None,
    output_dir: str = None,
    use_mini: bool = False,
):
    """
    Run complete EW analysis for all lines in catalog.

    Args:
        gaia_id: Gaia DR3 source ID (pair info auto-detected from calibration file)
        n_workers: Number of parallel workers
        light_only: Only analyze light elements (Z < 26)
        custom_spectrum: Path to custom spectrum file
        n_lines: Limit number of lines (for testing)
        output_dir: Custom output directory
        use_mini: Use mini model (faster, cheaper)

    Returns:
        List of result dictionaries
    """
    from ew_tools import load_spectrum
    from utils import get_pair_info_from_gaia_id

    config = get_config()

    # Look up pair info from Gaia ID
    pair_info = get_pair_info_from_gaia_id(gaia_id)
    if pair_info is None:
        print(f"Error: Gaia ID {gaia_id} not found in calibration file")
        print("Available spectra are in data/spectra_good_calibration.csv")
        return []
    
    pair_id = pair_info['pair_id']
    component = pair_info['component']
    ew_col = pair_info['ew_col']
    print(f"Gaia {gaia_id} → Pair {pair_id}{component}")

    global _CUSTOM_SPECTRUM_FILE
    _CUSTOM_SPECTRUM_FILE = custom_spectrum

    if custom_spectrum:
        print(f"Using custom spectrum: {custom_spectrum}")
    
    # Load spectrum (should be pre-corrected in spectra_corrected/)
    load_result = load_spectrum(gaia_id, custom_file=custom_spectrum)
    if not load_result['success']:
        print(f"Error: {load_result.get('error')}")
        return []
    
    print(f"Loaded spectrum: {load_result['n_points']} points, SNR={load_result['median_snr']:.0f}")
    print()

    # Load catalog
    ew_file = Path(__file__).parent / 'data' / 'c3po_equivalent_widths.csv'
    df_ews = pd.read_csv(ew_file)
    lines_df = df_ews[df_ews[ew_col].notna()].copy()

    if light_only:
        lines_df = lines_df[lines_df['Species'] < 26]

    if n_lines and n_lines < len(lines_df):
        lines_df = lines_df.head(n_lines)

    # Output directory
    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = config.get_output_dir(use_mini)
    out_dir.mkdir(exist_ok=True, parents=True)
    out_dir_str = str(out_dir)

    # Workers
    if n_workers is None:
        n_workers = config.get_workers(use_mini)

    # Build argument list
    lines = [(gaia_id, row['Wavelength'], row[ew_col], row['Species'], i, len(lines_df), out_dir_str, use_mini)
             for i, (_, row) in enumerate(lines_df.iterrows())]

    # Header
    model_name = config.get_model(use_mini)
    subset = " (light elements)" if light_only else ""
    if n_lines:
        subset += f" (limited to {n_lines})"
    print(f"{'='*70}")
    print(f"EGENT EW ANALYSIS: {len(lines)} lines{subset}")
    print(f"Backend: {config.backend.upper()} | Model: {model_name} | Workers: {n_workers}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    results = []
    stats = {'direct': 0, 'llm': 0, 'flagged': 0, 'timeout': 0, 'failed': 0}

    # Process lines in parallel
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for args in lines:
            time.sleep(0.05)
            futures[executor.submit(process_line, args)] = args[2]

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
                diff = result.get('diff_pct')
                t = result.get('time_sec', 0)

                if result.get('flagged'):
                    status = f"{symbol} FLAGGED ({result.get('flag_reason', '?')}) {t:.1f}s"
                elif result.get('llm_timeout'):
                    status = f"{symbol} TIMEOUT (exclude from stats) {t:.1f}s"
                elif ew:
                    diff_str = f"Δ={diff:+.1f}%" if diff else ""
                    status = f"{symbol} {ew:.1f}±{result.get('ew_err', 0):.1f} mÅ {diff_str} {t:.1f}s"
                else:
                    status = f"{symbol} failed {t:.1f}s"

                sp = f"Sp{result['species']:.0f}" if result['species'] < 26 else "Fe I"
                print(f"[{i+1:3d}/{len(lines)}] {wave:8.2f} Å ({sp:5s}) | {status}", flush=True)

            except Exception as e:
                print(f"[{i+1:3d}/{len(lines)}] {wave:8.2f} Å | ERROR: {e}", flush=True)
                results.append({'line': wave, 'success': False, 'error': str(e)})
                stats['failed'] += 1

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"  Direct accepted: {stats['direct']} (no LLM)")
    print(f"  LLM refined: {stats['llm']}")
    print(f"  Flagged: {stats['flagged']}")
    print(f"  Timeout: {stats['timeout']} (excluded from accuracy stats)")
    print(f"  Failed: {stats['failed']}")

    # Accuracy stats (exclude flagged AND timeouts)
    successful = [r for r in results if r.get('success') and r.get('measured_ew')
                  and not r.get('flagged') and not r.get('llm_timeout')]
    if successful:
        diffs = [abs(r['diff_pct']) for r in successful if r.get('diff_pct') is not None]
        good = sum(1 for d in diffs if d < 15)

        direct_diffs = [abs(r['direct_ew']/r['catalog_ew']-1)*100
                       for r in successful if r.get('direct_ew') and r.get('catalog_ew')]
        direct_15 = sum(1 for d in direct_diffs if d < 15) if direct_diffs else 0

        print(f"\nAccuracy vs catalog:")
        if direct_diffs:
            print(f"  Direct (no LLM): {direct_15}/{len(direct_diffs)} within 15% ({100*direct_15/len(direct_diffs):.0f}%)")
        print(f"  Final: {good}/{len(diffs)} within 15% ({100*good/len(diffs):.0f}%)")
        print(f"  Mean |diff|: {np.mean(diffs):.1f}%")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = out_dir / f"results_gaia{gaia_id}_{timestamp}.json"

    output = {
        'metadata': {
            'gaia_id': gaia_id,
            'pair_id': pair_id,
            'component': component,
            'timestamp': timestamp,
            'n_lines': len(lines),
            'n_workers': n_workers,
            'model': config.get_model(use_mini),
            'backend': config.backend,
            'custom_spectrum': custom_spectrum,
        },
        'summary': stats,
        'results': results,
    }

    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Generate comparison plot
    from utils import generate_all_plots, cleanup_temp_plots
    print(f"\nGenerating plots...")
    generate_all_plots(results, gaia_id, timestamp, out_dir)
    cleanup_temp_plots(out_dir / "plots")

    print(f"\nResults: {out_file}")
    print(f"{'='*70}")

    return results


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Egent EW Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_ew.py --gaia-id 4287378367995943680
    python run_ew.py --gaia-id 4287378367995943680 --mini
    python run_ew.py --gaia-id 4287378367995943680 --workers 20

Environment Variables:
    EGENT_BACKEND: 'openai' or 'azure' (default: 'openai')
    OPENAI_API_KEY: OpenAI API key
    AZURE_API_KEY or AZURE: Azure OpenAI API key
        """
    )
    parser.add_argument("--gaia-id", type=int, required=True,
                        help="Gaia DR3 source ID")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers")
    parser.add_argument("--light-only", action="store_true",
                        help="Only analyze light elements (Z < 26)")
    parser.add_argument("--custom-spectrum", type=str, default=None,
                        help="Path to custom spectrum file")
    parser.add_argument("--n-lines", type=int, default=None,
                        help="Limit number of lines (for testing)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Custom output directory")
    parser.add_argument("--mini", action="store_true",
                        help="Use mini model (faster, cheaper)")

    args = parser.parse_args()

    run_ew_analysis(
        gaia_id=args.gaia_id,
        n_workers=args.workers,
        light_only=args.light_only,
        custom_spectrum=args.custom_spectrum,
        n_lines=args.n_lines,
        output_dir=args.output_dir,
        use_mini=args.mini,
    )

