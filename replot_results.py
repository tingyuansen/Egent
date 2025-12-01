#!/usr/bin/env python3
"""
Replot Results from JSON
=========================

Regenerate plots from saved JSON results.

Usage:
    python replot_results.py results_gaia*.json                    # Comparison plot
    python replot_results.py results_gaia*.json --all-lines        # All line plots
    python replot_results.py results_gaia*.json --line 5380.34     # Single line
"""

import json
import argparse
from pathlib import Path

from utils import generate_comparison_plot, plot_single_line


def load_results(json_file: str) -> dict:
    """Load results from JSON file."""
    with open(json_file) as f:
        return json.load(f)


def plot_all_lines(data: dict, output_dir: Path):
    """Plot all lines from results using stored parameters."""
    from ew_tools import load_spectrum, _get_session
    
    gaia_id = data['metadata']['gaia_id']
    results = data['results']
    cal_info = data['metadata'].get('calibration') or {}
    emp_corr = cal_info.get('empirical_correction', 0) if cal_info else 0
    custom_spectrum = data['metadata'].get('custom_spectrum')
    
    output_dir.mkdir(exist_ok=True)
    
    # Pre-load spectrum once
    print(f"Loading spectrum for Gaia {gaia_id}...")
    load_spectrum(gaia_id, custom_file=custom_spectrum, empirical_correction=emp_corr)
    session = _get_session()
    spec = session['spectrum']
    wave_col = session.get('wave_col', 'wavelength')
    
    if wave_col not in spec.columns:
        wave_col = 'wavelength'
    
    spectrum_data = {'df': spec, 'wave_col': wave_col}
    
    print(f"Plotting {len(results)} lines...")
    counts = {'direct': 0, 'llm': 0, 'flagged': 0}
    
    for i, r in enumerate(results):
        if r.get('line'):
            plot_single_line(gaia_id, r, output_dir, spectrum_data=spectrum_data)
            
            if r.get('flagged') or r.get('posthoc_flagged'):
                counts['flagged'] += 1
            elif r.get('used_llm'):
                counts['llm'] += 1
            else:
                counts['direct'] += 1
            
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(results)} done...")
    
    print(f"Done: {counts['direct']} direct, {counts['llm']} LLM, {counts['flagged']} flagged")


def main():
    parser = argparse.ArgumentParser(description="Replot EW results from JSON")
    parser.add_argument("json_file", help="Path to results JSON file")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory")
    parser.add_argument("--all-lines", action="store_true", help="Plot all individual lines")
    parser.add_argument("--line", type=float, default=None, help="Plot specific line wavelength")
    
    args = parser.parse_args()
    
    data = load_results(args.json_file)
    gaia_id = data['metadata']['gaia_id']
    timestamp = data['metadata']['timestamp']
    
    print(f"Loaded results for Gaia {gaia_id}")
    print(f"  {len(data['results'])} lines, timestamp: {timestamp}")
    
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.json_file).parent
    out_dir.mkdir(exist_ok=True)
    
    if args.line:
        line_result = next((r for r in data['results'] if abs(r['line'] - args.line) < 0.1), None)
        if line_result:
            plot_single_line(gaia_id, line_result, out_dir)
            print(f"  ✓ Plotted line {args.line}")
        else:
            print(f"Line {args.line} not found in results")
    elif args.all_lines:
        plot_all_lines(data, out_dir / f'gaia{gaia_id}_fits')
    else:
        generate_comparison_plot(data['results'], gaia_id, timestamp, out_dir, prefix=f"gaia{gaia_id}_")
        print(f"  ✓ Comparison plot saved")


if __name__ == "__main__":
    main()

