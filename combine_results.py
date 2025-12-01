#!/usr/bin/env python3
"""
Combine Results: Merge individual JSON result files into combined format.

Creates a single combined_results.json that:
- Aggregates all individual result files
- Adds gaia_id to each result for identification
- Computes summary statistics
- Compatible with manuscript notebook

Usage:
    python combine_results.py                    # Combine mini results
    python combine_results.py --full             # Combine full model results
    python combine_results.py --output-dir PATH  # Custom output directory
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import linregress


def load_catalog_mapping():
    """Load C3PO catalog and Gaia mapping."""
    data_dir = Path(__file__).parent / 'data'
    
    c3po_ew = pd.read_csv(data_dir / 'c3po_equivalent_widths.csv')
    mapping_df = pd.read_csv(data_dir / 'spectra_good_calibration.csv')
    
    mapping = {}
    for _, row in mapping_df.iterrows():
        gaia_str = str(row['gaia_id'])
        mapping[gaia_str] = {
            'pair': int(row['pair_id']),
            'component': row['component'].upper()
        }
    
    return c3po_ew, mapping


def calculate_r2(results, gaia_id, c3po_ew, mapping):
    """Calculate R² for a spectrum vs catalog."""
    if gaia_id not in mapping:
        return None, None
    
    m = mapping[gaia_id]
    c3po_col = f"EW_Pair_{m['pair']}{m['component']}"
    
    if c3po_col not in c3po_ew.columns:
        return None, None
    
    egent_ews, c3po_ews = [], []
    for r in results:
        if not r.get('success') or r.get('flagged'):
            continue
        
        line_wave = r.get('line', 0)
        egent_ew = r.get('measured_ew') or 0
        if egent_ew <= 0:
            continue
        
        c3po_match = c3po_ew[abs(c3po_ew['Wavelength'] - line_wave) < 0.1]
        if len(c3po_match) == 0:
            continue
        c3po_val = c3po_match[c3po_col].values[0]
        if pd.isna(c3po_val) or c3po_val <= 0:
            continue
        
        egent_ews.append(egent_ew)
        c3po_ews.append(c3po_val)
    
    if len(egent_ews) >= 20:
        slope, _, r_value, _, _ = linregress(c3po_ews, egent_ews)
        return r_value ** 2, slope
    
    return None, None


def combine_results(output_dir: Path, min_r2: float = 0.8):
    """
    Combine individual result files into combined format.
    
    Args:
        output_dir: Directory containing individual result JSON files
        min_r2: Minimum R² to include spectrum (default 0.8)
    
    Returns:
        Combined data dictionary
    """
    c3po_ew, mapping = load_catalog_mapping()
    
    result_files = sorted(output_dir.glob('results_gaia*.json'))
    print(f"Found {len(result_files)} result files in {output_dir}")
    
    all_results = []
    spectra_meta = {}
    excluded = []
    
    # Stats counters
    n_direct = 0
    n_llm = 0
    n_flagged = 0
    n_accepted = 0
    
    for jf in result_files:
        with open(jf) as f:
            data = json.load(f)
        
        gaia_id = str(data['metadata']['gaia_id'])
        
        # Calculate R² to check quality
        r2, slope = calculate_r2(data['results'], gaia_id, c3po_ew, mapping)
        
        # Exclude if not in mapping OR R² too low
        if r2 is None:
            excluded.append((gaia_id, None))
            print(f"  Excluding {gaia_id}: Not in catalog mapping")
            continue
        
        if r2 < min_r2:
            excluded.append((gaia_id, r2))
            print(f"  Excluding {gaia_id}: R²={r2:.3f} < {min_r2}")
            continue
        
        # Store spectrum metadata
        spectra_meta[gaia_id] = {
            'pair_id': data['metadata'].get('pair_id'),
            'component': data['metadata'].get('component'),
            'n_lines': len(data['results']),
            'r2': r2,
            'slope': slope
        }
        
        # Add gaia_id and posthoc_flagged to each result
        for r in data['results']:
            r['gaia_id'] = gaia_id
            r['posthoc_flagged'] = False  # No longer used, kept for compatibility
            
            # Count stats
            if r.get('flagged'):
                n_flagged += 1
            elif r.get('success'):
                n_accepted += 1
                if r.get('used_llm'):
                    n_llm += 1
                else:
                    n_direct += 1
            
            all_results.append(r)
    
    n_lines = len(all_results)
    n_spectra = len(spectra_meta)
    
    combined = {
        'metadata': {
            'name': f'Combined results from {output_dir.name}',
            'created': datetime.now().isoformat(),
            'n_spectra': n_spectra,
            'n_lines': n_lines,
            'n_direct': n_direct,
            'n_llm': n_llm,
            'n_flagged_fit': n_flagged,
            'n_posthoc_flagged': 0,  # No longer used
            'n_accepted': n_accepted,
            'n_final': n_accepted,
            'pct_direct': 100 * n_direct / n_lines if n_lines else 0,
            'pct_llm': 100 * n_llm / n_lines if n_lines else 0,
            'pct_flagged_fit': 100 * n_flagged / n_lines if n_lines else 0,
            'pct_posthoc': 0,
            'pct_final': 100 * n_accepted / n_lines if n_lines else 0,
            'spectra': spectra_meta,
            'excluded_low_r2': excluded
        },
        'results': all_results
    }
    
    return combined


def main():
    parser = argparse.ArgumentParser(description="Combine individual result files")
    parser.add_argument("--full", action="store_true", help="Use full model output dir")
    parser.add_argument("--output-dir", type=str, help="Custom output directory")
    parser.add_argument("--min-r2", type=float, default=0.8, help="Minimum R² threshold")
    args = parser.parse_args()
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.full:
        output_dir = Path.home() / 'Egent_output'
    else:
        output_dir = Path.home() / 'Egent_output_mini'
    
    if not output_dir.exists():
        print(f"Error: {output_dir} does not exist")
        return
    
    print(f"\n{'='*70}")
    print(f"COMBINING RESULTS")
    print(f"{'='*70}")
    print(f"Source: {output_dir}")
    print(f"Min R²: {args.min_r2}")
    
    combined = combine_results(output_dir, min_r2=args.min_r2)
    
    # Save combined file
    out_file = output_dir / 'combined_results.json'
    with open(out_file, 'w') as f:
        json.dump(combined, f, indent=2)
    
    meta = combined['metadata']
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Spectra: {meta['n_spectra']}")
    print(f"Lines: {meta['n_lines']}")
    print(f"Direct accepted: {meta['n_direct']} ({meta['pct_direct']:.1f}%)")
    print(f"LLM refined: {meta['n_llm']} ({meta['pct_llm']:.1f}%)")
    print(f"Flagged: {meta['n_flagged_fit']} ({meta['pct_flagged_fit']:.1f}%)")
    print(f"Final accepted: {meta['n_accepted']} ({meta['pct_final']:.1f}%)")
    
    if meta['excluded_low_r2']:
        print(f"\nExcluded (low R²): {len(meta['excluded_low_r2'])}")
    
    print(f"\nSaved: {out_file}")
    print(f"Size: {out_file.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()

