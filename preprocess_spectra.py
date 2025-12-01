#!/usr/bin/env python3
"""
Pre-process Spectra with Barycentric + Empirical Corrections
=============================================================

Applies wavelength corrections to spectra ONCE, saving corrected versions.
This eliminates the need for runtime barycentric/empirical corrections.

For each spectrum in the "good calibration" list:
1. Apply barycentric correction (observatory → barycentric frame)
2. Apply empirical calibration offset (fine-tuning from Fe I lines)
3. Save corrected spectrum with 'wavelength' already in rest frame

Usage:
    python preprocess_spectra.py                  # Process all good spectra
    python preprocess_spectra.py --dry-run        # Preview without saving
    python preprocess_spectra.py --gaia-id 123    # Process single spectrum
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Speed of light in km/s
C_KMS = 299792.458


def load_corrections():
    """Load barycentric and empirical corrections."""
    data_dir = Path(__file__).parent / 'data'
    
    # Barycentric corrections
    bary_file = data_dir / 'magellan_bary_corrections.csv'
    bary_df = pd.read_csv(bary_file)
    bary_corr = dict(zip(bary_df['gaia_id'], bary_df['bary_corr_kms']))
    
    # Good calibration with empirical offsets
    cal_file = data_dir / 'spectra_good_calibration.csv'
    cal_df = pd.read_csv(cal_file)
    
    return bary_corr, cal_df


def process_spectrum(gaia_id: int, bary_corr: dict, cal_row: pd.Series,
                     spectra_dir: Path, output_dir: Path, dry_run: bool = False):
    """
    Apply corrections to a single spectrum.
    
    Args:
        gaia_id: Gaia source ID
        bary_corr: Dict of barycentric corrections
        cal_row: Row from calibration dataframe with empirical offset
        spectra_dir: Directory with raw spectra
        output_dir: Directory for corrected spectra
        dry_run: If True, print info but don't save
        
    Returns:
        dict with status info
    """
    # Load raw spectrum
    raw_file = spectra_dir / f'{gaia_id}_magellan.csv'
    if not raw_file.exists():
        return {'gaia_id': gaia_id, 'status': 'missing', 'error': 'File not found'}
    
    spec = pd.read_csv(raw_file)
    
    # Get corrections
    bary_kms = bary_corr.get(gaia_id, 0.0)
    emp_kms = cal_row['offset_kms'] if pd.notna(cal_row['offset_kms']) else 0.0
    
    # Total correction
    total_corr_kms = bary_kms + emp_kms
    
    # Apply correction: λ_rest = λ_obs / (1 + v/c)
    wave_raw = spec['wavelength'].values
    wave_corrected = wave_raw / (1 + total_corr_kms / C_KMS)
    
    # Check correction magnitude
    shift_A = np.mean(wave_corrected - wave_raw)
    
    result = {
        'gaia_id': gaia_id,
        'pair_id': cal_row['pair_id'],
        'component': cal_row['component'],
        'bary_corr_kms': bary_kms,
        'emp_corr_kms': emp_kms,
        'total_corr_kms': total_corr_kms,
        'mean_shift_A': shift_A,
        'status': 'ok' if not dry_run else 'dry_run',
    }
    
    if dry_run:
        print(f"  {gaia_id}: bary={bary_kms:+.2f} + emp={emp_kms:+.2f} = {total_corr_kms:+.2f} km/s (Δλ={shift_A:+.3f} Å)")
        return result
    
    # Create corrected spectrum
    spec_out = spec.copy()
    spec_out['wavelength'] = wave_corrected
    
    # Remove old helio column if exists (no longer needed)
    if 'wavelength_helio' in spec_out.columns:
        spec_out = spec_out.drop(columns=['wavelength_helio'])
    
    # Add metadata columns
    spec_out.attrs['bary_corr_kms'] = bary_kms
    spec_out.attrs['emp_corr_kms'] = emp_kms
    spec_out.attrs['total_corr_kms'] = total_corr_kms
    
    # Save to output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    out_file = output_dir / f'{gaia_id}_magellan.csv'
    spec_out.to_csv(out_file, index=False)
    
    result['status'] = 'saved'
    result['output_file'] = str(out_file)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Pre-process spectra with wavelength corrections",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview corrections without saving")
    parser.add_argument("--gaia-id", type=int, default=None,
                        help="Process single spectrum by Gaia ID")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: spectra_corrected/)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite original spectra (CAUTION)")
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(__file__).parent
    spectra_dir = base_dir / 'spectra'
    
    if args.overwrite:
        output_dir = spectra_dir
        print("⚠️  WARNING: Overwriting original spectra!")
    elif args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = base_dir / 'spectra_corrected'
    
    # Load corrections
    print("Loading corrections...")
    bary_corr, cal_df = load_corrections()
    print(f"  {len(bary_corr)} barycentric corrections")
    print(f"  {len(cal_df)} calibrated spectra")
    
    # Filter to single spectrum if specified
    if args.gaia_id:
        cal_df = cal_df[cal_df['gaia_id'] == args.gaia_id]
        if len(cal_df) == 0:
            print(f"Error: Gaia ID {args.gaia_id} not in good calibration list")
            return
    
    # Process spectra
    print(f"\nProcessing {len(cal_df)} spectra...")
    if args.dry_run:
        print("(DRY RUN - not saving)")
    print()
    
    results = []
    for _, row in cal_df.iterrows():
        gaia_id = int(row['gaia_id'])
        result = process_spectrum(gaia_id, bary_corr, row, spectra_dir, output_dir, args.dry_run)
        results.append(result)
        
        if not args.dry_run and result['status'] == 'saved':
            print(f"  ✓ {gaia_id} (Pair {row['pair_id']}{row['component']}): {result['total_corr_kms']:+.2f} km/s")
    
    # Summary
    saved = sum(1 for r in results if r['status'] == 'saved')
    missing = sum(1 for r in results if r['status'] == 'missing')
    
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    if args.dry_run:
        print(f"  Dry run: {len(results)} spectra would be processed")
    else:
        print(f"  Saved: {saved}")
        if missing > 0:
            print(f"  Missing: {missing}")
        print(f"  Output: {output_dir}")
    print(f"{'='*50}")
    
    # Save correction log
    if not args.dry_run and saved > 0:
        log_file = output_dir / 'correction_log.csv'
        log_df = pd.DataFrame(results)
        log_df.to_csv(log_file, index=False)
        print(f"\nCorrection log saved to: {log_file}")


if __name__ == "__main__":
    main()

